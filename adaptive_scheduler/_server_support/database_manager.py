"""The DatabaseManager."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Union

import zmq
import zmq.asyncio
import zmq.ssh
from tinydb import Query, TinyDB

from adaptive_scheduler.utils import (
    _deserialize,
    _now,
    _serialize,
    cloudpickle_learners,
)

from .base_manager import BaseManager
from .common import log

if TYPE_CHECKING:
    import adaptive

    from adaptive_scheduler.scheduler import BaseScheduler

ctx = zmq.asyncio.Context()

FnamesTypes = Union[List[str], List[Path], List[List[str]], List[List[Path]]]


class JobIDExistsInDbError(Exception):
    """Raised when a job id already exists in the database."""


def _ensure_str(
    fnames: str | Path | FnamesTypes,
) -> str | list[str] | list[list[str]]:
    """Make sure that `pathlib.Path`s are converted to strings."""
    if isinstance(fnames, (str, Path)):
        return str(fnames)

    if isinstance(fnames, (list, tuple)):
        if len(fnames) == 0:
            return []  # type: ignore[return-value]
        if isinstance(fnames[0], (str, Path)):
            return [str(f) for f in fnames]
        if isinstance(fnames[0], list):
            return [[str(f) for f in sublist] for sublist in fnames]  # type: ignore[union-attr]
    msg = (
        "Invalid input: expected a  string/Path, or list of"
        " strings/Paths, a list of lists of strings/Paths."
    )
    raise ValueError(msg)


class DatabaseManager(BaseManager):
    """Database manager.

    Parameters
    ----------
    url : str
        The url of the database manager, with the format
        ``tcp://ip_of_this_machine:allowed_port.``. Use `get_allowed_url`
        to get a `url` that will work.
    scheduler : `~adaptive_scheduler.scheduler.BaseScheduler`
        A scheduler instance from `adaptive_scheduler.scheduler`.
    db_fname : str
        Filename of the database, e.g. 'running.json'.
    learners : list of `adaptive.BaseLearner` isinstances
        List of `learners` corresponding to `fnames`.
    fnames : list
        List of `fnames` corresponding to `learners`.
    overwrite_db : bool, default: True
        Overwrite the existing database upon starting.

    Attributes
    ----------
    failed : list
        A list of entries that have failed and have been removed from the database.
    """

    def __init__(  # noqa: PLR0913
        self,
        url: str,
        scheduler: BaseScheduler,
        db_fname: str | Path,
        learners: list[adaptive.BaseLearner],
        fnames: FnamesTypes,
        *,
        overwrite_db: bool = True,
    ) -> None:
        super().__init__()
        self.url = url
        self.scheduler = scheduler
        self.db_fname = Path(db_fname)
        self.learners = learners
        self.fnames = fnames
        self.overwrite_db = overwrite_db

        self.defaults: dict[str, Any] = {
            "job_id": None,
            "is_done": False,
            "log_fname": None,
            "job_name": None,
            "output_logs": [],
            "start_time": None,
        }

        self._last_reply: str | Exception | None = None
        self._last_request: tuple[str, ...] | None = None
        self.failed: list[dict[str, Any]] = []
        self._pickling_time: float | None = None
        self._total_learner_size: int | None = None

    def _setup(self) -> None:
        if self.db_fname.exists() and not self.overwrite_db:
            return
        self.create_empty_db()
        self._total_learner_size, self._pickling_time = cloudpickle_learners(
            self.learners,
            self.fnames,
            with_progress_bar=True,
        )

    def update(self, queue: dict[str, dict[str, str]] | None = None) -> None:
        """If the ``job_id`` isn't running anymore, replace it with None."""
        if queue is None:
            queue = self.scheduler.queue(me_only=True)

        with TinyDB(self.db_fname) as db:
            failed = [
                entry
                for entry in db.all()
                if (entry["job_id"] is not None) and (entry["job_id"] not in queue)
            ]
            self.failed.extend(failed)
            doc_ids = [e.doc_id for e in failed]
            db.update({"job_id": None, "job_name": None}, doc_ids=doc_ids)

    def n_done(self) -> int:
        entry = Query()
        with TinyDB(self.db_fname) as db:
            return db.count(entry.is_done == True)  # noqa: E712

    def create_empty_db(self) -> None:
        """Create an empty database.

        It keeps track of ``fname -> (job_id, is_done, log_fname, job_name)``.
        """
        entries = [
            dict(fname=_ensure_str(fname), **self.defaults) for fname in self.fnames
        ]
        if self.db_fname.exists():
            self.db_fname.unlink()
        with TinyDB(self.db_fname) as db:
            db.insert_multiple(entries)

    def as_dicts(self) -> list[dict[str, str]]:
        with TinyDB(self.db_fname) as db:
            return db.all()

    def _output_logs(self, job_id: str, job_name: str) -> list[Path]:
        job_id = self.scheduler.sanatize_job_id(job_id)
        output_fnames = self.scheduler.output_fnames(job_name)
        return [
            f.with_name(f.name.replace(self.scheduler._JOB_ID_VARIABLE, job_id))
            for f in output_fnames
        ]

    def _start_request(self, job_id: str, log_fname: str, job_name: str) -> str | None:
        entry = Query()
        with TinyDB(self.db_fname) as db:
            if db.contains(entry.job_id == job_id):
                entry = db.get(entry.job_id == job_id)
                fname = entry["fname"]  # already running
                msg = (
                    f"The job_id {job_id} already exists in the database and "
                    f"runs {fname}. You might have forgotten to use the "
                    "`if __name__ == '__main__': ...` idom in your code. Read the "
                    "warning in the [mpi4py](https://bit.ly/2HAk0GG) documentation.",
                )
                raise JobIDExistsInDbError(msg)
            entry = db.get(
                (entry.job_id == None) & (entry.is_done == False),  # noqa: E711,E712
            )
            log.debug("choose fname", entry=entry)
            if entry is None:
                return None
            db.update(
                {
                    "job_id": job_id,
                    "log_fname": log_fname,
                    "job_name": job_name,
                    "output_logs": _ensure_str(self._output_logs(job_id, job_name)),
                    "start_time": _now(),
                },
                doc_ids=[entry.doc_id],
            )
        return entry["fname"]

    def _stop_request(self, fname: str | list[str] | Path | list[Path]) -> None:
        fname_str = _ensure_str(fname)
        entry = Query()
        with TinyDB(self.db_fname) as db:
            reset = {"job_id": None, "is_done": True, "job_name": None}
            assert (
                db.get(entry.fname == fname_str) is not None
            )  # make sure the entry exists
            db.update(reset, entry.fname == fname_str)

    def _stop_requests(self, fnames: FnamesTypes) -> None:
        # Same as `_stop_request` but optimized for processing many `fnames` at once
        fnames_str = {str(fname) for fname in _ensure_str(fnames)}
        with TinyDB(self.db_fname) as db:
            reset = {"job_id": None, "is_done": True, "job_name": None}
            doc_ids = [e.doc_id for e in db.all() if str(e["fname"]) in fnames_str]
            db.update(reset, doc_ids=doc_ids)

    def _dispatch(self, request: tuple[str, ...]) -> str | Exception | None:
        request_type, *request_arg = request
        log.debug("got a request", request=request)
        try:
            if request_type == "start":
                # workers send us their slurm ID for us to fill in
                job_id, log_fname, job_name = request_arg
                kwargs = {
                    "job_id": job_id,
                    "log_fname": log_fname,
                    "job_name": job_name,
                }
                # give the worker a job and send back the fname to the worker
                fname = self._start_request(**kwargs)
                if fname is None:
                    msg = "No more learners to run in the database."
                    raise RuntimeError(msg)  # noqa: TRY301
                log.debug("choose a fname", fname=fname, **kwargs)
                return fname
            if request_type == "stop":
                fname = request_arg[0]  # workers send us the fname they were given
                log.debug("got a stop request", fname=fname)
                self._stop_request(fname)  # reset the job_id to None
                return None
        except Exception as e:  # noqa: BLE001
            return e
        msg = f"Unknown request type: {request_type}"
        raise ValueError(msg)

    async def _manage(self) -> None:
        """Database manager co-routine.

        Returns
        -------
        coroutine
        """
        log.debug("started database")
        socket = ctx.socket(zmq.REP)
        socket.bind(self.url)
        try:
            while True:
                try:
                    self._last_request = await socket.recv_serialized(_deserialize)
                except zmq.error.Again:
                    log.exception(
                        "socket.recv_serialized failed in the DatabaseManager"
                        " with `zmq.error.Again`.",
                    )
                except pickle.UnpicklingError as e:
                    if r"\x03" in str(e):
                        # Empty frame received.
                        # TODO: not sure why this happens
                        pass
                    else:
                        log.exception(
                            "socket.recv_serialized failed in the DatabaseManager"
                            " with `pickle.UnpicklingError` in _deserialize.",
                        )
                else:
                    assert self._last_request is not None  # for mypy
                    self._last_reply = self._dispatch(self._last_request)
                    await socket.send_serialized(self._last_reply, _serialize)
        finally:
            socket.close()
