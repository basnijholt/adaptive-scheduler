"""The DatabaseManager."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import zmq
import zmq.asyncio
import zmq.ssh

from adaptive_scheduler.utils import (
    _deserialize,
    _now,
    _serialize,
    cloudpickle_learners,
)

from .base_manager import BaseManager
from .common import log

if TYPE_CHECKING:
    from collections.abc import Callable

    import adaptive

    from adaptive_scheduler.scheduler import BaseScheduler


ctx = zmq.asyncio.Context()
FnameType = str | Path | list[str] | list[Path]
FnamesTypes = list[str] | list[Path] | list[list[str]] | list[list[Path]]


class JobIDExistsInDbError(Exception):
    """Raised when a job id already exists in the database."""


def _ensure_str(
    fnames: str | Path | FnamesTypes,
) -> str | list[str] | list[list[str]]:
    """Make sure that `pathlib.Path`s are converted to strings."""
    if isinstance(fnames, str | Path):
        return str(fnames)

    if isinstance(fnames, list | tuple):
        if len(fnames) == 0:
            return []  # type: ignore[return-value]
        if isinstance(fnames[0], str | Path):
            return [str(f) for f in fnames]
        if isinstance(fnames[0], list):
            return [[str(f) for f in sublist] for sublist in fnames]  # type: ignore[union-attr]
    msg = (
        "Invalid input: expected a  string/Path, or list of"
        " strings/Paths, a list of lists of strings/Paths."
    )
    raise ValueError(msg)


@dataclass
class _DBEntry:
    fname: str | list[str]
    job_id: str | None = None
    is_pending: bool = False
    is_done: bool = False
    log_fname: str | None = None
    job_name: str | None = None
    output_logs: list[str] = field(default_factory=list)
    start_time: float | None = None
    depends_on: list[int] = field(default_factory=list)


class SimpleDatabase:
    def __init__(self, db_fname: str | Path, *, clear_existing: bool = False) -> None:
        self.db_fname = Path(db_fname)
        self._data: list[_DBEntry] = []
        self._meta: dict[str, Any] = {}

        if self.db_fname.exists():
            if clear_existing:
                self.db_fname.unlink()
            else:
                with self.db_fname.open() as f:
                    raw_data = json.load(f)
                    self._data = [_DBEntry(**entry) for entry in raw_data["data"]]

    def all(self) -> list[_DBEntry]:
        return self._data

    def insert_multiple(self, entries: list[_DBEntry]) -> None:
        self._data.extend(entries)
        self._save()

    def update(self, update_dict: dict, indices: list[int] | None = None) -> None:
        for index, entry in enumerate(self._data):
            if indices is None or index in indices:
                for key, value in update_dict.items():
                    assert hasattr(entry, key)
                    setattr(entry, key, value)
        self._save()

    def count(self, condition: Callable[[_DBEntry], bool]) -> int:
        return sum(1 for entry in self._data if condition(entry))

    def get(self, condition: Callable[[_DBEntry], bool]) -> _DBEntry | None:
        for entry in self._data:
            if condition(entry):
                return entry
        return None

    def get_all(
        self,
        condition: Callable[[_DBEntry], bool],
    ) -> list[tuple[int, _DBEntry]]:
        return [(i, entry) for i, entry in enumerate(self._data) if condition(entry)]

    def contains(self, condition: Callable[[_DBEntry], bool]) -> bool:
        return any(condition(entry) for entry in self._data)

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(entry) for entry in self._data]

    def _save(self) -> None:
        with self.db_fname.open("w") as f:
            json.dump({"data": self.as_dicts(), "meta": self._meta}, f, indent=4)

    def dependencies_satisfied(self, entry: _DBEntry) -> bool:
        return all(self._data[i].is_done for i in entry.depends_on)


class DatabaseManager(BaseManager):
    """Database manager.

    Parameters
    ----------
    url
        The url of the database manager, with the format
        ``tcp://ip_of_this_machine:allowed_port.``. Use `get_allowed_url`
        to get a `url` that will work.
    scheduler
        A scheduler instance from `adaptive_scheduler.scheduler`.
    db_fname
        Filename of the database, e.g. 'running.json'.
    learners
        List of `learners` corresponding to `fnames`.
    fnames
        List of `fnames` corresponding to `learners`.
    dependencies
        Dictionary of dependencies, e.g., ``{1: [0]}`` means that the ``learners[1]``
        depends on the ``learners[0]``. This means that the ``learners[1]`` will only
        start when the ``learners[0]`` is done.
    overwrite_db
        Overwrite the existing database upon starting.
    initializers
        List of functions that are called before the job starts, can populate
        a cache.

    Attributes
    ----------
    failed : list
        A list of entries that have failed and have been removed from the database.

    """

    def __init__(
        self,
        url: str,
        scheduler: BaseScheduler,
        db_fname: str | Path,
        learners: list[adaptive.BaseLearner],
        fnames: FnamesTypes,
        *,
        dependencies: dict[int, list[int]] | None = None,
        overwrite_db: bool = True,
        initializers: list[Callable[[], None]] | None = None,
    ) -> None:
        super().__init__()
        self.url = url
        self.scheduler = scheduler
        self.db_fname = Path(db_fname)
        self.learners = learners
        self.fnames = fnames
        self.dependencies = dependencies or {}
        self.overwrite_db = overwrite_db
        self.initializers = initializers

        self._last_reply: str | list[str] | Exception | None = None
        self._last_request: tuple[str, ...] | None = None
        self.failed: list[dict[str, Any]] = []
        self._pickling_time: float | None = None
        self._total_learner_size: int | None = None
        self._db: SimpleDatabase | None = None

    def _setup(self) -> None:
        if self.db_fname.exists() and not self.overwrite_db:
            return
        self.create_empty_db()
        self._total_learner_size, self._pickling_time = cloudpickle_learners(
            self.learners,
            self.fnames,
            initializers=self.initializers,
            with_progress_bar=True,
        )

    def update(self, queue: dict[str, dict[str, str]] | None = None) -> None:
        """If the ``job_id`` isn't running anymore, replace it with None."""
        if self._db is None:
            return
        if queue is None:
            queue = self.scheduler.queue(me_only=True)
        job_names_in_queue = [x["job_name"] for x in queue.values()]
        failed = self._db.get_all(
            lambda e: e.job_name is not None and e.job_name not in job_names_in_queue,  # type: ignore[operator]
        )
        self.failed.extend([asdict(entry) for _, entry in failed])
        indices = [index for index, _ in failed]
        self._db.update(
            {"job_id": None, "job_name": None, "is_pending": False},
            indices,
        )

    def n_done(self) -> int:
        """Return the number of jobs that are done."""
        if self._db is None:
            return 0
        return self._db.count(lambda e: e.is_done)

    def is_done(self) -> bool:
        """Return True if all jobs are done."""
        return self.n_done() == len(self.fnames)

    def create_empty_db(self) -> None:
        """Create an empty database.

        It keeps track of ``fname -> (job_id, is_done, log_fname, job_name)``.
        """
        deps = self.dependencies
        entries: list[_DBEntry] = [
            _DBEntry(fname=fname, depends_on=deps.get(i, []))  # type: ignore[arg-type]
            for i, fname in enumerate(_ensure_str(self.fnames))
        ]
        if self.db_fname.exists():
            self.db_fname.unlink()
        self._db = SimpleDatabase(self.db_fname)
        self._db.insert_multiple(entries)

    def as_dicts(self) -> list[dict[str, str]]:
        """Return the database as a list of dictionaries."""
        if self._db is None:
            return []
        return self._db.as_dicts()

    def as_df(self) -> pd.DataFrame:
        """Return the database as a `pandas.DataFrame`."""
        return pd.DataFrame(self.as_dicts())

    def _output_logs(self, job_id: str, job_name: str) -> list[Path]:
        job_id = self.scheduler.sanatize_job_id(job_id)
        output_fnames = self.scheduler.output_fnames(job_name)
        return [
            f.with_name(f.name.replace(self.scheduler._JOB_ID_VARIABLE, job_id))
            for f in output_fnames
        ]

    def _choose_fname(self) -> tuple[int, str | list[str] | None]:
        assert self._db is not None
        entry = self._db.get(
            lambda e: e.job_id is None
            and not e.is_done
            and not e.is_pending
            and self._db.dependencies_satisfied(e),  # type: ignore[union-attr]
        )
        if all(e.is_done for e in self._db.all()):
            msg = "Requested a new job but no more learners to run in the database."
            raise RuntimeError(msg)
        if entry is None:
            # Currently, we cannot schedule any more jobs, because we're waiting
            # for dependencies to be satisfied.
            return -1, None
        log.debug("choose fname", entry=entry)
        index = self._db.all().index(entry)
        return index, _ensure_str(entry.fname)  # type: ignore[return-value]

    def _confirm_submitted(self, index: int, job_name: str) -> None:
        assert self._db is not None
        self._db.update(
            {
                "job_name": job_name,
                "is_pending": True,
            },
            indices=[index],
        )

    def _start_request(
        self,
        job_id: str,
        log_fname: str,
        job_name: str,
    ) -> str | list[str] | None:
        assert self._db is not None
        if self._db.contains(lambda e: e.job_id == job_id):
            entry = self._db.get(lambda e: e.job_id == job_id)
            assert entry is not None
            fname = entry.fname  # already running
            msg = (
                f"The job_id {job_id} already exists in the database and "
                f"runs {fname}. You might have forgotten to use the "
                "`if __name__ == '__main__': ...` idiom in your code. Read the "
                "warning in the [mpi4py](https://bit.ly/2HAk0GG) documentation.",
            )
            raise JobIDExistsInDbError(msg)
        entry = self._db.get(lambda e: e.job_name == job_name and e.is_pending)
        log.debug("choose fname", entry=entry)
        if entry is None:
            return None
        index = self._db.all().index(entry)
        self._db.update(
            {
                "job_id": job_id,
                "log_fname": log_fname,
                "output_logs": _ensure_str(self._output_logs(job_id, job_name)),
                "start_time": _now(),
                "is_pending": False,
            },
            indices=[index],
        )
        return _ensure_str(entry.fname)  # type: ignore[return-value]

    def _stop_request(self, fname: str | list[str] | Path | list[Path]) -> None:
        fname_str = _ensure_str(fname)
        reset = {"job_id": None, "is_done": True, "job_name": None, "is_pending": False}
        assert self._db is not None
        entry_indices = [index for index, _ in self._db.get_all(lambda e: e.fname == fname_str)]
        self._db.update(reset, entry_indices)

    def _stop_requests(self, fnames: FnamesTypes) -> None:
        # Same as `_stop_request` but optimized for processing many `fnames` at once
        assert self._db is not None
        fnames_str = {str(fname) for fname in _ensure_str(fnames)}
        reset = {"job_id": None, "is_done": True, "job_name": None, "is_pending": False}
        entry_indices = [
            index for index, _ in self._db.get_all(lambda e: str(e.fname) in fnames_str)
        ]
        self._db.update(reset, entry_indices)

    def _dispatch(
        self,
        request: tuple[str, str | list[str]] | tuple[str],
    ) -> str | list[str] | Exception | None:
        request_type, *request_arg = request
        log.debug("got a request", request=request)
        try:
            if request_type == "start":
                # workers send us their slurm ID for us to fill in
                job_id, log_fname, job_name = request_arg
                # give the worker a job and send back the fname to the worker
                fname = self._start_request(job_id, log_fname, job_name)  # type: ignore[arg-type]
                if fname is None:
                    # This should never happen because the _manage co-routine
                    # should have stopped the workers before this happens.
                    msg = "No more learners to run in the database."
                    raise RuntimeError(msg)  # noqa: TRY301
                log.debug(
                    "choose a fname",
                    fname=fname,
                    job_id=job_id,
                    log_fname=log_fname,
                    job_name=job_name,
                )
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
                    self._last_reply = self._dispatch(self._last_request)  # type: ignore[arg-type]
                    await socket.send_serialized(self._last_reply, _serialize)
                if self.is_done():
                    break
        finally:
            socket.close()
