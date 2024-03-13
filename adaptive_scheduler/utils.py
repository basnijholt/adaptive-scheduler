"""Utility functions for adaptive_scheduler."""

from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import inspect
import math
import os
import pickle
import platform
import random
import shutil
import tempfile
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, timezone
from inspect import signature
from itertools import chain
from multiprocessing import Manager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Literal,
    Union,
)
from typing import (
    get_args as get_type_args,
)

import adaptive
import cloudpickle
import numpy as np
import pandas as pd
import toolz
from adaptive.notebook_integration import in_ipynb
from rich.console import Console
from rich.progress import (
    Progress,
    TimeElapsedColumn,
    get_console,
)
from tqdm import tqdm, tqdm_notebook

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from multiprocessing.managers import ListProxy

    from rich.progress import TaskID

console = Console()

MAX_LINE_LENGTH = 100
_NONE_RETURN_STR = "__ReturnsNone__"
FnamesTypes = Union[List[str], List[Path], List[List[str]], List[List[Path]]]

LOKY_START_METHODS = Literal[
    "loky",
    "loky_int_main",
    "spawn",
    "fork",
    "forkserver",
]

EXECUTOR_TYPES = Literal[
    "mpi4py",
    "ipyparallel",
    "dask-mpi",
    "process-pool",
    "loky",
    "sequential",
]
GoalTypes = Union[
    Callable[[adaptive.BaseLearner], bool],
    int,
    float,
    datetime,
    timedelta,
    None,
]


def shuffle_list(*lists: list, seed: int | None = 0) -> zip:
    """Shuffle multiple lists in the same order."""
    combined = list(zip(*lists))
    random.Random(seed).shuffle(combined)
    return zip(*combined)


def hash_anything(x: Any) -> str:
    """Hash anything."""
    try:
        return hashlib.md5(x).hexdigest()  # noqa: S324
    except TypeError:
        return hashlib.md5(pickle.dumps(x)).hexdigest()  # noqa: S324


def split(seq: Iterable, n_parts: int) -> Iterable[tuple]:
    """Split up a sequence into ``n_parts``.

    Parameters
    ----------
    seq
        A list or other iterable that has to be split up.
    n_parts
        The sequence will be split up in this many parts.

    """
    lst = list(seq)
    n = math.ceil(len(lst) / n_parts)
    return toolz.partition_all(n, lst)


def split_in_balancing_learners(
    learners: list[adaptive.BaseLearner],
    fnames: list[str] | list[Path],
    n_parts: int,
    strategy: str = "npoints",
) -> tuple[list[adaptive.BaseLearner], list[list[str]] | list[list[Path]]]:
    r"""Split a list of learners and fnames into `adaptive.BalancingLearner`\s.

    Parameters
    ----------
    learners
        List of learners.
    fnames
        List of filenames.
    n_parts
        Total number of `~adaptive.BalancingLearner`\s.
    strategy
        Learning strategy of the `~adaptive.BalancingLearner`.

    Returns
    -------
    new_learners, new_fnames

    """
    new_learners = []
    new_fnames = []
    for x in split(zip(learners, fnames), n_parts):
        learners_part, fnames_part = zip(*x)
        learner = adaptive.BalancingLearner(learners_part, strategy=strategy)
        new_learners.append(learner)
        new_fnames.append(list(fnames_part))
    return new_learners, new_fnames


def split_sequence_learner(
    big_learner: adaptive.SequenceLearner,
    n_learners: int,
    folder: str | Path = "",
) -> tuple[list[adaptive.SequenceLearner], list[str]]:
    r"""Split a sinlge `~adaptive.SequenceLearner` into many.

    Split into mutiple `adaptive.SequenceLearner`\s
    (with the data loaded) and fnames.

    See also `split_sequence_in_sequence_learners`.

    Parameters
    ----------
    big_learner
        A `~adaptive.SequenceLearner` instance
    n_learners
        Total number of `~adaptive.SequenceLearner`\s.
    folder
        Folder to prepend to fnames.

    Returns
    -------
    new_learners
        List of `~adaptive.SequenceLearner`\s.
    new_fnames
        List of str based on a hash of the sequence.

    """
    new_learners, new_fnames = split_sequence_in_sequence_learners(
        function=big_learner._original_function,
        sequence=big_learner.sequence,
        n_learners=n_learners,
        folder=folder,
    )
    # Load the new learners with data
    index_parts = split(range(len(big_learner.sequence)), n_learners)
    for small_learner, part in zip(new_learners, index_parts):
        for i_small, i_big in enumerate(part):
            y = big_learner.data.get(i_big)
            if y is None:
                continue
            x = i_small, big_learner.sequence[i_big]
            small_learner.tell(x, y)

    return new_learners, new_fnames


def split_sequence_in_sequence_learners(
    function: Callable[[Any], Any],
    sequence: Sequence[Any],
    n_learners: int,
    folder: str | Path = "",
) -> tuple[list[adaptive.SequenceLearner], list[str]]:
    r"""Split a sequenceinto `adaptive.SequenceLearner`\s and fnames.

    Parameters
    ----------
    function
        Function for `adaptive.SequenceLearner`\s.
    sequence
        The sequence to split into ``n_learners``.
    n_learners
        Total number of `~adaptive.SequenceLearner`\s.
    folder
        Folder to prepend to fnames.

    Returns
    -------
    new_learners
        List of `~adaptive.SequenceLearner`\s.
    new_fnames
        List of str based on a hash of the sequence.

    """
    folder = Path(folder)
    new_learners = []
    new_fnames = []
    for sequence_part in split(sequence, n_learners):
        learner = adaptive.SequenceLearner(function, sequence_part)
        new_learners.append(learner)
        hsh = hash_anything((sequence_part[0], len(sequence_part)))
        fname = folder / f"{hsh}.pickle"
        new_fnames.append(str(fname))
    return new_learners, new_fnames


def combine_sequence_learners(
    learners: list[adaptive.SequenceLearner],
    big_learner: adaptive.SequenceLearner | None = None,
) -> adaptive.SequenceLearner:
    r"""Combine several `~adaptive.SequenceLearner`\s into a single one.

    Also copy over the data.

    Assumes that all ``learners`` take the same function.

    Parameters
    ----------
    learners
        List of `~adaptive.SequenceLearner`\s.
    big_learner
        A learner to load, if None, a new learner will be generated.

    Returns
    -------
    adaptive.SequenceLearner
        Big `~adaptive.SequenceLearner` with data from ``learners``.

    """
    if big_learner is None:
        big_sequence: list[Any] = list(
            chain.from_iterable(learner.sequence for learner in learners),
        )
        big_learner = adaptive.SequenceLearner(
            learners[0]._original_function,
            sequence=big_sequence,
        )

    cnt = 0
    for learner in learners:
        for i, key in enumerate(learner.sequence):
            if i in learner.data:
                x = cnt, key
                y = learner.data[i]
                big_learner.tell(x, y)
            cnt += 1
    return big_learner


def copy_from_sequence_learner(
    learner_from: adaptive.SequenceLearner,
    learner_to: adaptive.SequenceLearner,
) -> None:
    """Copy the data from a `~adaptive.SequenceLearner` into a different one.

    Parameters
    ----------
    learner_from
        Learner to take the data from.
    learner_to
        Learner to tell the data to.

    """
    mapping = {
        hash_anything(learner_from.sequence[i]): v for i, v in learner_from.data.items()
    }
    for i, key in enumerate(learner_to.sequence):
        hsh = hash_anything(key)
        if hsh in mapping:
            v = mapping[hsh]
            learner_to.tell((i, key), v)


def _get_npoints(learner: adaptive.BaseLearner) -> int | None:
    with suppress(AttributeError):
        return learner.npoints
    with suppress(AttributeError):
        # If the Learner is a BalancingLearner
        return sum(learner.npoints for learner in learner.learners)
    return None


def _progress(
    seq: Iterable[Any],
    with_progress_bar: bool = True,  # noqa: FBT001, FBT002
    desc: str = "",
) -> Iterable | tqdm:
    if not with_progress_bar:
        return seq
    if in_ipynb():
        return tqdm_notebook(list(seq), desc=desc)
    return tqdm(list(seq), desc=desc)


def combo_to_fname(
    combo: dict[str, Any],
    folder: str | Path | None = None,
    ext: str = ".pickle",
) -> Path:
    """Converts a dict into a human readable filename."""
    console.log(
        "Use `adaptive_scheduler.utils.combo2fname` instead of `combo_to_fname`.",
    )
    fname = "__".join(f"{k}_{v}" for k, v in combo.items()) + ext
    if folder is None:
        return Path(fname)
    return Path(folder) / fname


def combo2fname(
    combo: dict[str, Any],
    folder: str | Path | None = None,
    ext: str = ".pickle",
    sig_figs: int = 8,
) -> Path:
    """Converts a dict into a human readable filename.

    Improved version of `combo_to_fname`.
    """
    name_parts = [f"{k}_{maybe_round(v, sig_figs)}" for k, v in sorted(combo.items())]
    fname = Path("__".join(name_parts) + ext)
    if folder is None:
        return fname
    return folder / fname


def add_constant_to_fname(
    combo: dict[str, Any],
    constant: dict[str, Any],
    *,
    folder: str | Path | None = None,
    ext: str = ".pickle",
    sig_figs: int = 8,
) -> tuple[Path, Path]:
    """Construct old and new filename based on a combo.

    Assumes `combo2fname` has been used to construct the old filename.
    Adds `constant` dict to the `combo` and returns the new filename too.

    Returns a tuple of ``old_fname`` and ``new_fname``.
    """
    for k in constant:
        combo.pop(k, None)
    old_fname = combo2fname(combo, folder, ext, sig_figs)
    combo.update(constant)
    new_fname = combo2fname(combo, folder, ext, sig_figs)
    return old_fname, new_fname


def maybe_round(x: Any, sig_figs: int) -> Any:
    """Round to specified number of sigfigs if x is a float or complex."""
    rnd = functools.partial(round_sigfigs, sig_figs=sig_figs)

    def try_is_nan_inf(x: Any) -> bool:
        try:
            return np.isnan(x) or np.isinf(x)
        except Exception:  # noqa: BLE001
            return False

    if try_is_nan_inf(x):
        return x
    if isinstance(x, float):
        return rnd(x)
    if isinstance(x, complex):
        return complex(rnd(x.real), rnd(x.imag))
    return x


def round_sigfigs(num: float, sig_figs: int) -> float:
    """Round to specified number of sigfigs.

    From
    http://code.activestate.com/recipes/578114-round-number-to-specified-number-of-significant-di/
    """
    num = float(num)
    if num != 0:
        return round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs - 1)))
    return 0.0  # Can't take the log of 0


def _remove_or_move_files(
    fnames: Sequence[str] | set[str] | Sequence[Path] | set[Path],
    *,
    with_progress_bar: bool = True,
    move_to: str | Path | None = None,
    desc: str | None = None,
) -> None:
    """Remove files by filename.

    Parameters
    ----------
    fnames
        List of filenames.
    with_progress_bar
        Display a progress bar using `tqdm`.
    move_to
        Move the file to a different directory.
        If None the file is removed.
    desc
        Description of the progressbar.

    """
    n_failed = 0
    for fname in _progress(fnames, with_progress_bar, desc or "Removing files"):
        fname = Path(fname)  # noqa: PLW2901
        try:
            if move_to is None:
                fname.unlink()
            else:
                move_to = Path(move_to)
                move_to.mkdir(parents=True, exist_ok=True)
                src = fname.resolve()
                dst = (move_to / src.name).resolve()
                shutil.move(src, dst)  # overwrites old files
        except Exception:  # noqa: BLE001
            n_failed += 1

    if n_failed:
        warnings.warn(
            f"Failed to remove (or move) {n_failed}/{len(fnames)} files.",
            stacklevel=2,
        )


def load_parallel(
    learners: list[adaptive.BaseLearner],
    fnames: list[str] | list[Path],
    *,
    with_progress_bar: bool = True,
    max_workers: int | None = None,
) -> None:
    r"""Load a sequence of learners in parallel.

    Parameters
    ----------
    learners
        The learners to be loaded.
    fnames
        A list of filenames corresponding to `learners`.
    with_progress_bar
        Display a progress bar using `tqdm`.
    max_workers
        The maximum number of parallel threads when loading the data.
        If ``None``, use the maximum number of threads that is possible.

    """

    def load(learner: adaptive.BaseLearner, fname: str) -> None:
        learner.load(fname)

    with ThreadPoolExecutor(max_workers) as ex:
        iterator = zip(learners, fnames)
        pbar = _progress(iterator, with_progress_bar, "Submitting loading tasks")
        futs = [ex.submit(load, *args) for args in pbar]
        for fut in _progress(futs, with_progress_bar, "Finishing loading"):
            fut.result()


def save_parallel(
    learners: list[adaptive.BaseLearner],
    fnames: list[str] | list[Path],
    *,
    with_progress_bar: bool = True,
) -> None:
    r"""Save a sequence of learners in parallel.

    Parameters
    ----------
    learners
        The learners to be saved.
    fnames
        A list of filenames corresponding to `learners`.
    with_progress_bar
        Display a progress bar using `tqdm`.

    """

    def save(learner: adaptive.BaseLearner, fname: str) -> None:
        learner.save(fname)

    with ThreadPoolExecutor() as ex:
        iterator = zip(learners, fnames)
        pbar = _progress(iterator, with_progress_bar, "Submitting saving tasks")
        futs = [ex.submit(save, *args) for args in pbar]
        for fut in _progress(futs, with_progress_bar, "Finishing saving"):
            fut.result()


def _print_same_line(msg: str, *, new_line_end: bool = False) -> None:
    msg = msg.strip()
    global MAX_LINE_LENGTH
    MAX_LINE_LENGTH = max(len(msg), MAX_LINE_LENGTH)
    empty_space = max(MAX_LINE_LENGTH - len(msg), 0) * " "
    print(msg + empty_space, end="\r" if not new_line_end else "\n")


def _wait_for_successful_ipyparallel_client_start(
    client: Any,
    n: int,
    timeout: int,
) -> Any:
    from ipyparallel.error import NoEnginesRegistered

    n_engines_old = 0
    for t in range(timeout):
        n_engines = len(client)
        with suppress(NoEnginesRegistered):
            # This can happen, we just need to wait a little longer.
            dview = client[:]
        msg = f"Connected to {n_engines} out of {n} engines after {t} seconds."
        _print_same_line(msg, new_line_end=(n_engines_old != n_engines))
        if n_engines >= n:
            return dview
        n_engines_old = n_engines
        time.sleep(1)
    msg = f"Not all ({n_engines}/{n}) connected after {timeout} seconds."
    raise Exception(msg)  # noqa: TRY002


def connect_to_ipyparallel(
    n: int,
    profile: str,
    timeout: int = 300,
    folder: str | None = None,
    client_kwargs: dict | None = None,
) -> Any:
    """Connect to an `ipcluster` on the cluster headnode.

    Parameters
    ----------
    n
        Number of engines to be started.
    profile
        Profile name of IPython profile.
    timeout
        Time for which we try to connect to get all the engines.
    folder
        Folder that is added to the path of the engines, e.g. ``"~/Work/my_current_project"``.
    client_kwargs
        Keyword arguments passed to `ipyparallel.Client`.

    Returns
    -------
    client
        An IPyparallel client.

    """
    from ipyparallel import Client

    client = Client(profile=profile, **(client_kwargs or {}))
    dview = _wait_for_successful_ipyparallel_client_start(client, n, timeout)
    dview.use_dill()

    if folder is not None:
        console.print(f"Adding {folder} to path.")
        cmd = f"import sys, os; sys.path.append(os.path.expanduser('{folder}'))"
        dview.execute(cmd).result()

    return client


def _get_default_args(func: Callable) -> dict[str, str]:
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def log_exception(log: Any, msg: str, exception: Exception) -> None:
    """Log an exception with a message."""
    try:
        raise exception  # noqa: TRY301
    except Exception:
        log.exception(msg)


def _serialize(msg: Any) -> list:
    return [cloudpickle.dumps(msg)]


def _deserialize(frames: list) -> Any:
    try:
        return cloudpickle.loads(frames[0])
    except pickle.UnpicklingError as e:
        if r"\x03" in str(e):
            # Means that the frame is empty because it only contains an end of text char
            # `\x03  ^C    (End of text)`
            # TODO: Not sure why this happens.
            console.log(
                r"pickle.UnpicklingError in _deserialize: Received an empty frame (\x03).",
            )
            console.print_exception(show_locals=True)
        raise


class LRUCachedCallable:
    """Wraps a function to become cached.

    Parameters
    ----------
    function
    max_size
        Cache size of the LRU cache, by default 128.
    with_cloudpickle
        Use cloudpickle for storing the data in memory.

    """

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        max_size: int = 128,
        with_cloudpickle: bool = False,
    ) -> None:
        """Initialize the cache."""
        self.max_size = max_size
        self.function = function
        self._with_cloudpickle = with_cloudpickle
        self._signature = signature(self.function)
        if max_size == 0:
            return
        manager = Manager()
        self._cache_dict = manager.dict()
        self._cache_queue = manager.list()
        self._cache_lock = manager.Lock()

    def _get_from_cache(self, key: str) -> tuple[bool, Any]:
        """Get a value from the cache by key."""
        if self.max_size == 0:
            value = None
        with self._cache_lock:
            value = self._cache_dict.get(key)
            if value is not None:  # Move key to back of queue
                self._cache_queue.remove(key)
                self._cache_queue.append(key)
        if value is not None:
            found = True
            if value == _NONE_RETURN_STR:
                value = None
            elif self._with_cloudpickle:
                value = cloudpickle.loads(value)
        else:
            found = False
        return found, value

    def _insert_into_cache(self, key: str, value: Any) -> ListProxy[Any]:
        """Insert a key value pair into the cache."""
        if value is None:
            value = _NONE_RETURN_STR
        elif self._with_cloudpickle:
            value = cloudpickle.dumps(value)
        with self._cache_lock:
            cache_size = len(self._cache_queue)
            self._cache_dict[key] = value
            if cache_size < self.max_size:
                self._cache_queue.append(key)
            else:
                key_to_evict = self._cache_queue.pop(0)
                self._cache_dict.pop(key_to_evict)
                self._cache_queue.append(key)
            return self._cache_queue

    @property
    def cache_dict(self) -> dict:
        """Returns a copy of the cache."""
        return dict(self._cache_dict.items())

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function and cache the result."""
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        if self.max_size == 0:
            return self.function(*args, **kwargs)
        key = str(bound_args.arguments)
        found, value = self._get_from_cache(key)
        if found:
            return value
        ret = self.function(*args, **kwargs)
        self._insert_into_cache(key, ret)
        return ret


def shared_memory_cache(cache_size: int = 128) -> Callable:
    """Create a cache similar to `functools.lru_cache`.

    This will actually cache the return values of the function, whereas
    `functools.lru_cache` will pickle the decorated function each time
    with an empty cache.
    """

    def cache_decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        return functools.wraps(function)(
            LRUCachedCallable(function, max_size=cache_size),
        )

    return cache_decorator


def _prefix(
    fname: str | list[str] | Path | list[Path],
) -> str:
    if isinstance(fname, (tuple, list)):
        return f".{len(fname):08}_learners."
    if isinstance(fname, (str, Path)):
        return ".learner."
    msg = "Incorrect type for fname."
    raise TypeError(msg)


def fname_to_learner_fname(
    fname: str | list[str] | Path | list[Path],
) -> Path:
    """Convert a learner filename (data) to a filename used to cloudpickle the learner."""
    prefix = _prefix(fname)
    if isinstance(fname, (tuple, list)):
        fname = fname[0]
    p = Path(fname)
    new_name = f"{prefix}{p.stem}{p.suffix}"
    return p.with_name(new_name)


def fname_to_learner(
    fname: str | list[str] | Path | list[Path],
    *,
    return_initializer: bool = False,
) -> tuple[adaptive.BaseLearner, Callable[[], None] | None] | adaptive.BaseLearner:
    """Load a learner from a filename (based on cloudpickled learner)."""
    learner_name = fname_to_learner_fname(fname)
    with learner_name.open("rb") as f:
        learner, initializer = cloudpickle.load(f)
    if return_initializer:
        return learner, initializer
    return learner


def _ensure_folder_exists(
    fnames: FnamesTypes,
) -> None:
    if isinstance(fnames[0], (tuple, list)):
        for _fnames in fnames:
            assert isinstance(_fnames, (tuple, list))
            _ensure_folder_exists(_fnames)
    else:
        assert isinstance(fnames[0], (str, Path))
        folders = {Path(fname).parent for fname in fnames}  # type: ignore[arg-type]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)


def cloudpickle_learners(
    learners: list[adaptive.BaseLearner],
    fnames: FnamesTypes,
    *,
    initializers: list[Callable[[], None]] | None = None,
    with_progress_bar: bool = False,
    empty_copies: bool = True,
) -> tuple[int, float]:
    """Save a list of learners to disk using cloudpickle.

    Returns the total size of the saved files in bytes and the total time.
    """
    _ensure_folder_exists(fnames)
    total_filesize = 0
    t_start = time.time()
    if initializers is None:
        initializers = [None] * len(learners)  # type: ignore[list-item]
    for learner, fname, initializer in _progress(
        zip(learners, fnames, initializers),
        with_progress_bar,
        desc="Cloudpickling learners",
    ):
        fname_learner = fname_to_learner_fname(fname)
        if empty_copies:
            _at_least_adaptive_version("0.14.1", "empty_copies")
            learner = learner.new()  # noqa: PLW2901
        with fname_learner.open("wb") as f:
            # Make sure that the learner and initializer are ALWAYS pickled together.
            # Otherwise, the learner function and initializer might not use to the
            # same objects even when referring to the same objects in a namespace.
            cloudpickle.dump((learner, initializer), f)
        filesize = fname_learner.stat().st_size
        total_filesize += filesize
    total_time = time.time() - t_start
    return total_filesize, total_time


def fname_to_dataframe(
    fname: str | list[str] | Path | list[Path],
    format: str = "pickle",  # noqa: A002
) -> Path:
    """Convert a learner filename (data) to a filename is used to save the dataframe."""
    if format == "excel":
        format = "xlsx"  # noqa: A001
    if isinstance(fname, (tuple, list)):
        fname = fname[0]
    p = Path(fname)
    new_name = f"dataframe.{p.stem}.{format}"
    return p.with_name(new_name)


def save_dataframe(
    fname: str | list[str],
    *,
    format: _DATAFRAME_FORMATS = "pickle",  # noqa: A002
    save_kwargs: dict[str, Any] | None = None,
    expand_dicts: bool = True,
    atomically: bool = True,
    **to_dataframe_kwargs: Any,
) -> Callable[[adaptive.BaseLearner], None]:
    """Save the learner's data to disk as pandas.DataFrame."""
    save_kwargs = save_kwargs or {}

    def save(learner: adaptive.BaseLearner) -> None:
        df = learner.to_dataframe(**to_dataframe_kwargs)
        if expand_dicts:
            df = expand_dict_columns(df)
        fname_df = fname_to_dataframe(fname, format=format)

        if format not in get_type_args(_DATAFRAME_FORMATS):
            msg = f"Unknown format {format}"
            raise ValueError(msg)
        do_save = getattr(df, f"to_{format}")

        if format == "hdf":
            assert save_kwargs is not None  # for mypy
            if "key" not in save_kwargs:
                save_kwargs["key"] = "data"

        if atomically:
            with atomic_write(fname_df, return_path=True) as fname_temp:
                do_save(fname_temp, **save_kwargs)
        else:
            do_save(fname_df, **save_kwargs)

    return save


@contextmanager
def atomic_write(
    dest: os.PathLike,
    mode: str = "w",
    *args: Any,
    return_path: bool = False,
    **kwargs: Any,
) -> Any:
    """Write atomically to 'dest', using a temporary file in the same directory.

    This function has the same signature as 'open', except that the default
    mode is 'w', not 'r', and there is an additional keyword-only parameter,
    'return_path'. If 'return_path=True' then a Path pointing to the (as yet
    nonexistant) temporary file is yielded, rather than a file handle.
    This is useful when calling libraries that expect a path, rather than an
    open file handle.
    """
    temp_dest = Path(dest).with_suffix(f".temp.{os.getpid()}.{uuid.uuid4()}")
    try:
        # First create an empty file; this ensures we have the same semantics
        # as 'open(..., mode="w")'.
        temp_dest.open("w").close()
        # Now give control back to the caller.
        if return_path:
            yield temp_dest
        else:
            with temp_dest.open(mode, *args, **kwargs) as fp:
                yield fp
        # Atomically change 'dest' to point to the 'temp_dest' inode.
        os.replace(temp_dest, dest)  # noqa: PTH105
    except Exception:
        with suppress(FileNotFoundError):
            os.remove(temp_dest)  # noqa: PTH107
        raise


_DATAFRAME_FORMATS = Literal[
    "parquet",
    "csv",
    "hdf",
    "pickle",
    "feather",
    "excel",
    "json",
]


def expand_dict_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Expand dict columns in a dataframe."""
    if df.empty:
        return df

    for col, val in df.iloc[0].items():
        if isinstance(val, dict):
            prefix = f"{col}."
            x = pd.json_normalize(df.pop(col)).add_prefix(prefix)  # type: ignore[arg-type]
            x.index = df.index
            for _col in x:
                assert _col not in df, f"{_col=} already exists in df."
            df = df.join(x)
    return df


def load_dataframes(
    fnames: list[str] | list[list[str]] | list[Path] | list[list[Path]],
    *,
    concat: bool = True,
    read_kwargs: dict[str, Any] | None = None,
    format: _DATAFRAME_FORMATS = "pickle",  # noqa: A002
) -> pd.DataFrame | list[pd.DataFrame]:
    """Load a list of dataframes from disk."""
    read_kwargs = read_kwargs or {}
    if format == "hdf" and "key" not in read_kwargs:
        read_kwargs["key"] = "data"

    if format not in get_type_args(_DATAFRAME_FORMATS):
        msg = f"Unknown format {format}."
        raise ValueError(msg)

    do_read = getattr(pd, f"read_{format}")

    dfs = []
    for fn in fnames:
        fn_df = fname_to_dataframe(fn, format=format)
        if not os.path.exists(fn_df):  # noqa: PTH110
            continue
        try:
            df = do_read(fn_df, **read_kwargs)
        except Exception:  # noqa: BLE001
            msg = f"`{fn}`'s DataFrame ({fn_df}) could not be read."
            console.print(msg)
            continue

        df["fname"] = len(df) * [fn]
        dfs.append(df)

    if concat:
        if dfs:
            return pd.concat(dfs, axis=0)
        return pd.DataFrame()
    return dfs


def _at_least_adaptive_version(
    version: str,
    name: str = "",
    *,
    raises: bool = True,
) -> bool:
    import pkg_resources

    required = pkg_resources.parse_version(version)
    v = adaptive.__version__
    v_clean = ".".join(v.split(".")[:3])  # remove the dev0 or other suffix
    current = pkg_resources.parse_version(v_clean)
    if current < required:
        if raises:
            msg = (
                f"`{name}` requires adaptive version "
                f"of at least '{required}', currently using '{current}'.",
            )
            raise RuntimeError(msg)
        return False
    return True


class _TimeGoal:
    def __init__(self, dt: timedelta | datetime) -> None:
        self.dt = dt
        self.start_time: datetime | None = None

    def __call__(self, learner: adaptive.BaseLearner) -> bool:  # noqa: ARG002
        if isinstance(self.dt, timedelta):
            if self.start_time is None:
                self.start_time = datetime.now()  # noqa: DTZ005
            return datetime.now() - self.start_time > self.dt  # noqa: DTZ005
        if isinstance(self.dt, datetime):
            return datetime.now() > self.dt  # noqa: DTZ005
        msg = f"{self.dt=} is not a datetime or timedelta."
        raise TypeError(msg)


def smart_goal(
    goal: GoalTypes,
    learners: list[adaptive.BaseLearner],
) -> Callable[[adaptive.BaseLearner], bool]:
    """Extract a goal from the learners.

    Parameters
    ----------
    goal
        Either a typical callable goal, or integer for number of points goal,
        or float for loss goal, or None to automatically determine, or
        `datetime.timedelta` for a time-based goal.
    learners
        List of learners.

    Returns
    -------
    Callable[[adaptive.BaseLearner], bool]

    """
    if callable(goal):
        return goal
    if isinstance(goal, int):
        return lambda learner: learner.npoints >= goal
    if isinstance(goal, float):
        return lambda learner: learner.loss() <= goal
    if isinstance(goal, (timedelta, datetime)):
        return _TimeGoal(goal)
    if goal is None:
        learner_types = {type(learner) for learner in learners}
        if len(learner_types) > 1:
            msg = "Multiple learner types found."
            raise TypeError(msg)
        if isinstance(learners[0], adaptive.SequenceLearner):
            return adaptive.SequenceLearner.done
        warnings.warn(
            "Goal is None which means the learners continue forever!",
            stacklevel=2,
        )
        return lambda _: False
    msg = "goal must be `callable | float | None`"
    raise ValueError(msg)


def _serialize_to_b64(x: Any) -> str:
    serialized_x = cloudpickle.dumps(x)
    return base64.b64encode(serialized_x).decode("utf-8")


def _deserialize_from_b64(x: str) -> Any:
    bytes_ = base64.b64decode(x)
    return cloudpickle.loads(bytes_)


_GLOBAL_CACHE = {}


class WrappedFunction:
    """A wrapper to allow `cloudpickle.load`ed functions with `ProcessPoolExecutor`.

    A wrapper around a serialized function that handles deserialization and
    caches the deserialized function in the worker process.

    Parameters
    ----------
    function
        The function to be serialized and wrapped.
    mode
        All of the options avoids sending the function to all workers.
        If "random_id", store the serialized function only in the global cache.
        If "file", save the serialized function to a file and store the path
        to the file in the global cache. Only keep the path in this object.
        If "memory", store the full serialized function in the object.

    Attributes
    ----------
    _cache_key
        The key used to access the deserialized function in the global cache.

    Examples
    --------
    >>> import cloudpickle
    >>> def square(x):
    ...     return x * x
    >>> wrapped_function = WrappedFunction(square)
    >>> wrapped_function(4)
    16

    """

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        mode: Literal["memory", "random_id", "file"] = "random_id",
    ) -> None:
        """Initialize WrappedFunction."""
        serialized_function = cloudpickle.dumps(function)
        self.mode = mode

        if mode == "file":
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(serialized_function)
                self._cache_key = f.name
        elif mode == "memory":
            self._cache_key = serialized_function
        elif mode == "random_id":
            assert platform.system() == "Linux"
            name = function.__name__ if hasattr(function, "__name__") else "function"
            self._cache_key = f"{name}_{os.urandom(16).hex()}"
        else:
            msg = f"mode={mode} is not valid."
            raise ValueError(msg)

        # This setting of the cache only works on Linux where the default start method
        # is 'fork'. On MacOS it is 'spawn', so the cache can be populated in __call__.
        global _GLOBAL_CACHE  # noqa: PLW0602
        _GLOBAL_CACHE[self._cache_key] = function

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function.

        Retrieves the deserialized function from the global cache and calls it
        with the provided arguments and keyword arguments.

        Parameters
        ----------
        *args
            Positional arguments to pass to the deserialized function.
        **kwargs
            Keyword arguments to pass to the deserialized function.

        Returns
        -------
        Any
            The result of calling the deserialized function with the provided
            arguments and keyword arguments.

        """
        global _GLOBAL_CACHE  # noqa: PLW0602

        if self._cache_key not in _GLOBAL_CACHE:
            if self.mode == "file":
                with open(self._cache_key, "rb") as f:  # noqa: PTH123
                    serialized_function = f.read()
                deserialized_function = cloudpickle.loads(serialized_function)
                _GLOBAL_CACHE[self._cache_key] = deserialized_function
            elif self.mode == "memory":
                _GLOBAL_CACHE[self._cache_key] = cloudpickle.loads(self._cache_key)
            elif self.mode == "random_id":
                msg = (
                    "The function was not found in the global cache. "
                    "This is likely because the default start method on "
                    "your system is 'spawn' instead of 'fork'. "
                    "Try setting `mode='file' or `mode='memory'`."
                )
                raise RuntimeError(msg)

        deserialized_function = _GLOBAL_CACHE[self._cache_key]
        return deserialized_function(*args, **kwargs)


def _now() -> str:
    """Return the current time as a string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _from_datetime(dt: str) -> datetime:
    """Return a string representation of a datetime object."""
    return datetime.fromisoformat(dt)


def _time_between(start: str, end: str) -> float:
    """Return the time between two strings representing datetimes."""
    dt_start = _from_datetime(start)
    dt_end = _from_datetime(end)
    return (dt_end - dt_start).total_seconds()


async def sleep_unless_task_is_done(
    task: asyncio.Task,
    sleep_duration: float,
) -> bool:
    """Sleep for an interval, unless the task is done before then."""
    # Create the sleep task separately
    sleep_task = asyncio.create_task(asyncio.sleep(sleep_duration))

    # Await both the sleep_task and the passed task
    done, pending = await asyncio.wait(
        [sleep_task, task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel only the sleep_task if it's pending
    if sleep_task in pending:
        sleep_task.cancel()
        return True  # means that the task is done
    return False


def _update_progress_for_paths(
    paths_dict: dict[str, set[Path | tuple[Path, ...]]],
    progress: Progress,
    total_task: TaskID | None,
    task_ids: dict[str, TaskID],
) -> int:
    """Update progress bars for each set of paths."""
    n_completed = _remove_completed_paths(paths_dict)
    total_completed = sum(n_completed.values())
    for key, n_done in n_completed.items():
        progress.update(task_ids[key], advance=n_done)
    if total_task is not None:
        progress.update(total_task, advance=total_completed)
    return total_completed


def _remove_completed_paths(
    paths_dict: dict[str, set[Path | tuple[Path, ...]]],
) -> dict[str, int]:
    n_completed = {}

    for key, paths in paths_dict.items():
        completed_count = 0
        to_discard = set()
        to_add = set()

        for path_unit in paths:
            # Check if it's a single Path or a tuple of Paths
            paths_to_check = [path_unit] if isinstance(path_unit, Path) else path_unit

            # Check if all paths in the path_unit exist
            if all(p.exists() for p in paths_to_check):
                completed_count += 1
                to_discard.add(path_unit)
            elif isinstance(path_unit, tuple):
                exists = {p for p in path_unit if p.exists()}
                if any(exists):
                    to_discard.add(path_unit)
                    to_add.add(tuple(p for p in path_unit if p not in exists))

        n_completed[key] = completed_count
        paths_dict[key] -= to_discard
        paths_dict[key] |= to_add

    return n_completed


async def _track_file_creation_progress(
    paths_dict: dict[str, set[Path | tuple[Path, ...]]],
    progress: Progress,
    interval: float = 1,
) -> None:
    """Asynchronously track and update the progress of file creation.

    Parameters
    ----------
    paths_dict
        A dictionary with keys representing categories and values being sets of file paths to monitor.
    progress
        The Progress object from the rich library for displaying progress.
    interval
        The time interval (in seconds) at which to update the progress. The interval is dynamically
        adjusted to be at least 50 times the time it takes to update the progress. This ensures that
        updating the progress does not take up a significant amount of time.

    """
    # create total_files and add_total_progress before updating paths_dict
    total_files = sum(len(paths) for paths in paths_dict.values())
    add_total_progress = len(paths_dict) > 1
    n_completed = _remove_completed_paths(paths_dict)  # updates paths_dict in-place
    total_done = sum(n_completed.values())
    task_ids: dict[str, TaskID] = {}

    # Add a total progress bar only if there are multiple entries in the dictionary
    total_task = (
        progress.add_task(
            "[cyan bold underline]Total",
            total=total_files,
            completed=total_done,
        )
        if add_total_progress
        else None
    )
    for key, n_done in n_completed.items():
        n_remaining = len(paths_dict.get(key, []))
        task_ids[key] = progress.add_task(
            f"[green]{key}",
            total=n_remaining + n_done,
            completed=n_done,
        )
    try:
        progress.start()  # Start the progress display
        total_processed = 0
        while True:
            t_start = time.time()
            total_processed += _update_progress_for_paths(
                paths_dict,
                progress,
                total_task,
                task_ids,
            )
            if total_processed >= total_files:
                progress.refresh()  # Final refresh to ensure 100%
                break  # Exit loop if all files are processed
            progress.refresh()
            # Sleep for at least 50 times the update time
            t_update = time.time() - t_start
            await asyncio.sleep(max(interval, 50 * t_update))

    finally:
        progress.stop()  # Stop the progress display, regardless of what happens


def track_file_creation_progress(
    paths_dict: dict[str, set[Path | tuple[Path, ...]]],
    interval: int = 1,
) -> asyncio.Task:
    """Initialize and asynchronously track the progress of file creation.

    WARNING: This function modifies the provided dictionary in-place.

    This function sets up an asynchronous monitoring system that periodically
    checks for the existence of specified files or groups of files. Each item
    in the provided dictionary can be a single file (Path object) or a group
    of files (tuple of Path objects). The progress is updated for each file or
    group of files only when all files in the group exist. This allows tracking
    of complex file creation processes where multiple files together constitute
    a single unit of work.

    The tracking occurs at regular intervals, specified by the user, and updates
    individual and, if applicable, total progress bars to reflect the current
    state of file creation. It is particularly useful in environments where files
    are expected to be created over time and need to be monitored collectively.

    Parameters
    ----------
    paths_dict : dict[str, set[Union[Path, Tuple[Path, ...]]]]
        A dictionary with keys representing categories and values being sets of
        file paths (Path objects) or groups of file paths (tuples of Path objects)
        to monitor.
    interval : int
        The time interval (in seconds) at which the progress is updated.

    Returns
    -------
    asyncio.Task
        The asyncio Task object that is tracking the file creation progress.

    Examples
    --------
    >>> paths_dict = {
        "docs": {Path("docs/environment.yml"), (Path("doc1.md"), Path("doc2.md"))},
        "example2": {Path("/path/to/file3"), Path("/path/to/file4")},
    }
    >>> task = track_file_creation_progress(paths_dict)

    """
    get_console().clear_live()  # avoid LiveError, only 1 live render allowed at a time
    columns = (*Progress.get_default_columns(), TimeElapsedColumn())
    progress = Progress(*columns, auto_refresh=False)
    coro = _track_file_creation_progress(paths_dict, progress, interval)
    ioloop = asyncio.get_event_loop()
    return ioloop.create_task(coro)
