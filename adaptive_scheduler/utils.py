from __future__ import annotations

import abc
import collections.abc
import functools
import hashlib
import inspect
import math
import os
import pickle
import random
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from inspect import signature
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import adaptive
import cloudpickle
import numpy as np
import pandas as pd
import toolz
from adaptive.notebook_integration import in_ipynb
from ipyparallel import Client
from tqdm import tqdm, tqdm_notebook

MAX_LINE_LENGTH = 100
_NONE_RETURN_STR = "__ReturnsNone__"


class _RequireAttrsABCMeta(abc.ABCMeta):
    required_attributes = []

    def __call__(self, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        for name in obj.required_attributes:
            if not hasattr(obj, name):
                raise ValueError(f"Required attribute {name} not set in __init__.")
        return obj


def shuffle_list(*lists, seed=0):
    """Shuffle multiple lists in the same order."""
    combined = list(zip(*lists))
    random.Random(seed).shuffle(combined)
    return zip(*combined)


def hash_anything(x):
    try:
        return hashlib.md5(x).hexdigest()
    except TypeError:
        return hashlib.md5(pickle.dumps(x)).hexdigest()


def _split(seq: collections.abc.Iterable, n_parts: int):
    # TODO: remove this in v1.0.0
    s = "adaptive_scheduler.utils."
    raise Exception(f"`{s}_split` is renamed to {s}split`.")


def split(seq: collections.abc.Iterable, n_parts: int):
    """Split up a sequence into ``n_parts``.

    Parameters
    ----------
    seq : sequence
        A list or other iterable that has to be split up.
    n_parts : int
        The sequence will be split up in this many parts.

    Returns
    -------
    iterable of tuples"""
    lst = list(seq)
    n = math.ceil(len(lst) / n_parts)
    return toolz.partition_all(n, lst)


def split_in_balancing_learners(
    learners: list[adaptive.BaseLearner],
    fnames: list[str],
    n_parts: int,
    strategy: str = "npoints",
) -> tuple[list[adaptive.BaseLearner], list[str]]:
    r"""Split a list of learners and fnames into `adaptive.BalancingLearner`\s.

    Parameters
    ----------
    learners : list
        List of learners.
    fnames : list
        List of filenames.
    n_parts : int
        Total number of `~adaptive.BalancingLearner`\s.
    strategy : str
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
        new_fnames.append(fnames_part)
    return new_learners, new_fnames


def split_sequence_learner(
    big_learner, n_learners: int, folder: str | Path = ""
) -> tuple[list[adaptive.SequenceLearner], list[str]]:
    r"""Split a sinlge `~adaptive.SequenceLearner` into
    mutiple `adaptive.SequenceLearner`\s (with the data loaded) and fnames.

    See also `split_sequence_in_sequence_learners`.

    Parameters
    ----------
    big_learner : callable
        A `~adaptive.SequenceLearner` instance
    n_learners : int
        Total number of `~adaptive.SequenceLearner`\s.
    folder : pathlib.Path or str
        Folder to prepend to fnames.

    Returns
    -------
    new_learners : List[adaptive.SequenceLearner]
        List of `~adaptive.SequenceLearner`\s.
    new_fnames : List[Path]
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
    function : callable
        Function for `adaptive.SequenceLearner`\s.
    sequence : sequence
        The sequence to split into ``n_learners``.
    n_learners : int
        Total number of `~adaptive.SequenceLearner`\s.
    folder : pathlib.Path or str
        Folder to prepend to fnames.

    Returns
    -------
    new_learners : List[adaptive.SequenceLearner]
        List of `~adaptive.SequenceLearner`\s.
    new_fnames : List[Path]
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
    r"""Combine several `~adaptive.SequenceLearner`\s into a single
    `~adaptive.SequenceLearner` any copy over the data.

    Assumes that all ``learners`` take the same function.

    Parameters
    ----------
    learners : List[adaptive.SequenceLearner]
        List of `~adaptive.SequenceLearner`\s.
    big_learner : Optional[adaptive.SequenceLearner]
        A learner to load, if None, a new learner will be generated.

    Returns
    -------
    adaptive.SequenceLearner
        Big `~adaptive.SequenceLearner` with data from ``learners``.
    """
    if big_learner is None:
        big_sequence = sum((list(learner.sequence) for learner in learners), [])
        big_learner = adaptive.SequenceLearner(
            learners[0]._original_function, sequence=big_sequence
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
    learner_from: adaptive.SequenceLearner, learner_to: adaptive.SequenceLearner
) -> None:
    """Convinience function to copy the data from a `~adaptive.SequenceLearner`
    into a different `~adaptive.SequenceLearner`.

    Parameters
    ----------
    learner_from : adaptive.SequenceLearner
        Learner to take the data from.
    learner_to : adaptive.SequenceLearner
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


def _progress(
    seq: collections.abc.Iterable, with_progress_bar: bool = True, desc: str = ""
):
    if not with_progress_bar:
        return seq
    else:
        if in_ipynb():
            return tqdm_notebook(list(seq), desc=desc)
        else:
            return tqdm(list(seq), desc=desc)


def combo_to_fname(
    combo: dict[str, Any], folder: str | None = None, ext: str | None = ".pickle"
) -> str:
    """Converts a dict into a human readable filename."""
    fname = "__".join(f"{k}_{v}" for k, v in combo.items()) + ext
    if folder is None:
        return fname
    return os.path.join(folder, fname)


def combo2fname(
    combo: dict[str, Any],
    folder: str | Path | None = None,
    ext: str | None = ".pickle",
    sig_figs: int = 8,
) -> str:
    """Converts a dict into a human readable filename.

    Improved version of `combo_to_fname`."""
    name_parts = [f"{k}_{maybe_round(v, sig_figs)}" for k, v in sorted(combo.items())]
    fname = Path("__".join(name_parts) + ext)
    if folder is None:
        return fname
    return str(folder / fname)


def add_constant_to_fname(
    combo: dict[str, Any],
    constant: dict[str, Any],
    folder: str | Path | None = None,
    ext: str | None = ".pickle",
    sig_figs: int = 8,
    dry_run: bool = True,
):
    for k in constant.keys():
        combo.pop(k, None)
    old_fname = combo2fname(combo, folder, ext, sig_figs)
    combo.update(constant)
    new_fname = combo2fname(combo, folder, ext, sig_figs)
    if not dry_run:
        old_fname.rename(new_fname)
    return old_fname, new_fname


def maybe_round(x: Any, sig_figs: int) -> Any:
    rnd = functools.partial(round_sigfigs, sig_figs=sig_figs)

    def try_is_nan_inf(x):
        try:
            return np.isnan(x) or np.isinf(x)
        except Exception:
            return False

    if try_is_nan_inf(x):
        return x
    elif isinstance(x, (np.float, float)):
        return rnd(x)
    elif isinstance(x, (complex, np.complex)):
        return complex(rnd(x.real), rnd(x.imag))
    else:
        return x


def round_sigfigs(num: float, sig_figs: int) -> float:
    """Round to specified number of sigfigs.

    From
    http://code.activestate.com/recipes/578114-round-number-to-specified-number-of-significant-di/
    """
    num = float(num)
    if num != 0:
        return round(num, -int(math.floor(math.log10(abs(num))) - (sig_figs - 1)))
    else:
        return 0.0  # Can't take the log of 0


def _remove_or_move_files(
    fnames: list[str],
    with_progress_bar: bool = True,
    move_to: str | None = None,
    desc: str | None = None,
) -> None:
    """Remove files by filename.

    Parameters
    ----------
    fnames : list
        List of filenames.
    with_progress_bar : bool, default: True
        Display a progress bar using `tqdm`.
    move_to : str, default None
        Move the file to a different directory.
        If None the file is removed.
    desc : str, default: None
        Description of the progressbar.
    """
    n_failed = 0
    for fname in _progress(fnames, with_progress_bar, desc or "Removing files"):
        try:
            if move_to is None:
                os.remove(fname)
            else:
                os.makedirs(move_to, exist_ok=True)
                src = Path(fname).resolve()
                dst = (Path(move_to) / src.name).resolve()
                shutil.move(src, dst)  # overwrites old files
        except Exception:
            n_failed += 1

    if n_failed:
        warnings.warn(f"Failed to remove (or move) {n_failed}/{len(fnames)} files.")


def load_parallel(
    learners: list[adaptive.BaseLearner],
    fnames: list[str],
    *,
    with_progress_bar: bool = True,
    max_workers: int | None = None,
) -> None:
    r"""Load a sequence of learners in parallel.

    Parameters
    ----------
    learners : sequence of `adaptive.BaseLearner`\s
        The learners to be loaded.
    fnames : sequence of str
        A list of filenames corresponding to `learners`.
    with_progress_bar : bool, default True
        Display a progress bar using `tqdm`.
    max_workers : int, optional
        The maximum number of parallel threads when loading the data.
        If ``None``, use the maximum number of threads that is possible.
    """

    def load(learner, fname):
        learner.load(fname)

    with ThreadPoolExecutor(max_workers) as ex:
        iterator = zip(learners, fnames)
        pbar = _progress(iterator, with_progress_bar, "Submitting loading tasks")
        futs = [ex.submit(load, *args) for args in pbar]
        for fut in _progress(futs, with_progress_bar, "Finishing loading"):
            fut.result()


def save_parallel(
    learners: list[adaptive.BaseLearner],
    fnames: list[str],
    *,
    with_progress_bar: bool = True,
) -> None:
    r"""Save a sequence of learners in parallel.

    Parameters
    ----------
    learners : sequence of `adaptive.BaseLearner`\s
        The learners to be saved.
    fnames : sequence of str
        A list of filenames corresponding to `learners`.
    with_progress_bar : bool, default True
        Display a progress bar using `tqdm`.
    """

    def save(learner, fname):
        learner.save(fname)

    with ThreadPoolExecutor() as ex:
        iterator = zip(learners, fnames)
        pbar = _progress(iterator, with_progress_bar, "Submitting saving tasks")
        futs = [ex.submit(save, *args) for args in pbar]
        for fut in _progress(futs, with_progress_bar, "Finishing saving"):
            fut.result()


def _print_same_line(msg: str, new_line_end: bool = False):
    msg = msg.strip()
    global MAX_LINE_LENGTH
    MAX_LINE_LENGTH = max(len(msg), MAX_LINE_LENGTH)
    empty_space = max(MAX_LINE_LENGTH - len(msg), 0) * " "
    print(msg + empty_space, end="\r" if not new_line_end else "\n")


def _wait_for_successful_ipyparallel_client_start(client, n: int, timeout: int):
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

    raise Exception(f"Not all ({n_engines}/{n}) connected after {timeout} seconds.")


def connect_to_ipyparallel(
    n: int,
    profile: str,
    timeout: int = 300,
    folder: str | None = None,
    client_kwargs=None,
):
    """Connect to an `ipcluster` on the cluster headnode.

    Parameters
    ----------
    n : int
        Number of engines to be started.
    profile : str
        Profile name of IPython profile.
    timeout : int
        Time for which we try to connect to get all the engines.
    folder : str, optional
        Folder that is added to the path of the engines, e.g. ``"~/Work/my_current_project"``.

    Returns
    -------
    client : `ipyparallel.Client` object
        An IPyparallel client.
    """
    client = Client(profile=profile, **(client_kwargs or {}))
    dview = _wait_for_successful_ipyparallel_client_start(client, n, timeout)
    dview.use_dill()

    if folder is not None:
        print(f"Adding {folder} to path.")
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


def log_exception(log, msg, exception):
    try:
        raise exception
    except Exception:
        log.exception(msg, exc_info=True)


def maybe_lst(fname: list[str] | str):
    if isinstance(fname, tuple):
        # TinyDB converts tuples to lists
        fname = list(fname)
    return fname


def _serialize(msg):
    return [cloudpickle.dumps(msg)]


def _deserialize(frames):
    try:
        return cloudpickle.loads(frames[0])
    except pickle.UnpicklingError as e:
        if r"\x03" in str(e):
            # Means that the frame is empty because it only contains an end of text char
            # `\x03  ^C    (End of text)`
            # TODO: Not sure why this happens.
            print(
                r"pickle.UnpicklingError in _deserialize: Received an empty frame (\x03)."
            )
        raise


class LRUCachedCallable(Callable[..., Any]):
    """Wraps a function to become cached.

    Parameters
    ----------
    function : Callable[..., Any]
    max_size : int, optional
        Cache size of the LRU cache, by default 128.
    with_cloudpickle : bool
        Use cloudpickle for storing the data in memory.
    """

    def __init__(
        self,
        function: Callable[..., Any],
        max_size: int = 128,
        with_cloudpickle: bool = False,
    ):
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

    def _get_from_cache(self, key: str) -> Any | None:
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

    def _insert_into_cache(self, key: str, value: Any):
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
    def cache_dict(self):
        """Returns a copy of the cache."""
        return dict(self._cache_dict.items())

    def __call__(self, *args, **kwargs) -> Any:
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


def shared_memory_cache(cache_size: int = 128):
    """Create a cache similar to `functools.lru_cache.

    This will actually cache the return values of the function, whereas
    `functools.lru_cache` will pickle the decorated function each time
    with an empty cache.
    """

    def cache_decorator(function):
        return functools.wraps(function)(LRUCachedCallable(function, cache_size))

    return cache_decorator


def _prefix(fname: str | list[str] | tuple[str, ...]) -> str:
    if isinstance(fname, (tuple, list)):
        return f".{len(fname):08}_learners."
    elif isinstance(fname, str):
        return ".learner."
    else:
        raise TypeError("Incorrect type for fname.")


def fname_to_learner_fname(fname: str | list[str] | tuple[str, ...]) -> str:
    prefix = _prefix(fname)
    if isinstance(fname, (tuple, list)):
        fname = fname[0]
    p = Path(fname)
    return str(p.with_stem(f"{prefix}{p.stem}"))


def fname_to_learner(fname: str | list[str] | tuple[str, ...]) -> adaptive.BaseLearner:
    learner_name = fname_to_learner_fname(fname)
    with open(learner_name, "rb") as f:
        return cloudpickle.load(f)


def _ensure_folder_exists(fnames: list[str | list[str] | tuple[str, ...]]) -> None:
    if isinstance(fnames[0], (tuple, list)):
        for _fnames in fnames:
            _ensure_folder_exists(_fnames)
    else:
        folders = {Path(fname).parent for fname in fnames}
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)


def cloudpickle_learners(
    learners,
    fnames: list[str | list[str] | tuple[str, ...]],
    with_progress_bar: bool = False,
    empty_copies: bool = True,
):
    """Save a list of learners to disk using cloudpickle."""
    _ensure_folder_exists(fnames)

    for learner, fname in _progress(
        zip(learners, fnames), with_progress_bar, desc="Cloudpickling learners"
    ):
        fname_learner = fname_to_learner_fname(fname)
        if empty_copies:
            _require_adaptive("0.14.1", "empty_copies")
            learner = learner.new()
        with open(fname_learner, "wb") as f:
            cloudpickle.dump(learner, f)


def fname_to_dataframe(
    fname: str | list[str] | tuple[str, ...], format: str = "parquet"
) -> str | list[str]:
    if format == "excel":
        format = "xlsx"
    if isinstance(fname, (tuple, list)):
        fname = fname[0]
    p = Path(fname)
    return str(p.with_stem(f"dataframe.{p.stem}").with_suffix(f".{format}"))


def save_dataframe(
    fname: str | list[str] | tuple[str, ...],
    format: _DATAFRAME_FORMATS = "parquet",
    save_kwargs: dict[str, Any] | None = None,
    expand_dicts: bool = True,
    **to_dataframe_kwargs: Any,
) -> Callable[[adaptive.BaseLearner], None]:
    save_kwargs = save_kwargs or {}

    def save(learner):
        df = learner.to_dataframe(**to_dataframe_kwargs)
        if expand_dicts:
            df = expand_dict_columns(df)
        fname_df = fname_to_dataframe(fname, format=format)
        if format == "parquet":
            df.to_parquet(fname_df, **save_kwargs)
        elif format == "csv":
            df.to_csv(fname_df, **save_kwargs)
        elif format == "hdf":
            if "key" not in save_kwargs:
                save_kwargs["key"] = "data"
            df.to_hdf(fname_df, **save_kwargs)
        elif format == "pickle":
            df.to_pickle(fname_df, **save_kwargs)
        elif format == "feather":
            df.to_feather(fname_df, **save_kwargs)
        elif format == "excel":
            df.to_excel(fname_df, **save_kwargs)
        elif format == "json":
            df.to_json(fname_df, **save_kwargs)
        else:
            raise ValueError(f"Unknown format {format}.")

    return save


_DATAFRAME_FORMATS = Literal[
    "parquet", "csv", "hdf", "pickle", "feather", "excel", "json"
]


def expand_dict_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    for col, val in df.iloc[0].iteritems():
        if isinstance(val, dict):
            prefix = f"{col}."
            x = pd.json_normalize(df.pop(col)).add_prefix(prefix)
            x.index = df.index
            for col in x:
                assert col not in df, f"{col=} already exists in df."
            df = df.join(x)
    return df


def load_dataframes(
    fnames: list[str] | list[list[str]],
    concat: bool = True,
    read_kwargs: dict[str, Any] | None = None,
    format: _DATAFRAME_FORMATS = "parquet",
) -> pd.DataFrame | list[pd.DataFrame]:
    read_kwargs = read_kwargs or {}
    dfs = []
    for fn in fnames:
        fn_df = fname_to_dataframe(fn, format=format)
        if not os.path.exists(fn):
            continue
        try:
            if format == "parquet":
                df = pd.read_parquet(fn_df, **read_kwargs)
            elif format == "csv":
                df = pd.read_csv(fn_df, **read_kwargs)
            elif format == "hdf":
                if "key" not in read_kwargs:
                    read_kwargs["key"] = "data"
                df = pd.read_hdf(fn_df, **read_kwargs)
            elif format == "pickle":
                df = pd.read_pickle(fn_df, **read_kwargs)
            elif format == "feather":
                df = pd.read_feather(fn_df, **read_kwargs)
            elif format == "excel":
                df = pd.read_excel(fn_df, **read_kwargs)
            elif format == "json":
                df = pd.read_json(fn_df, **read_kwargs)
            else:
                raise ValueError(f"Unknown format {format}.")
        except Exception:
            print(f"`{fn}`'s DataFrame ({fn_df}) could not be read.")
            continue
        df["fname"] = len(df) * [fn]
        dfs.append(df)
    if concat:
        if dfs:
            return pd.concat(dfs, axis=0)
        else:
            return pd.DataFrame()
    else:
        return dfs


def _require_adaptive(version: str, name: str) -> None:
    import pkg_resources

    required = pkg_resources.parse_version(version)
    v = adaptive.__version__
    v_clean = ".".join(v.split(".")[:3])  # remove the dev0 or other suffix
    current = pkg_resources.parse_version(v_clean)
    if current < required:
        raise RuntimeError(
            f"`{name}` requires adaptive version "
            f"of at least {required}, currently using {current}."
        )


def smart_goal(
    goal: Callable[[adaptive.BaseLearner], bool] | int | float | None,
    learners: list[adaptive.BaseLearner],
):
    """Extract a goal from the learners.

    Parameters
    ----------
    goal
        Either a typical callable goal, or integer for number of points goal,
        or float for loss goal, or None to automatically determine.

    Returns
    -------
    Callable[[adaptive.BaseLearner], bool]
    """
    if callable(goal):
        return goal
    elif isinstance(goal, int):
        return lambda learner: learner.npoints >= goal
    elif isinstance(goal, float):
        return lambda learner: learner.loss() <= goal
    elif goal is None:
        learner_types = {type(learner) for learner in learners}
        if len(learner_types) > 1:
            raise TypeError("Multiple learner types found.")
        if isinstance(learners[0], adaptive.SequenceLearner):
            return adaptive.SequenceLearner.done
        raise RuntimeError(f"Cannot determine goal for {type(learners[0])}")
    else:
        raise ValueError("goal must be `callable | int | float | None`")
