"""Tests for `adaptive_scheduler.utils`."""

from __future__ import annotations

import platform
import sys
import time
import typing
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import adaptive
import cloudpickle
import pandas as pd
import pytest

from adaptive_scheduler import utils

if TYPE_CHECKING:
    from typing import Literal


def test_shuffle_list() -> None:
    """Test `utils.shuffle_list`."""
    list1 = [1, 2, 3, 4, 5]
    list2 = ["a", "b", "c", "d", "e"]
    shuffled1, shuffled2 = utils.shuffle_list(list1, list2, seed=42)
    assert shuffled1 == (4, 2, 3, 5, 1)
    assert shuffled2 == ("d", "b", "c", "e", "a")


def test_hash_anything() -> None:
    """Test `utils.hash_anything`."""
    assert utils.hash_anything("test_string") == "fe61a408ddf146f33219dfcfbfae6fe6"
    assert utils.hash_anything(12345) == "ab828429197deaaefe90ed273900e4a3"
    assert utils.hash_anything(["a", "b", "c"]) == "92c9643673b92ef3058d62fccca628e2"


def test_split() -> None:
    """Test `utils.split`."""
    seq = list(range(10))
    n_parts = 3
    result = list(utils.split(seq, n_parts))
    assert result == [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9)]


def test_split_in_balancing_learners(
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> None:
    """Test `utils.split_in_balancing_learners`."""
    n_parts = 2
    new_learners, new_fnames = utils.split_in_balancing_learners(
        learners,
        fnames,
        n_parts,
    )
    assert len(new_learners) == n_parts
    assert all(isinstance(lrn, adaptive.BalancingLearner) for lrn in new_learners)
    assert new_fnames == [[fnames[0]], [fnames[1]]]


def test_split_sequence_learner() -> None:
    """Test `utils.split_sequence_learner`."""
    big_learner = adaptive.SequenceLearner(lambda x: x, sequence=list(range(10)))
    n_learners = 3
    new_learners, new_fnames = utils.split_sequence_learner(big_learner, n_learners)
    assert len(new_learners) == n_learners
    assert all(isinstance(lrn, adaptive.SequenceLearner) for lrn in new_learners)


def test_split_sequence_in_sequence_learners() -> None:
    """Test `utils.split_sequence_in_sequence_learners`."""

    def function(x: int) -> int:
        return x

    sequence = list(range(10))
    n_learners = 3
    new_learners, new_fnames = utils.split_sequence_in_sequence_learners(
        function,
        sequence,
        n_learners,
    )
    assert len(new_learners) == n_learners
    assert all(isinstance(lrn, adaptive.SequenceLearner) for lrn in new_learners)
    assert len(new_fnames) == n_learners


def test_combine_sequence_learners() -> None:
    """Test `utils.combine_sequence_learners`."""
    learners = [
        adaptive.SequenceLearner(lambda x: x, sequence=list(range(5))),
        adaptive.SequenceLearner(lambda x: x, sequence=list(range(5, 10))),
    ]
    big_learner = utils.combine_sequence_learners(learners)
    assert isinstance(big_learner, adaptive.SequenceLearner)
    assert list(big_learner.sequence) == list(range(10))


def test_copy_from_sequence_learner() -> None:
    """Test `utils.copy_from_sequence_learner`."""
    learner1 = adaptive.SequenceLearner(lambda x: x, sequence=list(range(5)))
    learner2 = adaptive.SequenceLearner(lambda x: x, sequence=list(range(5)))
    for i, x in enumerate(learner1.sequence):
        learner1.tell((i, x), x * 2)
    utils.copy_from_sequence_learner(learner1, learner2)
    assert learner1.data == learner2.data


def test_get_npoints() -> None:
    """Test `utils._get_npoints`."""
    learner1 = adaptive.Learner1D(lambda x: x, bounds=(-10, 10))
    assert utils._get_npoints(learner1) == 0
    learner1.ask(10)
    learner1.tell(10, 42)
    assert utils._get_npoints(learner1) == 1


def test_progress() -> None:
    """Test `utils._progress`."""
    seq = list(range(10))
    with_progress_bar = False
    result = list(utils._progress(seq, with_progress_bar))
    assert result == seq


def test_combo_to_fname() -> None:
    """Test `utils.combo_to_fname`."""
    combo = {"x": 1, "y": 2, "z": 3}
    fname = utils.combo_to_fname(combo)
    assert str(fname) == "x_1__y_2__z_3.pickle"


def test_combo2fname() -> None:
    """Test `utils.combo2fname`."""
    combo = {"x": 1, "y": 2, "z": 3}
    fname = utils.combo2fname(combo)
    assert str(fname) == "x_1__y_2__z_3.pickle"


def test_add_constant_to_fname() -> None:
    """Test `utils.add_constant_to_fname`."""
    combo = {"x": 1, "y": 2, "z": 3}
    constant = {"a": 42}
    old_fname, new_fname = utils.add_constant_to_fname(combo, constant)
    assert str(old_fname) == "x_1__y_2__z_3.pickle"
    assert str(new_fname) == "a_42__x_1__y_2__z_3.pickle"


def test_maybe_round() -> None:
    """Test `utils.maybe_round`."""
    assert utils.maybe_round(3.14159265, 3) == 3.14
    assert utils.maybe_round(1 + 2j, 3) == 1 + 2j
    assert utils.maybe_round("test", 3) == "test"


def test_round_sigfigs() -> None:
    """Test `utils.round_sigfigs`."""
    assert utils.round_sigfigs(3.14159265, 3) == 3.14
    assert utils.round_sigfigs(123456789, 3) == 123000000


def test_remove_or_move_files(tmp_path: Path) -> None:
    """Test `utils._remove_or_move_files`."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    assert test_file.exists()

    utils._remove_or_move_files([str(test_file)], with_progress_bar=False)

    assert not test_file.exists()


def test_load_parallel(tmp_path: Path) -> None:
    """Test `utils.load_parallel`."""
    learners = [
        adaptive.Learner1D(lambda x: x, bounds=(-10, 10)),
        adaptive.SequenceLearner(lambda x: x, sequence=list(range(5))),
    ]
    fnames = [str(tmp_path / "learner1.pickle"), str(tmp_path / "learner2.pickle")]

    for learner, fname in zip(learners, fnames):
        learner.save(fname)

    loaded_learners = [lrn.new() for lrn in learners]
    utils.load_parallel(loaded_learners, fnames, with_progress_bar=False)

    for learner, loaded_learner in zip(learners, loaded_learners):
        assert learner.data == loaded_learner.data


def test_save_parallel(tmp_path: Path) -> None:
    """Test `utils.save_parallel`."""
    learners = [
        adaptive.Learner1D(lambda x: x, bounds=(-10, 10)),
        adaptive.SequenceLearner(lambda x: x, sequence=list(range(5))),
    ]
    fnames = [str(tmp_path / "learner1.pickle"), str(tmp_path / "learner2.pickle")]

    utils.save_parallel(learners, fnames, with_progress_bar=False)

    for fname in fnames:
        assert Path(fname).exists()


def test_connect_to_ipyparallel() -> None:
    """Test `utils.connect_to_ipyparallel`."""
    with pytest.raises(
        OSError,
        match="You have attempted to connect to an IPython Cluster",
    ):
        utils.connect_to_ipyparallel(n=1, profile="non_existent_profile", timeout=1)


def test_lru_cached_callable() -> None:
    """Test `utils.LRUCachedCallable`."""

    @utils.shared_memory_cache(cache_size=5)
    def cached_function(x: float) -> float:
        return x**2

    assert cached_function.cache_dict == {}
    assert cached_function(2) == 4
    assert (
        cached_function.cache_dict == {"{'x': 2}": 4}
        if sys.version_info >= (3, 9)
        else {"OrderedDict([('x', 2)])": 4}
    )

    assert cached_function(3) == 9
    assert (
        cached_function.cache_dict == {"{'x': 2}": 4, "{'x': 3}": 9}
        if sys.version_info >= (3, 9)
        else {"OrderedDict([('x', 2)]): 4, OrderedDict([('x', 3)]): 9"}
    )


def test_smart_goal() -> None:
    """Test `utils.smart_goal`."""
    learner = adaptive.Learner1D(lambda x: x, bounds=(-10, 10))

    goal_callable = utils.smart_goal(
        lambda lrn: lrn.npoints >= 10,
        [learner],
    )
    assert not goal_callable(learner)
    adaptive.runner.simple(learner, npoints_goal=5)
    assert not goal_callable(learner)
    adaptive.runner.simple(learner, npoints_goal=11)
    assert goal_callable(learner)

    goal_int = utils.smart_goal(10, [learner])
    assert goal_int(learner)

    goal_float = utils.smart_goal(0.1, [learner])
    adaptive.runner.simple(learner, loss_goal=0.09)
    assert goal_float(learner)

    goal_timedelta = utils.smart_goal(timedelta(milliseconds=50), [learner])
    assert not goal_timedelta(learner)
    time.sleep(0.1)
    assert goal_timedelta(learner)

    goal_datetime = utils.smart_goal(
        datetime.now() + timedelta(milliseconds=50),  # noqa: DTZ005
        [learner],
    )
    assert not goal_datetime(learner)
    time.sleep(0.1)
    assert goal_datetime(learner)

    goal_none = utils.smart_goal(None, [learner])
    assert not goal_none(learner)

    seq_learner = adaptive.SequenceLearner(lambda x: x, sequence=list(range(10)))
    goal_none = utils.smart_goal(None, [seq_learner])
    assert goal_none(seq_learner) == adaptive.SequenceLearner.done(seq_learner)


def test_time_goal() -> None:
    """Test `utils._TimeGoal`."""
    time_goal = utils._TimeGoal(timedelta(seconds=1))
    learner = adaptive.Learner1D(lambda x: x, bounds=(-10, 10))
    assert not time_goal(learner)
    time.sleep(1)
    assert time_goal(learner)


def test_fname_to_learner_fname() -> None:
    """Test `utils.fname_to_learner_fname`."""
    fname = "test.pickle"
    learner_fname = utils.fname_to_learner_fname(fname)
    assert learner_fname == Path(".learner.test.pickle")


def test_fname_to_learner(tmp_path: Path) -> None:
    """Test `utils.fname_to_learner`."""
    learner = adaptive.Learner1D(lambda x: x, bounds=(-10, 10))
    fname = tmp_path / "test.pickle"
    utils.cloudpickle_learners([learner], [fname], with_progress_bar=False)

    loaded_learner = utils.fname_to_learner(fname)
    assert isinstance(loaded_learner, adaptive.Learner1D)


def test_ensure_folder_exists(tmp_path: Path) -> None:
    """Test `utils._ensure_folder_exists`."""
    fnames = [tmp_path / "test1.pickle", tmp_path / "test2.pickle"]

    for fname in fnames:
        assert not Path(fname).exists()

    utils._ensure_folder_exists(fnames)

    for fname in fnames:
        assert Path(fname.parent).exists()


def test_cloudpickle_learners(tmp_path: Path) -> None:
    """Test `utils.cloudpickle_learners`."""
    learners = [
        adaptive.Learner1D(lambda x: x, bounds=(-10, 10)),
        adaptive.SequenceLearner(lambda x: x, sequence=list(range(5))),
    ]
    fnames = [tmp_path / "learner1.pickle", tmp_path / "learner2.pickle"]

    utils.cloudpickle_learners(learners, fnames, with_progress_bar=False)

    for fname in fnames:
        fname_learner = utils.fname_to_learner_fname(fname)
        assert fname_learner.exists()


@pytest.mark.parametrize("atomically", [False, True])
@pytest.mark.parametrize("fmt", ["pickle", "csv", "json"])
def test_save_dataframe(
    tmp_path: Path,
    atomically: bool,  # noqa: FBT001
    fmt: str,
) -> None:
    """Test `utils.save_dataframe`."""
    fmt = typing.cast(utils._DATAFRAME_FORMATS, fmt)
    learner = adaptive.Learner1D(lambda x: x, bounds=(-10, 10))
    fname = str(tmp_path / f"test.{fmt}")
    save_df = utils.save_dataframe(fname, atomically=atomically, format=fmt)

    learner.ask(10)
    learner.tell(10, 42)

    save_df(learner)

    assert utils.fname_to_dataframe(fname, format=fmt).exists()


@pytest.mark.parametrize("atomically", [False, True])
def test_load_dataframes(
    tmp_path: Path,
    atomically: bool,  # noqa: FBT001
) -> None:
    """Test `utils.load_dataframes`."""
    learner = adaptive.Learner1D(lambda x: x, bounds=(-10, 10))
    fname = str(tmp_path / "test.pickle")
    save_df = utils.save_dataframe(fname, atomically=atomically)

    learner.ask(10)
    learner.tell(10, 42)

    save_df(learner)

    df = utils.load_dataframes([fname], concat=False)[0]

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_expand_dict_columns() -> None:
    """Test `utils.expand_dict_columns`."""
    data = {"a": 1, "b": {"c": 2, "d": 3}}
    df = pd.DataFrame([data])

    expanded_df = utils.expand_dict_columns(df)

    assert "b" not in expanded_df.columns
    assert {"a", "b.c", "b.d"} == set(expanded_df.columns)


def test_remove_or_move_files_move(tmp_path: Path) -> None:
    """Test `utils._remove_or_move_files` with `move_to`."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    new_folder = tmp_path / "new_folder"
    new_folder.mkdir()

    assert test_file.exists()

    utils._remove_or_move_files(
        [test_file],
        with_progress_bar=False,
        move_to=new_folder,
    )

    assert not test_file.exists()
    assert (new_folder / "test.txt").exists()


def test_combo_to_fname_with_folder() -> None:
    """Test `utils.combo_to_fname` with `folder`."""
    combo = {"x": 1, "y": 2, "z": 3}
    folder = "test_folder"
    fname = utils.combo_to_fname(combo, folder=folder)
    assert fname == Path(f"{folder}/x_1__y_2__z_3.pickle")


def test_combo2fname_with_folder() -> None:
    """Test `utils.combo2fname` with `folder`."""
    combo = {"x": 1, "y": 2, "z": 3}
    folder = "test_folder"
    fname = utils.combo2fname(combo, folder=folder)
    assert fname == Path(f"{folder}/x_1__y_2__z_3.pickle")


def test_add_constant_to_fname_with_folder() -> None:
    """Test `utils.add_constant_to_fname` with `folder`."""
    combo = {"x": 1, "y": 2, "z": 3}
    constant = {"a": 42}
    folder = "test_folder"
    old_fname, new_fname = utils.add_constant_to_fname(
        combo,
        constant,
        folder=folder,
    )
    assert old_fname == Path(f"{folder}/x_1__y_2__z_3.pickle")
    assert new_fname == Path(f"{folder}/a_42__x_1__y_2__z_3.pickle")


def test_fname_to_dataframe_with_folder() -> None:
    """Test `utils.fname_to_dataframe` with `folder`."""
    fname = "test_folder/test.pickle"
    df_fname = utils.fname_to_dataframe(fname, "parquet")
    assert df_fname == Path("test_folder/dataframe.test.parquet")


def test_atomic_write(tmp_path: Path) -> None:
    """Ensure atomic_write works when operated in a basic manner."""
    path = tmp_path / "testfile"
    content = "this is some content"

    # Works correctly in basic mode, with no file existing
    with utils.atomic_write(path) as fp:
        fp.write(content)
    with path.open() as fp:
        assert content == fp.read()

    # Works correctly in basic mode, with the file existing
    assert path.exists()
    assert path.is_file()
    content = "this is some additional content"
    with utils.atomic_write(path) as fp:
        fp.write(content)
    with path.open() as fp:
        assert content == fp.read()

    # Works correctly when 'return_path' is used.
    content = "even more content"
    with utils.atomic_write(path, return_path=True) as tmp_path, tmp_path.open(
        "w",
    ) as fp:
        fp.write(content)
    with path.open() as fp:
        assert content == fp.read()


def test_atomic_write_no_write(tmp_path: Path) -> None:
    """Ensure atomic_write creates an empty file, if we do nothing.

    This gives 'atomic_write' the same semantics as 'open', when
    the file does not exist.
    """
    path = tmp_path / "testfile"

    assert not path.exists()  # Sanity check

    with utils.atomic_write(path) as _:
        pass
    assert path.stat().st_size == 0

    with utils.atomic_write(path, return_path=True) as _:
        pass
    assert path.stat().st_size == 0

    #
    with utils.atomic_write(path) as fp:
        fp.write("content")
    assert path.stat().st_size > 0

    with utils.atomic_write(path) as _:
        pass
    assert path.stat().st_size == 0

    with utils.atomic_write(path) as fp:
        fp.write("content")
    assert path.stat().st_size > 0

    with utils.atomic_write(path, return_path=True) as _:
        pass
    assert path.stat().st_size == 0


def test_atomic_write_nested(tmp_path: Path) -> None:
    """Ensure nested calls to atomic_write on the same file work as expected."""
    path = tmp_path / "testfile"
    with utils.atomic_write(path, mode="w") as fp, utils.atomic_write(
        path,
        mode="w",
    ) as fp2:
        fp.write("one")
        fp2.write("two")
    # Outer call wins
    with path.open() as fp:
        assert fp.read() == "one"


@pytest.mark.parametrize("atomically", [False, True])
def test_load_dataframes_with_folder(
    tmp_path: Path,
    atomically: bool,  # noqa: FBT001
) -> None:
    """Test `utils.load_dataframes` with `folder`."""
    folder = tmp_path / "test_folder"
    folder.mkdir()
    learner = adaptive.Learner1D(lambda x: x, bounds=(-10, 10))
    fname = str(folder / "test.pickle")
    save_df = utils.save_dataframe(fname, atomically=atomically)

    learner.ask(10)
    learner.tell(10, 42)

    save_df(learner)

    df = utils.load_dataframes([fname], concat=False)[0]

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_at_least_adaptive_version() -> None:
    """Test `utils._at_least_adaptive_version`."""
    assert utils._at_least_adaptive_version("0.0.0", raises=False)
    assert utils._at_least_adaptive_version("0.0.0")
    assert not utils._at_least_adaptive_version("100000.0.0", raises=False)
    with pytest.raises(RuntimeError, match="requires adaptive version"):
        assert utils._at_least_adaptive_version("100000.0.0", raises=True)
    utils._at_least_adaptive_version(adaptive.__version__)


def _is_key_in_global_cache(key: str | bytes) -> bool:
    return key in utils._GLOBAL_CACHE


@pytest.mark.parametrize("mode", ["file", "memory", "random_id"])
def test_executor_with_wrapped_function_that_is_loaded_with_cloudpickle(
    *,
    mode: Literal["memory", "random_id", "file"],
) -> None:
    """Test executor with WrappedFunction that is loaded with cloudpickle."""
    if mode == "random_id" and platform.system() == "Darwin":
        pytest.skip("Not possible on MacOS")

    # Define a simple test function
    def square(x: int) -> int:
        return x * x

    # Serialize the function using cloudpickle
    serialized_function = cloudpickle.dumps(square)

    # Remove the function from the current scope
    del square

    # Load the serialized function using cloudpickle
    loaded_function = cloudpickle.loads(serialized_function)

    # Wrap the loaded function using WrappedFunction
    wrapped_function = utils.WrappedFunction(
        loaded_function,
        mode=mode,
    )

    # Run the wrapped function using ProcessPoolExecutor
    ex = ProcessPoolExecutor()

    # Check if the global cache contains the key in the cache
    # Note that behaviour on Linux and MacOS is different due to 'fork' vs 'spawn'
    fut_is_key = ex.submit(_is_key_in_global_cache, wrapped_function._cache_key)
    is_key = fut_is_key.result()
    if platform.system() == "Darwin":
        assert not is_key
    elif platform.system() == "Linux":
        assert is_key

    # Note that passing the loaded_function directly to ex.submit will not work!
    fut = ex.submit(wrapped_function, 4)
    result = fut.result()

    assert result == 16, f"Expected 16, but got {result}"

    # Check if the global cache contains the key in the executor
    fut_is_key = ex.submit(_is_key_in_global_cache, wrapped_function._cache_key)
    is_key = fut_is_key.result()
    assert is_key


def test_datetime_now() -> None:
    """Test `utils.datetime_now`."""
    # Test that the functions are compatible and can roundtrip a datetime string
    now = utils._now()
    now_roundtripped = utils._from_datetime(now)

    assert now == now_roundtripped.isoformat()
    assert utils._time_between(now, now) == 0
