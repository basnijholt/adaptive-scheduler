"""Tests for `adaptive_scheduler.utils`."""
import time
from datetime import datetime, timedelta
from pathlib import Path

import adaptive
import pandas as pd
import pytest

from adaptive_scheduler import utils


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


def test_split_in_balancing_learners() -> None:
    """Test `utils.split_in_balancing_learners`."""
    learners = [
        adaptive.Learner1D(lambda x: x, bounds=(-10, 10)),
        adaptive.Learner1D(lambda x: x, bounds=(-10, 10)),
        adaptive.Learner1D(lambda x: x, bounds=(-10, 10)),
    ]
    fnames = ["learner1.pickle", "learner2.pickle", "learner3.pickle"]
    n_parts = 2
    new_learners, new_fnames = utils.split_in_balancing_learners(
        learners,
        fnames,
        n_parts,
    )
    assert len(new_learners) == n_parts
    assert all(isinstance(lrn, adaptive.BalancingLearner) for lrn in new_learners)
    assert new_fnames == [(fnames[0], fnames[1]), (fnames[2],)]


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
        adaptive.SequenceLearner(lambda x: x, sequence=list(range(0, 5))),
        adaptive.SequenceLearner(lambda x: x, sequence=list(range(5, 10))),
    ]
    big_learner = utils.combine_sequence_learners(learners)
    assert isinstance(big_learner, adaptive.SequenceLearner)
    assert list(big_learner.sequence) == list(range(10))


def test_copy_from_sequence_learner() -> None:
    """Test `utils.copy_from_sequence_learner`."""
    learner1 = adaptive.SequenceLearner(lambda x: x, sequence=list(range(0, 5)))
    learner2 = adaptive.SequenceLearner(lambda x: x, sequence=list(range(0, 5)))
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
    assert fname == "x_1__y_2__z_3.pickle"


def test_add_constant_to_fname() -> None:
    """Test `utils.add_constant_to_fname`."""
    combo = {"x": 1, "y": 2, "z": 3}
    constant = {"a": 42}
    old_fname, new_fname = utils.add_constant_to_fname(combo, constant, dry_run=True)
    assert old_fname == "x_1__y_2__z_3.pickle"
    assert new_fname == "a_42__x_1__y_2__z_3.pickle"


def test_maybe_round() -> None:
    """Test `utils.maybe_round`."""
    assert utils.maybe_round(3.14159265, 3) == 3.14  # noqa: PLR2004
    assert utils.maybe_round(1 + 2j, 3) == 1 + 2j
    assert utils.maybe_round("test", 3) == "test"


def test_round_sigfigs() -> None:
    """Test `utils.round_sigfigs`."""
    assert utils.round_sigfigs(3.14159265, 3) == 3.14  # noqa: PLR2004
    assert utils.round_sigfigs(123456789, 3) == 123000000  # noqa: PLR2004


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
        adaptive.SequenceLearner(lambda x: x, sequence=list(range(0, 5))),
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
        adaptive.SequenceLearner(lambda x: x, sequence=list(range(0, 5))),
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
    assert cached_function(2) == 4  # noqa: PLR2004
    assert cached_function.cache_dict == {"{'x': 2}": 4}
    assert cached_function(3) == 9  # noqa: PLR2004
    assert cached_function.cache_dict == {"{'x': 2}": 4, "{'x': 3}": 9}


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
        adaptive.SequenceLearner(lambda x: x, sequence=list(range(0, 5))),
    ]
    fnames = [tmp_path / "learner1.pickle", tmp_path / "learner2.pickle"]

    utils.cloudpickle_learners(learners, fnames, with_progress_bar=False)

    for fname in fnames:
        fname_learner = utils.fname_to_learner_fname(fname)
        assert fname_learner.exists()


def test_save_dataframe(tmp_path: Path) -> None:
    """Test `utils.save_dataframe`."""
    learner = adaptive.Learner1D(lambda x: x, bounds=(-10, 10))
    fname = str(tmp_path / ("test.pickle"))
    save_df = utils.save_dataframe(fname)

    learner.ask(10)
    learner.tell(10, 42)

    save_df(learner)

    assert Path(utils.fname_to_dataframe(fname)).exists()


def test_load_dataframes(tmp_path: Path) -> None:
    """Test `utils.load_dataframes`."""
    learner = adaptive.Learner1D(lambda x: x, bounds=(-10, 10))
    fname = str(tmp_path / "test.pickle")
    save_df = utils.save_dataframe(fname)

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
    assert fname == f"{folder}/x_1__y_2__z_3.pickle"


def test_combo2fname_with_folder() -> None:
    """Test `utils.combo2fname` with `folder`."""
    combo = {"x": 1, "y": 2, "z": 3}
    folder = "test_folder"
    fname = utils.combo2fname(combo, folder=folder)
    assert fname == f"{folder}/x_1__y_2__z_3.pickle"


def test_add_constant_to_fname_with_folder() -> None:
    """Test `utils.add_constant_to_fname` with `folder`."""
    combo = {"x": 1, "y": 2, "z": 3}
    constant = {"a": 42}
    folder = "test_folder"
    old_fname, new_fname = utils.add_constant_to_fname(
        combo,
        constant,
        folder=folder,
        dry_run=True,
    )
    assert old_fname == f"{folder}/x_1__y_2__z_3.pickle"
    assert new_fname == f"{folder}/a_42__x_1__y_2__z_3.pickle"


def test_fname_to_dataframe_with_folder() -> None:
    """Test `utils.fname_to_dataframe` with `folder`."""
    fname = "test_folder/test.pickle"
    df_fname = utils.fname_to_dataframe(fname)
    assert df_fname == "test_folder/dataframe.test.parquet"


def test_load_dataframes_with_folder(tmp_path: Path) -> None:
    """Test `utils.load_dataframes` with `folder`."""
    folder = tmp_path / "test_folder"
    folder.mkdir()
    learner = adaptive.Learner1D(lambda x: x, bounds=(-10, 10))
    fname = str(folder / "test.pickle")
    save_df = utils.save_dataframe(fname)

    learner.ask(10)
    learner.tell(10, 42)

    save_df(learner)

    df = utils.load_dataframes([fname], concat=False)[0]

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
