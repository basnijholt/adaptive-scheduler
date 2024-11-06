"""Test the MultiRunManager class."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import patch

import ipywidgets as ipw
import pytest

from adaptive_scheduler._server_support.multi_run_manager import MultiRunManager
from adaptive_scheduler._server_support.run_manager import RunManager

if TYPE_CHECKING:
    from pathlib import Path

    import adaptive

    from .helpers import MockScheduler


@pytest.fixture()
def mock_run_manager(
    mock_scheduler: MockScheduler,
    learners: list[adaptive.Learner1D]
    | list[adaptive.BalancingLearner]
    | list[adaptive.SequenceLearner],
    fnames: list[str] | list[Path],
) -> RunManager:
    """Create a mock RunManager for testing."""
    return RunManager(mock_scheduler, learners[:1], fnames[:1], job_name="test-rm")


def test_multi_run_manager_init() -> None:
    """Test the initialization of MultiRunManager."""
    mrm = MultiRunManager()
    assert isinstance(mrm, MultiRunManager)
    assert mrm.run_managers == {}


def test_multi_run_manager_add_run_manager(mock_run_manager: RunManager) -> None:
    """Test adding a RunManager to MultiRunManager."""
    mrm = MultiRunManager()
    mrm.add_run_manager(mock_run_manager)
    assert len(mrm.run_managers) == 1
    assert "test-rm" in mrm.run_managers
    assert mrm.run_managers["test-rm"] == mock_run_manager


def test_multi_run_manager_add_duplicate_run_manager(mock_run_manager: RunManager) -> None:
    """Test adding a duplicate RunManager to MultiRunManager."""
    mrm = MultiRunManager()
    mrm.add_run_manager(mock_run_manager)
    with pytest.raises(ValueError, match="A RunManager with the name 'test-rm' already exists."):
        mrm.add_run_manager(mock_run_manager)


@pytest.mark.asyncio()
async def test_multi_run_manager_add_run_manager_with_start(mock_run_manager: RunManager) -> None:
    """Test adding a RunManager to MultiRunManager with start=True."""
    mrm = MultiRunManager()
    mrm.add_run_manager(mock_run_manager, start=True)
    await asyncio.sleep(0.1)
    assert mock_run_manager.status() == "running"
    mock_run_manager.cancel()


def test_multi_run_manager_add_run_manager_with_invalid_wait_for(
    mock_run_manager: RunManager,
) -> None:
    """Test adding a RunManager with an invalid wait_for parameter."""
    mrm = MultiRunManager()
    with pytest.raises(KeyError, match="No RunManager with the name 'non-existent' exists."):
        mrm.add_run_manager(mock_run_manager, start=True, wait_for="non-existent")


def test_multi_run_manager_add_run_manager_with_wait_for_without_start(
    mock_run_manager: RunManager,
) -> None:
    """Test adding a RunManager with wait_for but without start=True."""
    mrm = MultiRunManager()
    mrm.add_run_manager(
        RunManager(
            mock_run_manager.scheduler,
            mock_run_manager.learners,
            mock_run_manager.fnames,
            job_name="rm1",
        ),
    )
    with pytest.raises(ValueError, match="`start` must be True if `wait_for` is used."):
        mrm.add_run_manager(mock_run_manager, wait_for="rm1")


def test_multi_run_manager_remove_run_manager(mock_run_manager: RunManager) -> None:
    """Test removing a RunManager from MultiRunManager."""
    mrm = MultiRunManager()
    mrm.add_run_manager(mock_run_manager)
    mrm.remove_run_manager("test-rm")
    assert len(mrm.run_managers) == 0


def test_multi_run_manager_remove_non_existent_run_manager() -> None:
    """Test removing a non-existent RunManager from MultiRunManager."""
    mrm = MultiRunManager()
    with pytest.raises(KeyError, match="No RunManager with the name 'non-existent' exists."):
        mrm.remove_run_manager("non-existent")


@pytest.mark.asyncio()
async def test_multi_run_manager_start_all(mock_run_manager: RunManager) -> None:
    """Test starting all RunManagers in MultiRunManager."""
    mrm = MultiRunManager()
    mrm.add_run_manager(mock_run_manager)
    mrm.start_all()
    await asyncio.sleep(0.1)
    assert mock_run_manager.status() == "running"
    mock_run_manager.cancel()


@pytest.mark.asyncio()
async def test_multi_run_manager_cancel_all(mock_run_manager: RunManager) -> None:
    """Test cancelling all RunManagers in MultiRunManager."""
    mrm = MultiRunManager()
    mrm.add_run_manager(mock_run_manager)
    mrm.start_all()
    await asyncio.sleep(0.1)
    mrm.cancel_all()
    await asyncio.sleep(0.1)
    assert mock_run_manager.status() == "cancelled"


def test_multi_run_manager_create_widget(mock_run_manager: RunManager) -> None:
    """Test creating the widget for MultiRunManager."""
    mrm = MultiRunManager()
    mrm.add_run_manager(mock_run_manager)
    vbox = mrm._create_widget()
    assert isinstance(vbox, ipw.VBox)
    tab = vbox.children[1]
    assert isinstance(tab, ipw.Tab)
    assert len(tab.children) == 1
    assert tab.get_title(0) == "test-rm"


def test_multi_run_manager_update_widget(mock_run_manager: RunManager) -> None:
    """Test updating the widget for MultiRunManager."""
    mrm = MultiRunManager()
    mrm.add_run_manager(mock_run_manager)
    mrm._widget = mrm._create_widget()
    new_rm = RunManager(
        mock_run_manager.scheduler,
        mock_run_manager.learners,
        mock_run_manager.fnames,
        job_name="new-rm",
    )
    mrm.add_run_manager(new_rm)
    assert len(mrm._widget.children) == 2
    assert mrm._widget.children[1].get_title(1) == "new-rm"


def test_multi_run_manager_info(mock_run_manager: RunManager) -> None:
    """Test the info method of MultiRunManager."""
    mrm = MultiRunManager()
    mrm.add_run_manager(mock_run_manager)
    vbox = mrm.info()
    assert isinstance(vbox, ipw.VBox)
    tab = vbox.children[1]
    assert isinstance(tab, ipw.Tab)
    assert len(tab.children) == 1


def test_multi_run_manager_repr_html(mock_run_manager: RunManager) -> None:
    """Test the _repr_html_ method of MultiRunManager."""
    mrm = MultiRunManager()
    mrm.add_run_manager(mock_run_manager)
    with patch("IPython.display.display") as mocked_display:
        mrm._repr_html_()
        assert mocked_display.called
