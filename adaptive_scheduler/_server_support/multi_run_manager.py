from __future__ import annotations

from typing import TYPE_CHECKING

import ipywidgets as ipw

from adaptive_scheduler.widgets import info

if TYPE_CHECKING:
    from adaptive_scheduler._server_support.run_manager import RunManager


class MultiRunManager:
    """A manager that can contain multiple RunManagers.

    Parameters
    ----------
    run_managers
        Initial list of RunManagers to include.

    Attributes
    ----------
    run_managers
        Dictionary of managed RunManagers, keyed by their names.

    """

    def __init__(self, run_managers: list[RunManager] | None = None) -> None:
        self.run_managers: dict[str, RunManager] = {}
        if run_managers:
            for rm in run_managers:
                self.add_run_manager(rm)
        self._widget: ipw.Tab | None = None

    def add_run_manager(
        self,
        run_manager: RunManager,
        *,
        start: bool = False,
        wait_for: str | None = None,
    ) -> None:
        """Add a new RunManager to the MultiRunManager.

        Parameters
        ----------
        run_manager
            The RunManager to add.
        start
            Whether to start the RunManager immediately after adding it.
        wait_for
            The name of another RunManager to wait for before starting this one.
            Only applicable if start is True.

        Raises
        ------
        ValueError
            If a RunManager with the same name already exists.
        KeyError
            If the specified wait_for RunManager does not exist.

        """
        if run_manager.job_name in self.run_managers:
            msg = f"A RunManager with the name '{run_manager.job_name}' already exists."
            raise ValueError(msg)

        self.run_managers[run_manager.job_name] = run_manager

        if start:
            if wait_for:
                if wait_for not in self.run_managers:
                    msg = f"No RunManager with the name '{wait_for}' exists."
                    raise KeyError(msg)
                run_manager.start(wait_for=self.run_managers[wait_for])
            else:
                run_manager.start()
        elif wait_for:
            msg = "`start` must be True if `wait_for` is used."
            raise ValueError(msg)

        if self._widget is not None:
            self._update_widget()

    def remove_run_manager(self, name: str) -> None:
        """Remove a RunManager from the MultiRunManager.

        Parameters
        ----------
        name
            The name of the RunManager to remove.

        Raises
        ------
        KeyError
            If no RunManager with the given name exists.

        """
        if name in self.run_managers:
            rm = self.run_managers.pop(name)
            rm.cancel()
            if self._widget is not None:
                self._update_widget()
        else:
            msg = f"No RunManager with the name '{name}' exists."
            raise KeyError(msg)

    def start_all(self) -> None:
        """Start all RunManagers."""
        for run_manager in self.run_managers.values():
            run_manager.start()

    def cancel_all(self) -> None:
        """Cancel all RunManagers."""
        for run_manager in self.run_managers.values():
            run_manager.cancel()

    def _create_widget(self) -> ipw.Tab:
        """Create the tab widget for displaying RunManager info."""
        tab = ipw.Tab()
        children = [info(rm) for rm in self.run_managers.values()]
        tab.children = children
        for i, (name, _) in enumerate(self.run_managers.items()):
            tab.set_title(i, f"RunManager: {name}")
        return tab

    def _update_widget(self) -> None:
        """Update the widget when RunManagers are added or removed."""
        if self._widget is not None:
            self._widget.children = [info(rm, show_info=False) for rm in self.run_managers.values()]
            for i, (name, _) in enumerate(self.run_managers.items()):
                self._widget.set_title(i, f"RunManager: {name}")

    def info(self) -> ipw.Tab:
        """Display info about all RunManagers in a tab widget."""
        if self._widget is None:
            self._widget = self._create_widget()
        return self._widget

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        return self.info()._repr_html_()
