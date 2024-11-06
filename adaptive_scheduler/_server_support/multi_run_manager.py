from __future__ import annotations

from typing import TYPE_CHECKING

import ipywidgets as ipw
from IPython.display import display

from adaptive_scheduler.widgets import _disable_widgets_output_scrollbar, info

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
        self._widget: ipw.VBox | None = None
        self._tab_widget: ipw.Tab | None = None
        self._info_widgets: dict[str, ipw.Widget] = {}
        self._update_all_button: ipw.Button | None = None
        if run_managers:
            for rm in run_managers:
                self.add_run_manager(rm)

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
        self._info_widgets[run_manager.job_name] = info(
            run_manager,
            display_widget=False,
            disable_widgets_output_scrollbar=False,
        )

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
            self._info_widgets.pop(name)
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

    def _create_widget(self) -> ipw.VBox:
        """Create the widget for displaying RunManager info and update button."""
        self._tab_widget = ipw.Tab()
        children = list(self._info_widgets.values())
        self._tab_widget.children = children
        for i, name in enumerate(self.run_managers.keys()):
            self._tab_widget.set_title(i, name)

        self._update_all_button = ipw.Button(
            description="Update All",
            button_style="info",
            tooltip="Update all RunManagers",
        )
        self._update_all_button.on_click(self._update_all_callback)

        return ipw.VBox([self._update_all_button, self._tab_widget])

    def _update_widget(self) -> None:
        """Update the widget when RunManagers are added or removed."""
        if self._tab_widget is not None:
            current_children = list(self._tab_widget.children)
            new_children = list(self._info_widgets.values())

            # Create a new tuple of children
            updated_children = tuple(child for child in current_children if child in new_children)

            # Add new children
            for widget in new_children:
                if widget not in updated_children:
                    updated_children += (widget,)

            # Update the widget's children
            self._tab_widget.children = updated_children

            # Update titles
            for i, name in enumerate(self.run_managers.keys()):
                self._tab_widget.set_title(i, name)

    def _update_all_callback(self, _: ipw.Button) -> None:
        """Callback function for the Update All button."""
        assert self._tab_widget is not None
        for widget in self._tab_widget.children:
            update_button = widget.children[0].children[1].children[0]
            assert update_button.description == "update info"
            update_button.click()

    def info(self) -> ipw.VBox:
        """Display info about all RunManagers in a widget with an Update All button."""
        if self._widget is None:
            _disable_widgets_output_scrollbar()
            self._widget = self._create_widget()
        return self._widget

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        return self.info()

    def display(self) -> None:
        """Display the widget."""
        display(self.info())
