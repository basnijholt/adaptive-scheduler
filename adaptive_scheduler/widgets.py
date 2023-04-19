"""Adaptive Scheduler notebook widgets."""
from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from typing import Any, Callable

    from ipywidgets import VBox, Widget

    from adaptive_scheduler.server_support import RunManager
    from adaptive_scheduler.utils import FnamesTypes


def _get_fnames(run_manager: RunManager, *, only_running: bool) -> list[Path]:
    if only_running:
        fnames = []
        for entry in run_manager.database_manager.as_dicts():
            if entry["is_done"]:
                continue
            if entry["log_fname"] is not None:
                fnames.append(entry["log_fname"])
            fnames += entry["output_logs"]
        return sorted(map(Path, fnames))
    pattern = f"{run_manager.job_name}-*"
    logs = set(Path(run_manager.scheduler.log_folder).glob(pattern))
    logs |= set(Path(".").glob(pattern))
    return sorted(logs)


def _failed_job_logs(
    fnames: list[Path],
    run_manager: RunManager,
    *,
    only_running: bool,
) -> list[Path]:
    running = {
        Path(e["log_fname"]).stem
        for e in run_manager.database_manager.as_dicts()
        if e["log_fname"] is not None
    }
    fnames_set = {
        fname.stem for fname in fnames if fname.suffix != run_manager.scheduler.ext
    }
    failed_set = fnames_set - running
    failed = [Path(f) for stem in failed_set for f in glob(f"{stem}*")]

    def maybe_append(fname: str, other_dir: Path, lst: list[Path]) -> None:
        p = Path(fname)
        p_other = other_dir / p.name
        if p.exists():
            lst.append(p)
        elif p_other.exists():
            lst.append(p_other)

    if not only_running and run_manager.move_old_logs_to is not None:
        base = Path(run_manager.move_old_logs_to)
        for e in run_manager.database_manager.failed:
            if not e["is_done"]:
                for f in e["output_logs"]:
                    maybe_append(f, base, failed)
            maybe_append(e["log_fname"], base, failed)
    return failed


def _files_that_contain(fnames: list[Path], text: str) -> list[Path]:
    def contains(fname: Path, text: str) -> bool:
        with fname.open("r", encoding="utf-8") as f:
            return any(text in line for line in f)

    return [fname for fname in fnames if contains(fname, text)]


def _sort_fnames(
    sort_by: str,
    run_manager: RunManager,
    fnames: list[Path],
) -> list[Path] | list[tuple[str, Path]]:
    def _try(f: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def _f(x: Any) -> Any:
            try:
                return f(x)
            except Exception:  # noqa: BLE001
                return x

        return _f

    def _sort_key(value: tuple[float | str, str]) -> tuple[float | int, str]:
        x, fname = value
        if isinstance(x, str):
            return -1, fname
        return float(x), fname

    mapping = {
        "Alphabetical": (None, lambda _: ""),
        "CPU %": ("cpu_usage", lambda x: f"{x:.1f}%"),
        "Mem %": ("mem_usage", lambda x: f"{x:.1f}%"),
        "Last editted": (
            "timestamp",
            lambda x: f"{(np.datetime64(datetime.now()) - x) / 1e9}s ago",  # noqa: DTZ005
        ),
        "Loss": ("latest_loss", lambda x: f"{x:.2f}"),
        "npoints": ("npoints", lambda x: f"{x} pnts"),
        "Elapsed time": ("elapsed_time", lambda x: f"{x / 1e9}s"),
    }

    def extract(df: pd.DataFrame, fname: Path, key: str) -> Any | str:
        df_sel = df[df.log_fname.str.contains(fname.name)]
        values = df_sel[key].to_numpy()
        if values:
            return values[0]
        return "?"

    if sort_by != "Alphabetical":
        fname_mapping = defaultdict(list)
        for fname in fnames:
            fname_mapping[fname.stem].append(fname)

        df = run_manager.parse_log_files()
        if df.empty:
            return fnames
        log_fnames = set(df.log_fname.apply(Path))
        df_key, transform = mapping[sort_by]
        assert df_key is not None  # for mypy
        stems = [fname.stem for fname in log_fnames]
        vals = [extract(df, fname, df_key) for fname in log_fnames]
        val_stem = sorted(zip(vals, stems), key=_sort_key, reverse=True)

        result: list[tuple[str, Path]] = []
        for val, stem in val_stem:
            val = _try(transform)(val)  # noqa: PLW2901
            for fname in fname_mapping[stem]:
                result.append((f"{val}: {fname.name}", fname))

        missing = fname_mapping.keys() - set(stems)
        for stem in sorted(missing):
            for fname in fname_mapping[stem]:
                result.append((f"?: {fname.name}", fname))
        return result

    return fnames


def _read_file(fname: Path, max_lines: int = 500) -> str:
    try:
        with fname.open("r", encoding="utf-8") as f:
            lines = f.readlines()

            if len(lines) > max_lines:
                lines = lines[-max_lines:]
                lines.insert(0, f"Only displaying the last {max_lines} lines!")
                lines.insert(1, "\n")
            return "".join(lines)
    except UnicodeDecodeError:
        return f"Could not decode file ({fname})!"
    except Exception as e:  # noqa: BLE001
        return f"Exception with trying to read {fname}:\n{e}."


def log_explorer(run_manager: RunManager) -> VBox:  # noqa: C901, PLR0915
    """Log explorer widget."""
    from ipywidgets import (
        HTML,
        Button,
        Checkbox,
        Dropdown,
        Layout,
        Text,
        Textarea,
        VBox,
    )

    def _update_fname_dropdown(  # noqa: PLR0913
        run_manager: RunManager,
        fname_dropdown: Dropdown,
        only_running_checkbox: Checkbox,
        only_failed_checkbox: Checkbox,
        sort_by_dropdown: Dropdown,
        contains_text: Text,
    ) -> Callable[[Any], None]:
        def on_click(_: Any) -> None:
            current_value = fname_dropdown.value
            fnames = _get_fnames(run_manager, only_running=only_running_checkbox.value)
            if only_failed_checkbox.value:
                fnames = _failed_job_logs(
                    fnames,
                    run_manager,
                    only_running=only_running_checkbox.value,
                )
            if not contains_text.value.strip():
                fnames = _files_that_contain(fnames, contains_text.value.strip())
            sorted_fnames = _sort_fnames(sort_by_dropdown.value, run_manager, fnames)
            fname_dropdown.options = sorted_fnames
            with suppress(Exception):
                fname_dropdown.value = current_value
            fname_dropdown.disabled = not sorted_fnames

        return on_click

    def _last_editted(fname: Path) -> float:
        try:
            return fname.stat().st_mtime
        except FileNotFoundError:
            return -1.0

    async def _tail_log(fname: Path, textarea: Textarea) -> None:
        t = -2.0  # to make sure the update always triggers
        while True:
            await asyncio.sleep(2)
            try:
                t_new = _last_editted(fname)
                if t_new > t:
                    textarea.value = _read_file(fname, run_manager.max_log_lines)
                    t = t_new
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: S110, BLE001
                pass

    def _tail(  # noqa: PLR0913
        dropdown: Dropdown,
        tail_button: Button,
        textarea: Textarea,
        update_button: Button,
        only_running_checkbox: Checkbox,
        only_failed_checkbox: Checkbox,
    ) -> Callable[[Any], None]:
        tail_task = None
        ioloop = asyncio.get_running_loop()

        def on_click(_: Any) -> None:
            nonlocal tail_task
            if tail_task is None:
                fname = dropdown.options[dropdown.index]
                tail_task = ioloop.create_task(_tail_log(fname, textarea))
                tail_button.description = "cancel tail log"
                tail_button.button_style = "danger"
                tail_button.icon = "window-close"
                dropdown.disabled = True
                update_button.disabled = True
                only_running_checkbox.disabled = True
                only_failed_checkbox.disabled = True
            else:
                tail_button.description = "tail log"
                tail_button.button_style = "info"
                tail_button.icon = "refresh"
                dropdown.disabled = False
                only_running_checkbox.disabled = False
                only_failed_checkbox.disabled = False
                update_button.disabled = False
                tail_task.cancel()
                tail_task = None

        return on_click

    def _on_dropdown_change(textarea: Textarea) -> Callable[[dict[str, Any]], None]:
        def on_change(change: dict[str, Any]) -> None:
            if (
                change["type"] == "change"
                and change["name"] == "value"
                and change["new"] is not None
            ):
                textarea.value = _read_file(change["new"], run_manager.max_log_lines)

        return on_change

    def _click_button_on_change(button: Button) -> Callable[[dict[str, Any]], None]:
        def on_change(change: dict[str, Any]) -> None:
            if change["type"] == "change" and change["name"] == "value":
                button.click()

        return on_change

    fnames = _get_fnames(run_manager, only_running=False)
    # no need to sort `fnames` because the default sort_by option is alphabetical
    text = _read_file(fnames[0], run_manager.max_log_lines) if fnames else ""
    textarea = Textarea(text, layout={"width": "auto"}, rows=20)
    sort_by_dropdown = Dropdown(
        description="Sort by",
        options=["Alphabetical", "CPU %", "Mem %", "Last editted", "Loss", "npoints"],
    )
    contains_text = Text(description="Has string")
    fname_dropdown = Dropdown(description="File name", options=fnames)
    fname_dropdown.observe(_on_dropdown_change(textarea))
    only_running_checkbox = Checkbox(
        description="Only files of running jobs",
        indent=False,
    )
    only_failed_checkbox = Checkbox(
        description="Only files of failed jobs (might include false positives)",
        indent=False,
    )
    update_button = Button(
        description="update file list",
        button_style="info",
        icon="refresh",
    )
    update_button.on_click(
        _update_fname_dropdown(
            run_manager,
            fname_dropdown,
            only_running_checkbox,
            only_failed_checkbox,
            sort_by_dropdown,
            contains_text,
        ),
    )
    sort_by_dropdown.observe(_click_button_on_change(update_button))
    only_running_checkbox.observe(_click_button_on_change(update_button))
    only_failed_checkbox.observe(_click_button_on_change(update_button))
    tail_button = Button(description="tail log", button_style="info", icon="refresh")
    tail_button.on_click(
        _tail(
            fname_dropdown,
            tail_button,
            textarea,
            update_button,
            only_running_checkbox,
            only_failed_checkbox,
        ),
    )
    title = HTML("<h2><tt>adaptive_scheduler.widgets.log_explorer</tt></h2>")
    return VBox(
        [
            title,
            only_running_checkbox,
            only_failed_checkbox,
            update_button,
            sort_by_dropdown,
            contains_text,
            fname_dropdown,
            tail_button,
            textarea,
        ],
        layout=Layout(border="solid 2px gray"),
    )


def _bytes_to_human_readable(size_in_bytes: int) -> str:
    if size_in_bytes < 0:
        msg = "Size must be a positive integer"
        raise ValueError(msg)

    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    index = 0
    bytes_in_kb = 1024
    while size_in_bytes >= bytes_in_kb and index < len(units) - 1:
        size_in_bytes /= bytes_in_kb  # type: ignore[assignment]
        index += 1

    return f"{size_in_bytes:.2f} {units[index]}"


def _timedelta_to_human_readable(
    time_input: timedelta | int,
    *,
    short_format: bool = True,
) -> str:
    """Convert a timedelta object or an int (in seconds) into a human-readable format."""
    if isinstance(time_input, timedelta):
        total_seconds = int(time_input.total_seconds())
    elif isinstance(time_input, (int, float)):
        total_seconds = time_input
    else:
        msg = "Input must be a datetime.timedelta object or an int (in seconds)"
        raise TypeError(msg)

    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    periods = [
        ("day", days),
        ("hour", hours),
        ("minute", minutes),
        ("second", seconds),
    ]

    result = []
    for period, value in periods:
        if value:
            period_label = period if value == 1 else f"{period}s"
            if short_format:
                period_label = period_label[0]
            result.append(f"{value:.0f} {period_label}")

    return ", ".join(result)


def _total_size(fnames: FnamesTypes) -> int:
    """Return the total size of the files in `fnames`."""

    def flatten(
        items: FnamesTypes,
    ) -> Generator[str | Path, None, None]:
        """Flatten nested lists."""
        for item in items:
            if isinstance(item, (list, tuple)):
                yield from flatten(item)
            else:
                yield item

    flattened_fnames = list(flatten(fnames))
    return sum(
        os.path.getsize(str(fname))
        for fname in flattened_fnames
        if os.path.isfile(fname)  # noqa: PTH113
    )


def _info_html(run_manager: RunManager) -> str:
    queue = run_manager.scheduler.queue(me_only=True)
    dbm = run_manager.database_manager
    dbm.update(queue)
    jobs = [job for job in queue.values() if job["job_name"] in run_manager.job_names]
    n_running = sum(job["state"] in ("RUNNING", "R") for job in jobs)
    n_pending = sum(job["state"] in ("PENDING", "Q", "CONFIGURING") for job in jobs)
    n_done = sum(1 for job in dbm.as_dicts() if job["is_done"])
    n_failed = len(dbm.failed)
    n_failed_color = "red" if n_failed > 0 else "black"

    status = run_manager.status()
    color = {
        "cancelled": "orange",
        "not yet started": "orange",
        "running": "blue",
        "failed": "red",
        "finished": "green",
    }[status]

    def _table_row(i: int, key: str, value: Any) -> str:
        """Style the rows of a table. Based on the default Jupyterlab table style."""
        style = "text-align: right; padding: 0.5em 0.5em; line-height: 1.0;"
        if i % 2 == 1:
            style += " background: var(--md-grey-100);"
        return (
            f'<tr><th style="{style}">{key}</th><th style="{style}">{value}</th></tr>'
        )

    data_size = _total_size(dbm.fnames)
    info = [
        ("status", f'<font color="{color}">{status}</font>'),
        ("# running jobs", f'<font color="blue">{n_running}</font>'),
        ("# pending jobs", f'<font color="orange">{n_pending}</font>'),
        ("# finished jobs", f'<font color="green">{n_done}</font>'),
        ("# failed jobs", f'<font color="{n_failed_color}">{n_failed}</font>'),
        ("elapsed time", timedelta(seconds=run_manager.elapsed_time())),
        ("total data size", _bytes_to_human_readable(data_size)),
    ]
    if dbm._total_learner_size is not None:
        info.append(
            ("empty learner size", _bytes_to_human_readable(dbm._total_learner_size)),
        )

    starting_times = run_manager.job_starting_times()
    if starting_times:
        mean_starting_time = _timedelta_to_human_readable(np.mean(starting_times))
        std_starting_time = _timedelta_to_human_readable(np.std(starting_times))
        info.append(("avg job start time", mean_starting_time))
        info.append(("std job start time", std_starting_time))

    with suppress(Exception):
        df = run_manager.parse_log_files()
        t_last = (pd.Timestamp.now() - df.timestamp.max()).seconds

        overhead = df.mem_usage.mean()
        red_level = max(0, min(int(255 * overhead / 100), 255))
        overhead_color = f"#{red_level:02x}{255 - red_level:02x}{0:02x}"
        overhead_html_value = f'<font color="{overhead_color}">{overhead:.2f}%</font>'

        cpu = df.cpu_usage.mean()
        red_level = max(0, min(int(255 * cpu / 100), 255))
        cpu_color = f"#{red_level:02x}{red_level:02x}{0:02x}"
        cpu_html_value = f'<font color="{cpu_color}">{cpu:.2f}%</font>'

        from_logs = [
            ("# of points", df.npoints.sum()),
            ("mean CPU usage", cpu_html_value),
            ("mean memory usage", f"{df.mem_usage.mean().round(1)} %"),
            ("mean overhead", overhead_html_value),
            ("last log-entry", f"{t_last}s ago"),
        ]
        for key in ["npoints/s", "latest_loss", "nlearners"]:
            with suppress(Exception):
                from_logs.append((f"mean {key}", f"{df[key].mean().round(1)}"))
        msg = "this is extracted from the log files, so it might not be up-to-date"
        abbr = '<abbr title="{}">{}</abbr>'  # creates a tooltip
        info.extend([(abbr.format(msg, k), v) for k, v in from_logs])

    table = "\n".join(_table_row(i, k, v) for i, (k, v) in enumerate(info))

    return f"""
        <table>
        {table}
        </table>
    """


def _create_widget(
    data_provider: Callable[[], pd.DataFrame],
    update_button_text: str,
    *,
    use_itables_checkbox: bool = False,
    additional_widgets: list[Widget] | None = None,
) -> VBox:
    from IPython.display import display
    from ipywidgets import Button, Checkbox, Layout, Output, VBox
    from itables import show

    def _update_data_df(
        itables_checkbox: Checkbox,
        output_widget: Output,
    ) -> Callable[[Any], None]:
        def on_click(_: Any) -> None:
            with output_widget:
                output_widget.clear_output()
                df = data_provider()
                if itables_checkbox.value:
                    show(df)
                else:
                    with _display_all_dataframe_rows():
                        display(df)

        return on_click

    # Create widgets
    output_widget = Output()
    itables_checkbox = Checkbox(
        description="Use itables (interactive)",
        indent=False,
        value=use_itables_checkbox,
    )
    update_button = Button(
        description=update_button_text,
        button_style="info",
        icon="refresh",
    )

    # Update the DataFrame in the Output widget when the button is clicked or the checkbox is changed
    update_function = _update_data_df(
        itables_checkbox,
        output_widget,
    )
    update_button.on_click(update_function)
    itables_checkbox.observe(update_function, names="value")

    # Initialize the DataFrame display
    update_function(None)

    # Create a VBox and add the widgets to it
    widget_list = [itables_checkbox, update_button, output_widget]
    if additional_widgets:
        widget_list = additional_widgets + widget_list

    return VBox(
        widget_list,
        layout=Layout(border="solid 2px gray"),
    )


def queue_widget(run_manager: RunManager) -> VBox:
    """Create a widget that shows the current queue and allows to update it."""
    from ipywidgets import Checkbox

    me_only_checkbox = Checkbox(description="Only my jobs", indent=False, value=True)

    def get_queue_df() -> pd.DataFrame:
        queue = run_manager.scheduler.queue(me_only=me_only_checkbox.value)
        return pd.DataFrame(queue).transpose()

    return _create_widget(
        get_queue_df,
        "Update queue",
        additional_widgets=[me_only_checkbox],
    )


def database_widget(run_manager: RunManager) -> VBox:
    """Create a widget that shows the current database and allows to update it."""

    def get_database_df() -> pd.DataFrame:
        return run_manager.database_manager.as_df()

    return _create_widget(get_database_df, "Update database")


def _remove_widget(box: VBox, widget_to_remove: Widget) -> None:
    box.children = tuple(child for child in box.children if child != widget_to_remove)


def _toggle_widget(
    box: VBox,
    widget_key: str,
    widget_dict: dict[str, Widget | str],
    state_dict: dict[str, dict[str, Any]],
) -> Callable[[Any], None]:
    from ipywidgets import Button

    def on_click(_: Any) -> None:
        widget = state_dict[widget_key]["widget"]
        if widget is None:
            widget = state_dict[widget_key]["init_func"]()
            state_dict[widget_key]["widget"] = widget

        button = widget_dict[widget_key]
        assert isinstance(button, Button)
        show_description = state_dict[widget_key]["show_description"]
        hide_description = state_dict[widget_key]["hide_description"]

        if button.description == show_description:
            button.description = hide_description
            box.children = (*box.children, widget)
        else:
            button.description = show_description
            _remove_widget(box, widget)

    return on_click


def info(run_manager: RunManager) -> None:  # noqa: PLR0915
    """Display information about the `RunManager`.

    Returns an interactive ipywidget that can be
    visualized in a Jupyter notebook.
    """
    from IPython.display import display
    from ipywidgets import HTML, Button, Checkbox, HBox, Layout, VBox

    status = HTML(value=_info_html(run_manager))

    layout = Layout(width="200px")

    cancel_button = Button(
        description="cancel jobs",
        layout=layout,
        button_style="danger",
        icon="stop",
    )
    cleanup_button = Button(
        description="cleanup log and batch files",
        layout=layout,
        button_style="danger",
        icon="remove",
    )
    update_info_button = Button(
        description="update info",
        layout=layout,
        icon="refresh",
    )
    update_info_button.style.button_color = "lightgreen"
    load_learners_button = Button(
        description="load learners",
        layout=layout,
        button_style="info",
        icon="download",
    )
    show_logs_button = Button(
        description="show logs",
        layout=layout,
        button_style="info",
        icon="book",
    )
    show_queue_button = Button(
        description="show queue",
        layout=layout,
        button_style="info",
        icon="tasks",
    )
    show_db_button = Button(
        description="show database",
        layout=layout,
        button_style="info",
        icon="database",
    )
    widgets = {
        "update info": update_info_button,
        "cancel": HBox([cancel_button], layout=layout),
        "cleanup": HBox([cleanup_button], layout=layout),
        "load learners": load_learners_button,
        "show logs": show_logs_button,
        "show queue": show_queue_button,
        "show database": show_db_button,
    }

    def switch_to(
        box: HBox,
        *buttons: Button,
        _callable: Callable[[], None] | None = None,
    ) -> Callable[[Any], None]:
        def on_click(_: Any) -> None:
            box.children = tuple(buttons)
            if _callable is not None:
                _callable()

        return on_click

    box = VBox([])

    def update(_: Any) -> None:
        status.value = _info_html(run_manager)

    def load_learners(_: Any) -> None:
        run_manager.load_learners()

    state_dict = {
        "show logs": {
            "widget": None,
            "init_func": lambda: log_explorer(run_manager),
            "show_description": "show logs",
            "hide_description": "hide logs",
        },
        "show queue": {
            "widget": None,
            "init_func": lambda: queue_widget(run_manager),
            "show_description": "show queue",
            "hide_description": "hide queue",
        },
        "show database": {
            "widget": None,
            "init_func": lambda: database_widget(run_manager),
            "show_description": "show database",
            "hide_description": "hide database",
        },
    }

    def cancel() -> None:
        run_manager.cancel()
        update(None)

    def cleanup(*, include_old_logs: Checkbox) -> Callable[[], None]:
        def _callable() -> None:
            run_manager.cleanup(remove_old_logs_folder=include_old_logs.value)
            update(None)

        return _callable

    widgets["update info"].on_click(update)
    widgets["show logs"].on_click(_toggle_widget(box, "show logs", widgets, state_dict))
    widgets["show queue"].on_click(
        _toggle_widget(box, "show queue", widgets, state_dict),
    )
    widgets["show database"].on_click(
        _toggle_widget(box, "show database", widgets, state_dict),
    )
    widgets["load learners"].on_click(load_learners)

    # Cancel button with confirm/deny option
    confirm_cancel_button = Button(
        description="Confirm",
        button_style="success",
        icon="check",
    )
    deny_cancel_button = Button(description="Deny", button_style="danger", icon="close")

    cancel_button.on_click(
        switch_to(widgets["cancel"], confirm_cancel_button, deny_cancel_button),
    )
    deny_cancel_button.on_click(switch_to(widgets["cancel"], cancel_button))
    confirm_cancel_button.on_click(
        switch_to(widgets["cancel"], cancel_button, _callable=cancel),
    )

    # Cleanup button with confirm/deny option
    include_old_logs = Checkbox(
        value=False,
        description=f"Remove {run_manager.move_old_logs_to}/ folder",
        indent=False,
    )
    confirm_cleanup_button = Button(
        description="Confirm",
        button_style="success",
        icon="check",
    )
    deny_cleanup_button = Button(
        description="Deny",
        button_style="danger",
        icon="close",
    )

    cleanup_box = VBox(
        [HBox([confirm_cleanup_button, deny_cleanup_button]), include_old_logs],
    )
    cleanup_button.on_click(switch_to(widgets["cleanup"], cleanup_box))
    deny_cleanup_button.on_click(switch_to(widgets["cleanup"], cleanup_button))
    confirm_cleanup_button.on_click(
        switch_to(
            widgets["cleanup"],
            cleanup_button,
            _callable=cleanup(include_old_logs=include_old_logs),
        ),
    )

    box.children = (status, *tuple(widgets.values()))
    display(box)


@contextmanager
def _display_all_dataframe_rows(max_colwidth: int = 50) -> Generator[None, None, None]:
    """Display all rows in a `pandas.DataFrame`."""
    original_max_rows = pd.options.display.max_rows
    original_max_colwidth = pd.options.display.max_colwidth
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", max_colwidth)
    try:
        yield
    finally:
        pd.set_option("display.max_rows", original_max_rows)
        pd.set_option("display.max_colwidth", original_max_colwidth)
