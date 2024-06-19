"""Adaptive Scheduler notebook widgets."""

from __future__ import annotations

import asyncio
import os
from collections import defaultdict
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from adaptive_scheduler.utils import load_dataframes

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from typing import Any

    import ipywidgets as ipyw

    from adaptive_scheduler.scheduler import BaseScheduler
    from adaptive_scheduler.server_support import RunManager
    from adaptive_scheduler.utils import _DATAFRAME_FORMATS, FnamesTypes


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
    logs |= set(Path().glob(pattern))
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
    fnames_set = {fname.stem for fname in fnames if fname.suffix != run_manager.scheduler.ext}
    failed_set = fnames_set - running
    failed = [Path(f) for stem in failed_set for f in glob(f"{stem}*")]  # noqa: PTH207

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

    def _vec_timedelta(ts: pd.Timestamp) -> str:
        now = np.datetime64(datetime.now())  # noqa: DTZ005
        dt = np.timedelta64(now - ts, "s")  # type: ignore[operator]
        return f"{dt} ago"

    mapping = {
        "Alphabetical": (None, lambda _: ""),
        "CPU %": ("cpu_usage", lambda x: f"{x:.1f}%"),
        "Mem %": ("mem_usage", lambda x: f"{x:.1f}%"),
        "Last editted": ("timestamp", _vec_timedelta),
        "Loss": ("latest_loss", lambda x: f"{x:.2e}"),
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
        log_fnames = set(df.log_fname.apply(Path))  # type: ignore [arg-type]
        df_key, transform = mapping[sort_by]
        assert df_key is not None  # for mypy
        stems = [fname.stem for fname in log_fnames]
        vals = [extract(df, fname, df_key) for fname in log_fnames]
        val_stem = sorted(zip(vals, stems, strict=True), key=_sort_key, reverse=True)

        result: list[tuple[str, Path]] = []
        for val, stem in val_stem:
            val = _try(transform)(val)  # noqa: PLW2901
            for fname in fname_mapping[stem]:
                result.append((f"{val}: {fname.name}", fname))  # noqa: PERF401

        missing = fname_mapping.keys() - set(stems)
        for stem in sorted(missing):
            for fname in fname_mapping[stem]:
                result.append((f"?: {fname.name}", fname))  # noqa: PERF401
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


def log_explorer(run_manager: RunManager) -> ipyw.VBox:  # noqa: C901, PLR0915
    """Log explorer widget."""
    import ipywidgets as ipyw

    def _update_fname_dropdown(
        run_manager: RunManager,
        fname_dropdown: ipyw.Dropdown,
        only_running_checkbox: ipyw.Checkbox,
        only_failed_checkbox: ipyw.Checkbox,
        sort_by_dropdown: ipyw.Dropdown,
        contains_text: ipyw.Text,
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
            if contains_text.value.strip():
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

    async def _tail_log(fname: Path, textarea: ipyw.Textarea) -> None:
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

    def _tail(
        dropdown: ipyw.Dropdown,
        tail_button: ipyw.Button,
        textarea: ipyw.Textarea,
        update_button: ipyw.Button,
        only_running_checkbox: ipyw.Checkbox,
        only_failed_checkbox: ipyw.Checkbox,
        sort_by_dropdown: ipyw.Dropdown,
        contains_text: ipyw.Text,
    ) -> Callable[[Any], None]:
        tail_task = None
        ioloop = asyncio.get_running_loop()

        def on_click(_: Any) -> None:
            nonlocal tail_task
            tailing_log = tail_task is not None

            def update_ui_state(tailing: bool) -> None:  # noqa: FBT001
                tail_button.description = "cancel tail log" if tailing else "tail log"
                tail_button.button_style = "danger" if tailing else "info"
                tail_button.icon = "window-close" if tailing else "refresh"
                dropdown.disabled = tailing
                only_running_checkbox.disabled = tailing
                only_failed_checkbox.disabled = tailing
                update_button.disabled = tailing
                sort_by_dropdown.disabled = tailing
                contains_text.disabled = tailing

            if not tailing_log:
                fname = dropdown.options[dropdown.index]
                tail_task = ioloop.create_task(_tail_log(fname, textarea))
            else:
                assert tail_task is not None
                tail_task.cancel()
                tail_task = None

            update_ui_state(not tailing_log)

        return on_click

    def _on_dropdown_change(
        textarea: ipyw.Textarea,
    ) -> Callable[[dict[str, Any]], None]:
        def on_change(change: dict[str, Any]) -> None:
            if (
                change["type"] == "change"
                and change["name"] == "value"
                and change["new"] is not None
            ):
                textarea.value = _read_file(change["new"], run_manager.max_log_lines)

        return on_change

    def _click_button_on_change(
        button: ipyw.Button,
    ) -> Callable[[dict[str, Any]], None]:
        def on_change(change: dict[str, Any]) -> None:
            if change["type"] == "change" and change["name"] == "value":
                button.click()

        return on_change

    fnames = _get_fnames(run_manager, only_running=False)
    # no need to sort `fnames` because the default sort_by option is alphabetical
    text = _read_file(fnames[0], run_manager.max_log_lines) if fnames else ""
    textarea = ipyw.Textarea(text, layout={"width": "auto"}, rows=20)
    sort_by_dropdown = ipyw.Dropdown(
        description="Sort by",
        options=["Alphabetical", "CPU %", "Mem %", "Last editted", "Loss", "npoints"],
    )
    contains_text = ipyw.Text(description="Has string")
    fname_dropdown = ipyw.Dropdown(description="File name", options=fnames)
    fname_dropdown.observe(_on_dropdown_change(textarea))
    only_running_checkbox = ipyw.Checkbox(
        description="Only files of running jobs",
        indent=False,
    )
    only_failed_checkbox = ipyw.Checkbox(
        description="Only files of failed jobs (might include false positives)",
        indent=False,
    )
    update_button = ipyw.Button(
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
    tail_button = ipyw.Button(
        description="tail log",
        button_style="info",
        icon="refresh",
    )
    tail_button.on_click(
        _tail(
            fname_dropdown,
            tail_button,
            textarea,
            update_button,
            only_running_checkbox,
            only_failed_checkbox,
            sort_by_dropdown,
            contains_text,
        ),
    )
    vbox = ipyw.VBox(
        [
            only_running_checkbox,
            only_failed_checkbox,
            update_button,
            sort_by_dropdown,
            contains_text,
            fname_dropdown,
            tail_button,
            textarea,
        ],
        layout=ipyw.Layout(border="solid 2px gray"),
    )
    _add_title("adaptive_scheduler.widgets.log_explorer", vbox)
    return vbox


def _add_title(title: str, vbox: ipyw.VBox) -> None:
    import ipywidgets as ipyw

    title = ipyw.HTML(f"<h2><tt>{title}</tt></h2>")
    vbox.children = (title, *vbox.children)


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
    time_input: timedelta | float,
    *,
    short_format: bool = True,
) -> str:
    """Convert a timedelta object or an int (in seconds) into a human-readable format."""
    if isinstance(time_input, timedelta):
        total_seconds = int(time_input.total_seconds())
    elif isinstance(time_input, int | float):
        total_seconds = int(time_input)
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
            if isinstance(item, list | tuple):
                yield from flatten(item)
            else:
                yield item

    flattened_fnames = list(flatten(fnames))
    return sum(
        os.path.getsize(str(fname))  # noqa: PTH202
        for fname in flattened_fnames
        if os.path.isfile(fname)  # noqa: PTH113
    )


def _interp_red_green(
    percent: float,
    pct_red: int = 30,
    pct_green: int = 10,
) -> tuple[int, int, int]:
    if pct_green < pct_red:
        if percent <= pct_green:
            return 0, 255, 0
        if percent >= pct_red:
            return 255, 0, 0
    else:
        if percent >= pct_green:
            return 0, 255, 0
        if percent <= pct_red:
            return 255, 0, 0

    # Interpolate between green and red
    factor = (percent - pct_green) / (pct_red - pct_green)
    red_level = int(255 * factor)
    green_level = int(255 * (1 - factor))
    return red_level, green_level, 0


def _create_html_tag(value: float, color: tuple[int, int, int]) -> str:
    red_level, green_level, blue_level = color
    hex_color = f"#{red_level:02x}{green_level:02x}{blue_level:02x}"
    return f'<font color="{hex_color}">{value:.2f}%</font>'


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
        return f'<tr><th style="{style}">{key}</th><th style="{style}">{value}</th></tr>'

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
        mean_starting_time = _timedelta_to_human_readable(np.mean(starting_times))  # type: ignore[arg-type]
        std_starting_time = _timedelta_to_human_readable(np.std(starting_times))  # type: ignore[arg-type]
        info.append(("avg job start time", mean_starting_time))
        info.append(("std job start time", std_starting_time))

    with suppress(Exception):
        df = run_manager.parse_log_files()
        t_last = (pd.Timestamp.now() - df.timestamp.max()).seconds

        cpu = df.cpu_usage.mean()
        cpu_html_value = _create_html_tag(cpu, _interp_red_green(cpu, 50, 80))

        mem = df.mem_usage.mean()
        mem_html_value = _create_html_tag(mem, _interp_red_green(mem, 80, 50))

        max_mem = df.mem_usage.max()
        max_mem_html_value = _create_html_tag(
            max_mem,
            _interp_red_green(max_mem, 80, 50),
        )

        overhead = df.overhead.mean()
        overhead_html_value = _create_html_tag(
            overhead,
            _interp_red_green(overhead, 10, 30),
        )

        from_logs = [
            ("# of points", df.npoints.sum()),
            ("mean CPU usage", cpu_html_value),
            ("mean memory usage", mem_html_value),
            ("max memory usage", max_mem_html_value),
            ("mean overhead", overhead_html_value),
            ("last log-entry", f"{t_last}s ago"),
        ]
        for key in ["npoints/s", "latest_loss", "nlearners"]:
            with suppress(Exception):
                from_logs.append((f"mean {key}", f"{df[key].mean():.1f}"))
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
    itables_checkbox_default: bool = False,
    additional_widgets: list[ipyw.Widget] | None = None,
    extra_widget_config: dict | None = None,
) -> tuple[ipyw.VBox, Callable[[Any], None]]:
    import ipywidgets as ipyw
    from IPython.display import display

    def _update_data_df(
        itables_checkbox: ipyw.Checkbox,
        output_widget: ipyw.Output,
    ) -> Callable[[Any], None]:
        def on_click(_: Any) -> None:
            with output_widget:
                output_widget.clear_output()
                df = data_provider()
                if itables_checkbox.value:
                    from itables import show

                    _set_itables_opts()
                    show(df)
                else:
                    with _display_all_dataframe_rows():
                        display(df)

        return on_click

    # Create widgets
    output_widget = ipyw.Output()
    itables_checkbox = ipyw.Checkbox(
        description="Use itables (interactive)",
        indent=False,
        value=itables_checkbox_default,
    )
    update_button = ipyw.Button(
        description=update_button_text,
        button_style="info",
        icon="refresh",
    )

    # Update the DataFrame in the ipyw.Output widget when the button is clicked or the checkbox is changed
    update_function = _update_data_df(
        itables_checkbox,
        output_widget,
    )
    update_button.on_click(update_function)
    itables_checkbox.observe(update_function, names="value")

    # Initialize the DataFrame display
    update_function(None)

    # Create a ipyw.VBox and add the widgets to it
    widget_list = [itables_checkbox, update_button, output_widget]
    if additional_widgets:
        widget_list = additional_widgets + widget_list

    if extra_widget_config:
        for config in extra_widget_config.values():
            widget_list.insert(config["position"], config["widget"])

    vbox = ipyw.VBox(widget_list, layout=ipyw.Layout(border="solid 2px gray"))
    return (vbox, update_function)


def queue_widget(scheduler: BaseScheduler) -> ipyw.VBox:
    """Create a widget that shows the current queue and allows to update it."""
    import ipywidgets as ipyw

    me_only_checkbox = ipyw.Checkbox(
        description="Only my jobs",
        indent=False,
        value=True,
    )

    def get_queue_df() -> pd.DataFrame:
        queue = scheduler.queue(me_only=me_only_checkbox.value)
        return pd.DataFrame(queue).transpose()

    # Get both the VBox and the update_function from _create_widget
    vbox, update_function = _create_widget(
        get_queue_df,
        "Update queue",
        additional_widgets=[me_only_checkbox],
    )

    # Add an observer to the 'me_only_checkbox' that calls the 'update_function' when the checkbox value changes
    me_only_checkbox.observe(update_function, names="value")
    _add_title("adaptive_scheduler.widgets.queue_widget", vbox)
    return vbox


def database_widget(run_manager: RunManager) -> ipyw.VBox:
    """Create a widget that shows the current database and allows to update it."""

    def get_database_df() -> pd.DataFrame:
        return run_manager.database_manager.as_df()

    vbox, _ = _create_widget(get_database_df, "Update database")
    _add_title("adaptive_scheduler.widgets.database_widget", vbox)
    return vbox


def _set_itables_opts() -> None:
    import itables.options as opt

    opt.maxBytes = 262_144


def results_widget(
    fnames: list[str] | list[Path],
    dataframe_format: _DATAFRAME_FORMATS,
) -> ipyw.VBox:
    """Widget that loads and displays the results as `pandas.DataFrame`s."""
    import ipywidgets as ipyw

    def on_concat_checkbox_value_change(change: dict) -> None:
        if change["name"] == "value":
            dropdown.layout.visibility = "hidden" if change["new"] else "visible"
            update_function(None)

    def get_results_df() -> pd.DataFrame:
        selected_fname = dropdown.value
        dfs = [selected_fname] if not concat_checkbox.value else fnames
        df = load_dataframes(dfs, format=dataframe_format)
        assert isinstance(df, pd.DataFrame)

        if len(df) > max_rows.value:
            sample_indices: np.ndarray = np.linspace(
                0,
                len(df) - 1,
                num=max_rows.value,
                dtype=int,
            )
            df = df.iloc[sample_indices]

        return df  # type: ignore[return-value]

    # Create widgets
    dropdown = ipyw.Dropdown(options=fnames)
    concat_checkbox = ipyw.Checkbox(description="Concat all dataframes", indent=False)
    max_rows = ipyw.IntText(value=300, description="Max rows")

    # Observe the value change in the 'concat_checkbox'
    concat_checkbox.observe(on_concat_checkbox_value_change, names="value")

    extra_widget_config = {
        "concat_checkbox": {"widget": concat_checkbox, "position": 1},
        "dropdown": {"widget": dropdown, "position": 0},
        "max_rows": {"widget": max_rows, "position": 4},
    }

    vbox, update_function = _create_widget(
        get_results_df,
        "Update results",
        extra_widget_config=extra_widget_config,
    )

    # Add observers for the 'dropdown' and 'max_rows' widgets
    dropdown.observe(update_function, names="value")
    max_rows.observe(update_function, names="value")

    _add_title("adaptive_scheduler.widgets.results_widget", vbox)
    return vbox


def _toggle_widget(
    widget_key: str,
    widget_dict: dict[str, ipyw.Widget | str],
    toggle_dict: dict[str, dict[str, Any]],
) -> Callable[[Any], None]:
    import ipywidgets as ipyw
    from IPython.display import display

    def on_click(_: Any) -> None:
        widget = toggle_dict[widget_key]["widget"]
        if widget is None:
            widget = toggle_dict[widget_key]["init_func"]()
            toggle_dict[widget_key]["widget"] = widget

        button = widget_dict[widget_key]
        assert isinstance(button, ipyw.Button)
        show_description = f"show {widget_key}"
        hide_description = f"hide {widget_key}"
        output = toggle_dict[widget_key]["output"]
        if button.description == show_description:
            button.description = hide_description
            button.button_style = "warning"
            with output:
                output.clear_output()
                display(widget)
        else:
            button.description = show_description
            button.button_style = "info"
            with output:
                output.clear_output()

    return on_click


def _switch_to(
    box: ipyw.HBox,
    *buttons: ipyw.Button,
    _callable: Callable[[], None] | None = None,
) -> Callable[[Any], None]:
    def on_click(_: Any) -> None:
        box.children = tuple(buttons)
        if _callable is not None:
            _callable()

    return on_click


def _create_confirm_deny(
    initial_button: ipyw.Button,
    widgets: dict[str, ipyw.Button | ipyw.HBox],
    callable_func: Callable[[], None],
    key: str,
) -> None:
    import ipywidgets as ipyw

    confirm_button = ipyw.Button(
        description="Confirm",
        button_style="success",
        icon="check",
    )
    deny_button = ipyw.Button(
        description="Deny",
        button_style="danger",
        icon="close",
    )

    initial_button.on_click(
        _switch_to(widgets[key], confirm_button, deny_button),
    )
    deny_button.on_click(_switch_to(widgets[key], initial_button))
    confirm_button.on_click(
        _switch_to(widgets[key], initial_button, _callable=callable_func),
    )


def info(run_manager: RunManager) -> None:
    """Display information about the `RunManager`.

    Returns an interactive ipywidget that can be
    visualized in a Jupyter notebook.
    """
    import ipywidgets as ipyw
    from IPython.display import display

    _disable_widgets_output_scrollbar()

    status = ipyw.HTML(value=_info_html(run_manager))

    layout = ipyw.Layout(width="200px")

    cancel_button = ipyw.Button(
        description="cancel jobs",
        layout=layout,
        button_style="danger",
        icon="stop",
    )
    cleanup_button = ipyw.Button(
        description="cleanup log and batch files",
        layout=layout,
        button_style="danger",
        icon="remove",
    )
    update_info_button = ipyw.Button(
        description="update info",
        layout=layout,
        icon="refresh",
    )
    update_info_button.style.button_color = "lightgreen"
    load_learners_button = ipyw.Button(
        description="load learners",
        layout=layout,
        button_style="info",
        icon="download",
    )
    show_logs_button = ipyw.Button(
        description="show logs",
        layout=layout,
        button_style="info",
        icon="book",
    )
    show_queue_button = ipyw.Button(
        description="show queue",
        layout=layout,
        button_style="info",
        icon="tasks",
    )
    show_db_button = ipyw.Button(
        description="show database",
        layout=layout,
        button_style="info",
        icon="database",
    )
    show_results_button = ipyw.Button(
        description="show results",
        layout=layout,
        button_style="info",
        icon="table",
    )
    widgets = {
        "update info": update_info_button,
        "cancel": ipyw.HBox([cancel_button], layout=layout),
        "cleanup": ipyw.HBox([cleanup_button], layout=layout),
        "load learners": load_learners_button,
        "logs": show_logs_button,
        "queue": show_queue_button,
        "database": show_db_button,
        "results": show_results_button,
    }

    def update(_: Any) -> None:
        status.value = _info_html(run_manager)

    def load_learners(_: Any) -> None:
        run_manager.load_learners()

    toggle_dict = {
        "logs": {
            "widget": None,
            "init_func": lambda: log_explorer(run_manager),
            "show_description": "logs",
            "hide_description": "hide logs",
            "output": ipyw.Output(),
        },
        "queue": {
            "widget": None,
            "init_func": lambda: queue_widget(run_manager.scheduler),
            "output": ipyw.Output(),
        },
        "database": {
            "widget": None,
            "init_func": lambda: database_widget(run_manager),
            "output": ipyw.Output(),
        },
        "results": {
            "widget": None,
            "init_func": lambda: results_widget(
                run_manager.fnames,
                run_manager.dataframe_format,
            ),
            "output": ipyw.Output(),
        },
    }

    def cancel() -> None:
        run_manager.cancel()
        update(None)

    def cleanup(*, include_old_logs: ipyw.Checkbox) -> Callable[[], None]:
        def _callable() -> None:
            run_manager.cleanup(remove_old_logs_folder=include_old_logs.value)
            update(None)

        return _callable

    widgets["update info"].on_click(update)
    toggle_logs = _toggle_widget("logs", widgets, toggle_dict)
    toggle_queue = _toggle_widget("queue", widgets, toggle_dict)
    toggle_database = _toggle_widget("database", widgets, toggle_dict)
    toggle_results = _toggle_widget("results", widgets, toggle_dict)
    widgets["logs"].on_click(toggle_logs)
    widgets["queue"].on_click(toggle_queue)
    widgets["database"].on_click(toggle_database)
    widgets["results"].on_click(toggle_results)
    widgets["load learners"].on_click(load_learners)

    # Cancel button with confirm/deny option
    _create_confirm_deny(cancel_button, widgets, cancel, key="cancel")

    # Cleanup button with confirm/deny option
    include_old_logs = ipyw.Checkbox(
        value=False,
        description=f"Remove {run_manager.move_old_logs_to}/ folder",
        indent=False,
    )
    # Cleanup button with confirm/deny option
    cleanup_callable = cleanup(include_old_logs=include_old_logs)
    _create_confirm_deny(cleanup_button, widgets, cleanup_callable, key="cleanup")
    buttons_box = ipyw.VBox(tuple(widgets.values()))
    buttons_box.layout.margin = "0 0 0 100px"
    top_box = ipyw.HBox((status, buttons_box))
    box = ipyw.VBox((top_box, *(v["output"] for v in toggle_dict.values())))
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


def _disable_widgets_output_scrollbar() -> None:
    import ipywidgets as ipyw
    from IPython.display import display

    style = """
        <style>
            .jupyter-widgets-output-area .output_scroll {
                height: unset !important;
                border-radius: unset !important;
                -webkit-box-shadow: unset !important;
                box-shadow: unset !important;
            }
            .jupyter-widgets-output-area  {
                height: auto !important;
            }
        </style>
        """
    display(ipyw.HTML(style))
