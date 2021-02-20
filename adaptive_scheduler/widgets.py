import asyncio
from collections import defaultdict
from contextlib import suppress
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from ipywidgets import (
    HTML,
    Button,
    Checkbox,
    Dropdown,
    HBox,
    Layout,
    Text,
    Textarea,
    VBox,
)


def _get_fnames(run_manager, only_running: bool) -> List[Path]:
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


def _failed_job_logs(fnames, run_manager, only_running):

    running = {
        Path(e["log_fname"]).stem
        for e in run_manager.database_manager.as_dicts()
        if e["log_fname"] is not None
    }

    fnames = {
        fname.stem for fname in fnames if fname.suffix != run_manager.scheduler.ext
    }
    failed = fnames - running
    failed = [Path(f) for stem in failed for f in glob(f"{stem}*")]

    def maybe_append(fname: str, other_dir: Path, lst: List[Path]):
        p = Path(fname)
        p_other = other_dir / p.name
        if p.exists():
            lst.append(p)
        elif p_other.exists():
            lst.append(p_other)

    if not only_running:
        base = Path(run_manager.move_old_logs_to)
        for e in run_manager.database_manager.failed:
            if not e["is_done"]:
                for f in e["output_logs"]:
                    maybe_append(f, base, failed)
            maybe_append(e["log_fname"], base, failed)
    return failed


def _files_that_contain(fnames, text):
    def contains(fname, text):
        with fname.open() as f:
            for line in f:
                if text in line:
                    return True
            return False

    return [fname for fname in fnames if contains(fname, text)]


def _sort_fnames(sort_by, run_manager, fnames):
    def _try_transform(f):
        def _f(x):
            try:
                return f(x)
            except Exception:
                return x

        return _f

    def _sort_key(value):
        x, fname = value
        if isinstance(x, str):
            return -1, fname
        return float(x), fname

    mapping = {
        "Alphabetical": (None, lambda x: ""),
        "CPU %": ("cpu_usage", lambda x: f"{x:.1f}%"),
        "Mem %": ("mem_usage", lambda x: f"{x:.1f}%"),
        "Last editted": (
            "timestamp",
            lambda x: f"{(np.datetime64(datetime.now()) - x) / 1e9}s ago",
        ),
        "Loss": ("latest_loss", lambda x: f"{x:.2f}"),
        "npoints": ("npoints", lambda x: f"{x} pnts"),
        "Elapsed time": ("elapsed_time", lambda x: f"{x / 1e9}s"),
    }

    def extract(df, fname, key):
        df_sel = df[df.log_fname.str.contains(fname.name)]
        values = df_sel[key].values
        if values:
            return values[0]
        else:
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
        stems = [fname.stem for fname in log_fnames]
        vals = [extract(df, fname, df_key) for fname in log_fnames]
        val_stem = sorted(zip(vals, stems), key=_sort_key, reverse=True)

        result = []
        for val, stem in val_stem:
            val = _try_transform(transform)(val)
            for fname in fname_mapping[stem]:
                result.append((f"{val}: {fname.name}", fname))

        missing = fname_mapping.keys() - set(stems)
        for stem in sorted(missing):
            for fname in fname_mapping[stem]:
                result.append((f"?: {fname.name}", fname))
        return result

    return fnames


def _read_file(fname: Path) -> str:
    try:
        with fname.open() as f:
            return "".join(f.readlines())
    except UnicodeDecodeError:
        return f"Could not decode file ({fname})!"
    except Exception as e:
        return f"Exception with trying to read {fname}:\n{e}."


def log_explorer(run_manager) -> VBox:  # noqa: C901
    def _update_fname_dropdown(
        run_manager,
        fname_dropdown,
        only_running_checkbox,
        only_failed_checkbox,
        sort_by_dropdown,
        contains_text,
    ):
        def on_click(_):
            current_value = fname_dropdown.value
            fnames = _get_fnames(run_manager, only_running_checkbox.value)
            if only_failed_checkbox.value:
                fnames = _failed_job_logs(
                    fnames, run_manager, only_running_checkbox.value
                )
            if contains_text.value.strip() != "":
                fnames = _files_that_contain(fnames, contains_text.value.strip())
            fnames = _sort_fnames(sort_by_dropdown.value, run_manager, fnames)
            fname_dropdown.options = fnames
            with suppress(Exception):
                fname_dropdown.value = current_value
            fname_dropdown.disabled = not fnames

        return on_click

    def _last_editted(fname: Path) -> float:
        try:
            return fname.stat().st_mtime
        except FileNotFoundError:
            return -1.0

    async def _tail_log(fname: Path, textarea: Textarea) -> None:
        T = -2.0  # to make sure the update always triggers
        while True:
            await asyncio.sleep(2)
            try:
                T_new = _last_editted(fname)
                if T_new > T:
                    textarea.value = _read_file(fname)
                    T = T_new
            except asyncio.CancelledError:
                return
            except Exception:
                pass

    def _tail(
        dropdown,
        tail_button,
        textarea,
        update_button,
        only_running_checkbox,
        only_failed_checkbox,
    ):
        tail_task = None
        ioloop = asyncio.get_running_loop()

        def on_click(_):
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

    def _on_dropdown_change(textarea):
        def on_change(change):
            if (
                change["type"] == "change"
                and change["name"] == "value"
                and change["new"] is not None
            ):
                textarea.value = _read_file(change["new"])

        return on_change

    def _click_button_on_change(button):
        def on_change(change):
            if change["type"] == "change" and change["name"] == "value":
                button.click()

        return on_change

    fnames = _get_fnames(run_manager, only_running=False)
    # no need to sort `fnames` because the default sort_by option is alphabetical
    text = _read_file(fnames[0]) if fnames else ""
    textarea = Textarea(text, layout=dict(width="auto"), rows=20)
    sort_by_dropdown = Dropdown(
        description="Sort by",
        options=["Alphabetical", "CPU %", "Mem %", "Last editted", "Loss", "npoints"],
    )
    contains_text = Text(description="Has string")
    fname_dropdown = Dropdown(description="File name", options=fnames)
    fname_dropdown.observe(_on_dropdown_change(textarea))
    only_running_checkbox = Checkbox(
        description="Only files of running jobs", indent=False
    )
    only_failed_checkbox = Checkbox(
        description="Only files of failed jobs (might include false positives)",
        indent=False,
    )
    update_button = Button(
        description="update file list", button_style="info", icon="refresh"
    )
    update_button.on_click(
        _update_fname_dropdown(
            run_manager,
            fname_dropdown,
            only_running_checkbox,
            only_failed_checkbox,
            sort_by_dropdown,
            contains_text,
        )
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
        )
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


def _info_html(run_manager) -> str:
    queue = run_manager.scheduler.queue(me_only=True)
    run_manager.database_manager.update(queue)
    jobs = [job for job in queue.values() if job["job_name"] in run_manager.job_names]
    n_running = sum(job["state"] in ("RUNNING", "R") for job in jobs)
    n_pending = sum(job["state"] in ("PENDING", "Q", "CONFIGURING") for job in jobs)
    n_done = sum(job["is_done"] for job in run_manager.database_manager.as_dicts())
    n_failed = len(run_manager.database_manager.failed)
    n_failed_color = "red" if n_failed > 0 else "black"

    status = run_manager.status()
    color = {
        "cancelled": "orange",
        "not yet started": "orange",
        "running": "blue",
        "failed": "red",
        "finished": "green",
    }[status]

    def _table_row(i, key, value):
        """Style the rows of a table. Based on the default Jupyterlab table style."""
        style = "text-align: right; padding: 0.5em 0.5em; line-height: 1.0;"
        if i % 2 == 1:
            style += " background: var(--md-grey-100);"
        return (
            f'<tr><th style="{style}">{key}</th><th style="{style}">{value}</th></tr>'
        )

    info = [
        ("status", f'<font color="{color}">{status}</font>'),
        ("# running jobs", f'<font color="blue">{n_running}</font>'),
        ("# pending jobs", f'<font color="orange">{n_pending}</font>'),
        ("# finished jobs", f'<font color="green">{n_done}</font>'),
        ("# failed jobs", f'<font color="{n_failed_color}">{n_failed}</font>'),
        ("elapsed time", timedelta(seconds=run_manager.elapsed_time())),
    ]

    with suppress(Exception):
        df = run_manager.parse_log_files()
        t_last = (pd.Timestamp.now() - df.timestamp.max()).seconds

        overhead = df.mem_usage.mean()
        red_level = max(0, min(int(255 * overhead / 100), 255))
        overhead_color = "#{:02x}{:02x}{:02x}".format(red_level, 255 - red_level, 0)
        overhead_html_value = f'<font color="{overhead_color}">{overhead:.2f}%</font>'

        cpu = df.cpu_usage.mean()
        red_level = max(0, min(int(255 * cpu / 100), 255))
        cpu_color = "#{:02x}{:02x}{:02x}".format(red_level, red_level, 0)
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


def info(run_manager) -> None:
    """Display information about the `RunManager`.

    Returns an interactive ipywidget that can be
    visualized in a Jupyter notebook.
    """
    from IPython.display import display
    from ipywidgets import HTML, Button, Layout, VBox

    status = HTML(value=_info_html(run_manager))

    layout = Layout(width="200px")

    cancel_button = Button(
        description="cancel jobs", layout=layout, button_style="danger", icon="stop"
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
        description="show logs", layout=layout, button_style="info", icon="book"
    )
    widgets = {
        "update info": update_info_button,
        "cancel": HBox([cancel_button], layout=layout),
        "cleanup": HBox([cleanup_button], layout=layout),
        "load learners": load_learners_button,
        "show logs": show_logs_button,
    }

    def switch_to(box, *buttons, _callable=None):
        def on_click(_):
            box.children = tuple(buttons)
            if _callable is not None:
                _callable()

        return on_click

    box = VBox([])

    log_widget = None

    def update(_):
        status.value = _info_html(run_manager)

    def load_learners(_):
        run_manager.load_learners()

    def toggle_logs(_):
        nonlocal log_widget

        if log_widget is None:
            log_widget = log_explorer(run_manager)

        b = widgets["show logs"]
        if b.description == "show logs":
            b.description = "hide logs"
            box.children = (*box.children, log_widget)
        else:
            b.description = "show logs"
            box.children = box.children[:-1]

    def cancel():
        run_manager.cancel()
        update(None)

    def cleanup(include_old_logs):
        def _callable():
            run_manager.cleanup(include_old_logs.value)
            update(None)

        return _callable

    widgets["update info"].on_click(update)
    widgets["show logs"].on_click(toggle_logs)
    widgets["load learners"].on_click(load_learners)

    # Cancel button with confirm/deny option
    confirm_cancel_button = Button(
        description="Confirm", button_style="success", icon="check"
    )
    deny_cancel_button = Button(description="Deny", button_style="danger", icon="close")

    cancel_button.on_click(
        switch_to(widgets["cancel"], confirm_cancel_button, deny_cancel_button)
    )
    deny_cancel_button.on_click(switch_to(widgets["cancel"], cancel_button))
    confirm_cancel_button.on_click(
        switch_to(widgets["cancel"], cancel_button, _callable=cancel)
    )

    # Cleanup button with confirm/deny option
    include_old_logs = Checkbox(
        False,
        description=f"Remove {run_manager.move_old_logs_to}/ folder",
        indent=False,
    )
    confirm_cleanup_button = Button(
        description="Confirm", button_style="success", icon="check"
    )
    deny_cleanup_button = Button(
        description="Deny", button_style="danger", icon="close"
    )

    cleanup_box = VBox(
        [HBox([confirm_cleanup_button, deny_cleanup_button]), include_old_logs]
    )
    cleanup_button.on_click(switch_to(widgets["cleanup"], cleanup_box))
    deny_cleanup_button.on_click(switch_to(widgets["cleanup"], cleanup_button))
    confirm_cleanup_button.on_click(
        switch_to(
            widgets["cleanup"], cleanup_button, _callable=cleanup(include_old_logs)
        )
    )

    box.children = (status, *tuple(widgets.values()))
    display(box)
