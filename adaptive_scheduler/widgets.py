import asyncio
from collections import defaultdict
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from ipywidgets import HTML, Button, Checkbox, Dropdown, Layout, Text, Textarea, VBox


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
    return sorted(Path(".").glob(f"{run_manager.job_name}-*"))


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
        values = df[df.log_fname == fname.name][key].values
        if values:
            return values[0]
        else:
            return "?"

    if sort_by != "Alphabetical":
        fname_mapping = defaultdict(list)
        for fname in fnames:
            fname_mapping[fname.stem].append(fname)

        df = run_manager.parse_log_files()
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
    else:
        result = [(fname.name, fname) for fname in fnames]
    return result


def _read_file(fname: Path) -> str:
    try:
        with fname.open() as f:
            return "".join(f.readlines())
    except UnicodeDecodeError:
        return f"Could not decode file ({fname})!"
    except Exception as e:
        return f"Exception with trying to read {fname}:\n{e}."


def log_explorer(run_manager) -> VBox:  # noqa: C901
    def _update_fname_dropdown(run_manager, fname_dropdown, checkbox, sortby_dropdown):
        def on_click(_):
            current_value = fname_dropdown.value
            fnames = _get_fnames(run_manager, checkbox.value)
            fnames = _sort_fnames(sortby_dropdown.value, run_manager, fnames)
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

    def _tail(dropdown, tail_button, textarea, update_button, checkbox):
        tail_task = None
        ioloop = asyncio.get_running_loop()

        def on_click(_):
            nonlocal tail_task
            if tail_task is None:
                tail_button.description = "cancel tail log"
                tail_button.button_style = "danger"
                tail_button.icon = "window-close"
                dropdown.disabled = True
                update_button.disabled = True
                checkbox.disabled = True
                fname = dropdown.options[dropdown.index]
                tail_task = ioloop.create_task(_tail_log(fname, textarea))
            else:
                tail_button.description = "tail log"
                tail_button.button_style = "info"
                tail_button.icon = "refresh"
                dropdown.disabled = False
                checkbox.disabled = False
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
    text = _read_file(fnames[0]) if fnames else ""
    textarea = Textarea(text, layout=dict(width="auto"), rows=20)
    sortby_dropdown = Dropdown(
        description="Sort by",
        options=["Alphabetical", "CPU %", "Mem %", "Last editted", "Loss", "npoints"],
    )
    contains_text = Text(description="Has string")
    fname_dropdown = Dropdown(description="File name", options=fnames)
    fname_dropdown.observe(_on_dropdown_change(textarea))
    only_running_checkbox = Checkbox(
        description="Only files of running jobs", indent=False
    )
    update_button = Button(
        description="update file list", button_style="info", icon="refresh",
    )
    update_button.on_click(
        _update_fname_dropdown(
            run_manager, fname_dropdown, only_running_checkbox, sortby_dropdown
        )
    )
    only_running_checkbox.observe(_click_button_on_change(update_button))
    tail_button = Button(description="tail log", button_style="info", icon="refresh")
    tail_button.on_click(
        _tail(
            fname_dropdown, tail_button, textarea, update_button, only_running_checkbox
        )
    )
    title = HTML("<h2><tt>adaptive_scheduler.widgets.log_explorer</tt></h2>")
    return VBox(
        [
            title,
            only_running_checkbox,
            update_button,
            sortby_dropdown,
            fname_dropdown,
            contains_text,
            tail_button,
            textarea,
        ],
        layout=Layout(border="solid 2px gray"),
    )
