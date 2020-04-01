import asyncio
from pathlib import Path
from typing import List

from ipywidgets import HTML, Button, Checkbox, Dropdown, Layout, Textarea, VBox


def _get_fnames(run_manager, only_running: bool) -> List[Path]:
    if only_running:
        fnames = []
        for entry in run_manager.database_manager.as_dicts():
            if entry["log_fname"] is not None:
                fnames.append(entry["log_fname"])
            fnames += entry["output_logs"]
        return sorted(map(Path, fnames))
    return sorted(Path(".").glob(f"{run_manager.job_name}-*"))


def _read_file(fname: Path) -> str:
    with fname.open() as f:
        return "".join(f.readlines())


def log_explorer(run_manager) -> VBox:  # noqa: C901
    def _update_dropdown(run_manager, dropdown, checkbox):
        def on_click(_):
            fnames = _get_fnames(run_manager, checkbox.value)
            dropdown.options = fnames
            dropdown.disabled = not fnames

        return on_click

    def _last_editted(fname: Path) -> float:
        try:
            return fname.stat().st_mtime
        except FileNotFoundError:
            return -1.0

    async def _tail_log(fname: Path, textarea: Textarea) -> None:
        T = _last_editted(fname)
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
    dropdown = Dropdown(options=fnames)
    dropdown.observe(_on_dropdown_change(textarea))
    checkbox = Checkbox(description="Only files of running jobs", indent=False)
    update_button = Button(
        description="update file list", button_style="info", icon="refresh",
    )
    update_button.on_click(_update_dropdown(run_manager, dropdown, checkbox))
    checkbox.observe(_click_button_on_change(update_button))
    tail_button = Button(description="tail log", button_style="info", icon="refresh")
    tail_button.on_click(
        _tail(dropdown, tail_button, textarea, update_button, checkbox)
    )
    title = HTML("<h2><tt>adaptive_scheduler.widgets.log_explorer</tt></h2>")
    return VBox(
        [title, checkbox, update_button, dropdown, tail_button, textarea],
        layout=Layout(border="solid 2px gray", min_height="500px"),
    )
