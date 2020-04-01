import asyncio
from pathlib import Path

from ipywidgets import HTML, Button, Checkbox, Dropdown, Layout, Textarea, VBox


def _get_fnames(run_manager, only_running):
    if only_running:
        fnames = []
        for entry in run_manager.database_manager.as_dicts():
            if entry["log_fname"] is not None:
                fnames.append(entry["log_fname"])
            fnames += entry["output_logs"]
        return sorted(map(Path, fnames))
    return sorted(Path(".").glob(f"{run_manager.job_name}-*"))


def _read_file(fname):
    with fname.open() as f:
        return "".join(f.readlines())


def log_explorer(run_manager):
    def _update_dropdown(run_manager, dropdown, checkbox):
        def on_click(_):
            fnames = _get_fnames(run_manager, checkbox.value)
            dropdown.options = fnames
            dropdown.disabled = not fnames

        return on_click

    async def _tail_log(fname, textarea):
        while True:
            await asyncio.sleep(2)
            try:
                textarea.value = _read_file(fname)
            except asyncio.CancelledError:
                return
            except Exception:
                pass

    def _tail(dropdown, button, textarea):
        tail_task = None
        ioloop = asyncio.get_running_loop()

        def on_click(_):
            nonlocal tail_task
            if tail_task is None:
                fname = dropdown.options[dropdown.index]
                dropdown.disabled = True
                tail_task = ioloop.create_task(_tail_log(fname, textarea))
                button.description = "cancel tail log"
                button.button_style = "danger"
                button.icon = "window-close"
            else:
                button.description = "tail log"
                button.button_style = "info"
                dropdown.disabled = False
                button.icon = "refresh"
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
    textarea = Textarea(text, layout=Layout(width="auto"))
    dropdown = Dropdown(options=fnames)
    dropdown.observe(_on_dropdown_change(textarea))
    checkbox = Checkbox(description="Only files of running jobs", indent=False)
    update_button = Button(
        description="update file list", button_style="info", icon="refresh",
    )
    update_button.on_click(_update_dropdown(run_manager, dropdown, checkbox))
    checkbox.observe(_click_button_on_change(update_button))
    tail_button = Button(description="tail log", button_style="info", icon="refresh")
    tail_button.on_click(_tail(dropdown, tail_button, textarea))
    title = HTML("<h2><tt>adaptive_scheduler.widgets.log_explorer</tt></h2>")
    return VBox(
        [title, checkbox, update_button, dropdown, tail_button, textarea],
        layout=Layout(border="solid 2px gray"),
    )
