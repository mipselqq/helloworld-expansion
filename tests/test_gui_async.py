import os
import time
import threading
from concurrent.futures import CancelledError
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(not os.environ.get('DISPLAY'), reason='Tk needs an X display')


def test_preview_is_replaceable_during_export(monkeypatch, tmp_path):
    import tkinter as tk
    from artemis import gui
    class FakeProject:
        def __init__(self):
            self.save_started = threading.Event()
            self.save_done = threading.Event()
            self.previews = []
        def preview_view(self, options, size, dimensions, center, zoom, cancel):
            for _ in range(35):
                if cancel.is_set():
                    raise CancelledError()
                time.sleep(.01)
            self.previews.append((options, center, zoom))
            return Image.new('RGB', dimensions)
        def save(self, options, size, path):
            self.save_started.set()
            assert self.save_done.wait(8)
    root = tk.Tk()
    root.withdraw()
    monkeypatch.setattr(gui, 'screens', lambda: [])
    app = gui.StudioWindow(root, Path('config.toml'), fullscreen=False)
    picker = SimpleNamespace(poll=lambda: 0, returncode=0,
                             communicate=lambda: (str(tmp_path/'out.png')+'\n',''))
    monkeypatch.setattr(gui.subprocess, 'Popen', lambda *args, **kwargs: picker)
    fake = FakeProject()
    app.project = fake
    assert app.settings().correct_stars is True
    assert app.settings().amoled is False
    def wait_for(check, timeout=5):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            root.update()
            if check():
                return
            time.sleep(.01)
        pytest.fail('UI job did not finish in time')
    try:
        wait_for(lambda: app.future is not None)
        app.save()
        wait_for(fake.save_started.is_set)
        app.variables['grade'].set(False)
        app.variables['amoled'].set(True)
        app.wheel(SimpleNamespace(x=10, y=10, num=4, delta=0))
        assert app.zoom > 1
        wait_for(lambda: bool(fake.previews) and app.displayed_revision == app.revision)
        assert not fake.previews[-1][0].grade
        assert fake.previews[-1][0].amoled
        assert not fake.save_done.is_set()
    finally:
        fake.save_done.set()
        app.close()



@pytest.mark.parametrize("returncode, filename, error", [
    (1, "", False),
    (0, "/tmp/wallpaper.jpeg", True),
    (2, "", True),
])
def test_native_picker_cancel_and_errors(monkeypatch, returncode, filename, error):
    import tkinter as tk
    from artemis import gui
    root = tk.Tk()
    root.withdraw()
    monkeypatch.setattr(gui, "screens", lambda: [])
    app = gui.StudioWindow(root, Path("config.toml"), fullscreen=False)
    errors = []
    commands = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *args, **kwargs: errors.append(args))
    picker = SimpleNamespace(poll=lambda: returncode, returncode=returncode,
                             communicate=lambda: (filename, "test failure"))
    def launch(command, **kwargs):
        commands.append(command)
        return picker
    monkeypatch.setattr(gui.subprocess, "Popen", launch)
    try:
        app.save()
        app.finish_save_picker()
        assert commands[0][0] == "kdialog"
        assert "--getsavefilename" in commands[0]
        assert "*.tif *.tiff|TIFF (16-bit)" in commands[0][-1]
        assert bool(errors) == error
        assert app.save_future is None
        assert app.save_picker is None
        assert "disabled" not in app.save_button.state()
    finally:
        app.close()
