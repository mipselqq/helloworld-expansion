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
    monkeypatch.setattr(gui.filedialog, 'asksaveasfilename', lambda **kw: str(tmp_path/'out.png'))
    app = gui.StudioWindow(root, Path('config.toml'), fullscreen=False)
    fake = FakeProject()
    app.project = fake
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
        app.wheel(SimpleNamespace(x=10, y=10, num=4, delta=0))
        assert app.zoom > 1
        wait_for(lambda: bool(fake.previews) and app.displayed_revision == app.revision)
        assert not fake.previews[-1][0].grade
        assert not fake.save_done.is_set()
    finally:
        fake.save_done.set()
        app.close()


def test_save_dialog_file_list_has_readable_colors(monkeypatch, tmp_path):
    import tkinter as tk
    from tkinter import filedialog
    from artemis import gui

    root = tk.Tk()
    root.withdraw()
    monkeypatch.setattr(gui, 'screens', lambda: [])
    app = gui.StudioWindow(root, Path('config.toml'), fullscreen=False)
    # Disable the scheduled preview for this short visual-style test.
    app.displayed_revision = app.revision
    (tmp_path/'wallpaper.png').write_bytes(b'')
    observed = {}
    def inspect():
        def walk(widget):
            yield widget
            for child in root.tk.splitlist(root.tk.call('winfo','children',widget)):
                yield from walk(child)
        widgets = list(walk('.'))
        for widget in widgets:
            if str(root.tk.call('winfo','class',widget)) != 'Canvas' or '.icons.' not in widget:
                continue
            observed['background'] = root.tk.call(widget,'cget','-background')
            for item in root.tk.splitlist(root.tk.call(widget,'find','all')):
                if root.tk.call(widget,'type',item) == 'text':
                    observed['foreground'] = root.tk.call(widget,'itemcget',item,'-fill')
                    break
        for widget in widgets:
            if widget.endswith('.contents.f2.cancel'):
                root.tk.call(widget,'invoke')
                break
    root.after(500, inspect)
    try:
        filedialog.asksaveasfilename(parent=root,initialdir=tmp_path,initialfile='out.png')
        assert observed == {'background': '#202d3d', 'foreground': '#e5edf7'}
    finally:
        app.close()
