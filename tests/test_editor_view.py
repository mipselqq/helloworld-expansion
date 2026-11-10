import numpy as np
from artemis.studio import Project, Settings, view_layout


def test_full_frame_and_zoom_map_to_master():
    master = (2100, 900)
    full = view_layout(master, master, (0, 0), (700, 300), (.5, .5), 1)
    zoom = view_layout(master, master, (0, 0), (700, 300), (.5, .5), 3)
    assert full[0] == (0, 0, 2100, 900)
    assert zoom[0] == (700, 300, 1400, 600)
    assert zoom[1] == (0, 0, 700, 300)
    assert zoom[2] == 1


def test_layout_respects_output_crop_and_protected_disk():
    master = (2100, 900)
    layout = view_layout(master, (900, 900), (1200, 800), (600, 600), (.5, .5), 1)
    box, screen, _ = layout
    assert box[2] - box[0] >= 1200
    assert box[3] - box[1] >= 800
    assert screen[0] >= 0 and screen[1] >= 0
    assert screen[2] <= 600 and screen[3] <= 600


def test_preview_renders_only_visible_crop():
    height, width = 900, 2100
    y, x = np.indices((height, width))
    source = np.empty((height, width, 3), dtype=np.uint16)
    source[:, :, 0] = (x % 256) * 257
    source[:, :, 1] = (y % 256) * 257
    source[:, :, 2] = 40 * 257
    project = Project.__new__(Project)
    project.size = (width, height)
    project.keep = (0, 0)
    calls = []
    def render(options, *, viewport, sample, cancel):
        calls.append((viewport, sample))
        xx, yy, w, h = viewport
        return source[yy:yy+h:sample, xx:xx+w:sample].copy()
    project.render = render
    preview = project.preview_view(Settings(), (width, height), (700, 300), (.5, .5), 3)
    assert calls == [((668, 268, 764, 364), 1)]
    expected = (source[300:600, 700:1400] // 257).astype(np.uint8)
    np.testing.assert_array_equal(np.asarray(preview), expected)
