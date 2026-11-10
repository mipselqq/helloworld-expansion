import numpy as np
from artemis.geometry import Limb
from artemis import sun_glow


def test_glow_reduction_preserves_disk_and_limb():
    h, w = 800, 1000
    y, x = np.indices((h, w), dtype=np.float32)
    limb = Limb(np.array([450., 380.]), np.array([180., 180.]), 0., np.empty((0, 2)), 0.)
    r = limb.radius(x, y)
    rng = np.random.default_rng(12)
    image = np.full((h, w, 3), .045, dtype=np.float32)
    glow = np.exp(-((x-760)/270)**2-((y-610)/260)**2)*.075
    image += glow[...,None]
    image += rng.normal(0, .002, image.shape).astype(np.float32)
    image[r < 1.02] = .55
    before = image.copy()
    stats = sun_glow.apply(image, limb, [800., 630.], np.array([[800., 630.]]), {
        'enabled': True, 'analysis_scale': 8, 'strength': .9, 'noise_strength': .55,
        'rim_start': 1.025, 'rim_end': 1.24,
    })
    np.testing.assert_array_equal(image[r <= 1.025], before[r <= 1.025])
    region = (r > 1.4) & (x > 750) & (y > 540)
    assert image[region].mean() < before[region].mean() - .005
    assert stats['pixels_over_half_weight'] > 0


def test_glow_can_be_disabled():
    image = np.random.default_rng(4).random((8, 8, 3),dtype=np.float32)
    before = image.copy()
    result = sun_glow.apply(image, None, None, None, {'enabled': False})
    np.testing.assert_array_equal(image, before)
    assert result == {'enabled': False}
