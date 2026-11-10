import numpy as np

from artemis import amoled, compose
from artemis.color import srgb_decode, srgb_encode
from artemis.geometry import Limb


def setup():
    rng = np.random.default_rng(492)
    image = rng.integers(1000, 60000, (250, 400, 3), dtype=np.uint16)
    limb = Limb(np.array([190., 120.]), np.array([50., 50.]), 0., np.empty((0, 2)), 0.)
    affine = np.array([[1., 0., 0.], [0., 1., 0.]])
    points = np.array([[30.2, 55.7], [300., 180.]])
    venus = np.array([370., 220.])
    for x, y in [(30, 56), (300, 180), (370, 220)]:
        image[max(0,y-55):y+56, max(0,x-55):x+56] = (3000, 4000, 6000)
        image[y, x] = (45000, 51000, 57000)
    return image, limb, affine, points, venus


def test_black_sky_preserves_planet_rim_and_point_cores():
    image, limb, affine, points, venus = setup()
    original = image.copy()
    result = amoled.apply(image, limb, affine, points, venus)
    yy, xx = np.indices(image.shape[:2])
    protected = limb.radius(xx, yy) <= 1.025
    np.testing.assert_array_equal(image[protected], original[protected])
    for y, x in [(56, 30), (180, 300), (220, 370)]:
        assert image[y, x].min() > 35000
        ratio = image[y, x] / original[y, x]
        assert np.ptp(ratio) < .0001
    assert np.all(image[10, 10] == 0)
    assert result['black_pixels'] > image.shape[0] * image.shape[1] * .25


def test_viewport_mask_matches_full_resolution_crop():
    source, limb, affine, points, venus = setup()
    full = source.copy()
    amoled.apply(full, limb, affine, points, venus)
    x, y, w, h = 0, 20, 340, 200
    viewport = source[y:y+h, x:x+w].copy()
    amoled.apply(viewport, limb, affine, points, venus, viewport=(x, y, w, h))
    np.testing.assert_array_equal(viewport[24:-24,24:-24], full[y+24:y+h-24, x+24:x+w-24])


def test_grain_is_preserved_on_a_black_background():
    import cv2 as cv
    rng=np.random.default_rng(25)
    noise=rng.normal(0,220,(160,240,3))
    image=np.rint(np.clip(14000+noise,0,65535)).astype(np.uint16)
    before=image.copy()
    limb=Limb(np.array([-1000.,-1000.]),np.array([20.,20.]),0.,np.empty((0,2)),0.)
    affine=np.array([[1.,0.,0.],[0.,1.,0.]])
    amoled.apply(image,limb,affine,np.empty((0,2)),np.array([-800.,-800.]))
    expected=np.rint(np.maximum(before.astype(np.float32)-cv.GaussianBlur(before.astype(np.float32),(0,0),3),0)).astype(np.uint16)
    np.testing.assert_array_equal(image[16:-16,16:-16],expected[16:-16,16:-16])
    assert image.mean()<before.mean()*.02
    assert np.std(image)>30


def test_catalog_star_retains_linear_light_in_reduced_preview():
    sample = 3
    viewport = (300, 90, 240, 180)
    center = (40, 28)
    point = np.array(viewport[:2]) + sample*np.array(center) + (sample-1)/2
    background = srgb_decode(np.array([.07, .10, .15]))
    light = np.array([.003, .002, .001])
    image = np.broadcast_to(np.rint(srgb_encode(background)*65535), (60,80,3)).astype(np.uint16)
    image[center[1],center[0]] = np.rint(srgb_encode(background+light)*65535)
    limb = Limb(np.array([-1000.,-1000.]),np.array([20.,20.]),0.,np.empty((0,2)),0.)
    affine = np.array([[1.,0.,0.],[0.,1.,0.]])
    amoled.apply(image,limb,affine,(),[-800.,-800.],catalog_points=[point],
                 viewport=viewport,sample=sample)
    np.testing.assert_allclose(image[center[1],center[0]],srgb_encode(light)*65535,atol=3)
    assert np.all(image[5,5]==0)


def test_resized_padding_can_be_pure_black():
    image = np.full((120, 280, 3), 12000, dtype=np.uint16)
    resized = compose.resize_wallpaper(image, (100, 200), keep=(200, 100), black_padding=True)
    assert np.all(resized[0] == 0)
    assert np.all(resized[-1] == 0)
    assert np.all(resized[100, 50] > 0)
