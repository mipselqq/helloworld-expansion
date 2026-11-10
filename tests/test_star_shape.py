import numpy as np
from artemis import star_shape
from artemis.geometry import Limb


def ellipse(image, center, major, minor, angle):
    y,x=np.indices(image.shape[:2],dtype=float)
    dx,dy=x-center[0],y-center[1]
    c,s=np.cos(angle),np.sin(angle)
    r=((c*dx+s*dy)/major)**2+((-s*dx+c*dy)/minor)**2
    image[:]+=np.rint(np.exp(-.5*r)[...,None]*np.array([16000,22000,27000])).astype(np.uint16)


def axis_ratio(image, center):
    x0,y0=center
    patch=image[y0-9:y0+10,x0-9:x0+10].astype(float)
    light=np.maximum(patch.mean(axis=-1)-3000,0)
    y,x=np.mgrid[-9:10,-9:10]
    cov=np.array([[(light*x*x).sum(),(light*x*y).sum()],[(light*x*y).sum(),(light*y*y).sum()]])/light.sum()
    values=np.linalg.eigvalsh(cov)
    return np.sqrt(values[1]/values[0])


def test_circularization_conserves_color_and_flux():
    image=np.full((240,400,3),3000,np.uint16)
    ellipse(image,(60,60),3.4,1.7,.5)
    before=image.copy()
    limb=Limb(np.array([230.,120.]),np.array([45.,45.]),0.,np.empty((0,2)),0.)
    affine=np.array([[1.,0.,0.],[0.,1.,0.]])
    counts=star_shape.apply(image,np.array([[60.,60.]]),limb,affine)
    assert counts['corrected']==1
    assert axis_ratio(image,(60,60)) < axis_ratio(before,(60,60))*.85
    original=np.maximum(before[45:76,45:76].astype(float)-3000,0).sum((0,1))
    changed=np.maximum(image[45:76,45:76].astype(float)-3000,0).sum((0,1))
    np.testing.assert_allclose(changed,original,rtol=.025)
    np.testing.assert_array_equal(image[100:,100:],before[100:,100:])


def test_near_round_and_crowded_stars_are_not_changed():
    image=np.full((240,400,3),3000,np.uint16)
    ellipse(image,(60,60),2.,2.,0)
    ellipse(image,(120,60),3.,1.5,.3)
    ellipse(image,(135,60),3.,1.5,.3)
    before=image.copy()
    limb=Limb(np.array([230.,120.]),np.array([45.,45.]),0.,np.empty((0,2)),0.)
    affine=np.array([[1.,0.,0.],[0.,1.,0.]])
    counts=star_shape.apply(image,np.array([[60.,60.],[120.,60.],[135.,60.]]),limb,affine)
    assert counts['already_round']==1 and counts['crowded']==2
    np.testing.assert_array_equal(image,before)


def test_fractional_point_rounding_at_image_edge_is_skipped():
    image=np.full((100,100,3),3000,np.uint16)
    limb=Limb(np.array([50.,50.]),np.array([10.,10.]),0.,np.empty((0,2)),0.)
    affine=np.array([[1.,0.,0.],[0.,1.,0.]])
    counts=star_shape.apply(image,np.array([[89.8,40.],[40.,89.8]]),limb,affine)
    assert counts['examined']==0
