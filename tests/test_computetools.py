"""Tests for turbopy/computetools.py"""
import pytest
import numpy as np
from turbopy.computetools import *


@pytest.fixture
def interpolator():
    """Pytest fixture for basic Interpolator class"""
    return Interpolators(Simulation({}), {"type": "Interpolator"})


def test_interpolate1D(interpolator):
    """Tests for turbopy.computetools.Interpolator's interpolate1D method"""
    x = np.arange(0, 10, 1)
    y = np.exp(x)
    xnew = np.arange(0, 1, 0.1)

    f1 = interpolator.interpolate1D(x, y)
    f2 = interpolate.interp1d(x, y)
    assert np.allclose(f1(x), y)
    assert np.allclose(f1(xnew), f2(xnew))

    y = np.asarray([n**2 for n in x])
    f1 = interpolator.interpolate1D(x, y, 'quadratic')
    f2 = interpolate.interp1d(x, y, 'quadratic')
    assert np.allclose(f1(x), y)
    assert np.allclose(f1(xnew), f2(xnew))
