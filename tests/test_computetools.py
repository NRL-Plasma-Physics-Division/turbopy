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

    y = np.asarray([n ** 2 for n in x])
    f1 = interpolator.interpolate1D(x, y, 'quadratic')
    f2 = interpolate.interp1d(x, y, 'quadratic')
    assert np.allclose(f1(x), y)
    assert np.allclose(f1(xnew), f2(xnew))


@pytest.fixture
def centered_finite():
    dic = {"Grid": {"N": 10, "r_min": 0, "r_max": 10},
           "Clock": {"start_time": 0,
                     "end_time": 10,
                     "num_steps": 100},
           "Tools": {},
           "PhysicsModules": {},
           }
    sim = Simulation(dic)
    sim.run()
    return FiniteDifference(sim, {'type': 'FiniteDifference', 'method': 'centered'})


@pytest.fixture
def upwind_finite():
    dic = {"Grid": {"N": 10, "r_min": 1, "r_max": 11},
           "Clock": {"start_time": 0,
                     "end_time": 10,
                     "num_steps": 100},
           "Tools": {},
           "PhysicsModules": {},
           }
    sim = Simulation(dic)
    sim.run()
    return FiniteDifference(sim, {'type': 'FiniteDifference', 'method': 'upwind_left'})


def test_setup_ddx_returns_respected_functions(centered_finite, upwind_finite):
    center = centered_finite.setup_ddx()
    upwind = upwind_finite.setup_ddx()
    assert center == centered_finite.centered_difference
    assert upwind == upwind_finite.upwind_left
    
    y = np.arange(0, 10)
    assert center(y).shape == (10,)
    assert upwind(y).shape == (10,)
    assert np.allclose(center(y), centered_finite.centered_difference(y))
    assert np.allclose(upwind(y), upwind_finite.upwind_left(y))


def test_ddx(centered_finite):
    N = centered_finite.owner.grid.num_points
    g = 1 / (2.0 * centered_finite.dr)
    d = centered_finite.ddx()
    assert d.shape == (N, N)
    assert np.allclose(d.toarray(), sparse.dia_matrix(([np.zeros(N) - g, np.zeros(N) + g], [-1, 1]),
                                                      shape=(N, N)).toarray())


def test_radial_curl(upwind_finite):
    N = upwind_finite.owner.grid.num_points
    d = upwind_finite.radial_curl()
    d_array = d.toarray()
    assert d.shape == (N, N)
    for ind in range(N - 1):
        assert d_array[ind + 1][ind] == d.data[0][ind]
    for ind in range(N):
        assert d_array[ind][ind] == d.data[1][ind]
    for ind in range(N - 1):
        assert d_array[ind][ind + 1] == d.data[2][ind + 1]

def test_del2_radial(upwind_finite):
    N = upwind_finite.owner.grid.num_points
    dr = upwind_finite.owner.grid.dr
    r = upwind_finite.owner.grid.r
    g1 = 1 / (2.0 * dr)
    g2 = 1 / (dr ** 2)
    below = np.append(-g1 / r[1:], [-g1]) + (g2 * np.ones(N))
    above = np.append([g1, 0], g1 / r[1:-1]) + np.append([g2, g2 * 2], g2 * np.ones(N - 2))
    diag = -2 * g2 * np.ones(N)
    
    d = upwind_finite.del2_radial()
    d_array = d.toarray()
    assert d.shape == (N, N)
    for ind in range(N - 1):
        assert d_array[ind + 1][ind] == below[ind]
    for ind in range(N):
        assert d_array[ind][ind] == diag[ind]
    for ind in range(N - 1):
        assert d_array[ind][ind + 1] == above[ind + 1]


def test_del2(upwind_finite):
    N = upwind_finite.owner.grid.num_points
    d = upwind_finite.del2()
    d_array = d.toarray()
    assert d.shape == (N, N)
    for ind in range(N - 1):
        assert d_array[ind + 1][ind] == d.data[0][ind]
    for ind in range(N):
        assert d_array[ind][ind] == d.data[1][ind]
    for ind in range(N - 1):
        assert d_array[ind][ind + 1] == d.data[2][ind + 1]


def test_ddr(centered_finite):
    N = centered_finite.owner.grid.num_points
    d = centered_finite.ddr()
    d_array = d.toarray()
    assert d.shape == (N, N)
    for ind in range(N - 1):
        assert d_array[ind + 1][ind] == d.data[0][ind]
    for ind in range(N - 1):
        assert d_array[ind][ind + 1] == d.data[1][ind + 1]


def test_BC_left_extrap(centered_finite):
    N = centered_finite.owner.grid.num_points
    d = centered_finite.BC_left_extrap()
    d_array = d.toarray()
    assert d.shape == (N, N)
    for ind in range(N):
        assert d_array[ind][ind] == d.data[0][ind]
    for ind in range(N - 1):
        assert d_array[ind][ind + 1] == d.data[1][ind + 1]
    for ind in range(N - 2):
        assert d_array[ind][ind + 2] == d.data[2][ind + 2]


def test_BC_left_avg(centered_finite):
    N = centered_finite.owner.grid.num_points
    d = centered_finite.BC_left_avg()
    d_array = d.toarray()
    assert d.shape == (N, N)
    for ind in range(N):
        assert d_array[ind][ind] == d.data[0][ind]
    for ind in range(N - 1):
        assert d_array[ind][ind + 1] == d.data[1][ind + 1]
    for ind in range(N - 2):
        assert d_array[ind][ind + 2] == d.data[2][ind + 2]


def test_BC_left_quad(centered_finite):
    N = centered_finite.owner.grid.num_points
    d = centered_finite.BC_left_quad()
    d_array = d.toarray()
    assert d.shape == (N, N)
    for ind in range(N):
        assert d_array[ind][ind] == d.data[0][ind]
    for ind in range(N - 1):
        assert d_array[ind][ind + 1] == d.data[1][ind + 1]
    for ind in range(N - 2):
        assert d_array[ind][ind + 2] == d.data[2][ind + 2]


def test_BC_left_flat(centered_finite):
    N = centered_finite.owner.grid.num_points
    d = centered_finite.BC_left_flat()
    d_array = d.toarray()
    assert d.shape == (N, N)
    for ind in range(N):
        assert d_array[ind][ind] == d.data[0][ind]
    for ind in range(N - 1):
        assert d_array[ind][ind + 1] == d.data[1][ind + 1]


def test_BC_right_extrap(centered_finite):
    N = centered_finite.owner.grid.num_points
    d = centered_finite.BC_right_extrap()
    d_array = d.toarray()
    assert d.shape == (N, N)
    for ind in range(N - 2):
        assert d_array[ind + 2][ind] == d.data[0][ind]
    for ind in range(N - 1):
        assert d_array[ind + 1][ind] == d.data[1][ind]
    for ind in range(N):
        assert d_array[ind][ind] == d.data[2][ind]
