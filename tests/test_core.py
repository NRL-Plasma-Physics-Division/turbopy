"""Tests for turbopy/core.py"""
import pytest
import numpy as np
from turbopy.core import Grid

#Grid class test methods
@pytest.fixture(name='simple_grid')
def grid_conf():
    """Pytest fixture for grid configuration dictionary"""
    grid = {"N": 8,
            "r_min": 0,
            "r_max": 0.1}
    return Grid(grid)

def test_grid_init(simple_grid):
    """Test initialization of the Grid class"""
    assert simple_grid.r_min == 0.0
    assert simple_grid.r_max == 0.1

def test_parse_grid_data(simple_grid):
    """Test parse_grid_data method in Grid class"""
    assert simple_grid.num_points == 8
    assert simple_grid.dr == 0.1/7
    grid_conf2 = {"r_min": 0,
                  "r_max": 0.1,
                  "dr": 0.1/7}
    grid2 = Grid(grid_conf2)
    assert grid2.dr == 0.1/7
    assert grid2.num_points == 8

def test_set_value_from_keys(simple_grid):
    """Test set_value_from_keys method in Grid class"""
    assert simple_grid.r_min == 0
    assert simple_grid.r_max == 0.1
    grid_conf1 = {"N": 8,
                  "r_min": 0}
    with pytest.raises(Exception):
        assert Grid(grid_conf1)

def test_generate_field(simple_grid):
    """Test generate_field method in Grid class"""
    assert np.ndarray.all(simple_grid.generate_field() == np.zeros(8))
    assert np.ndarray.all(simple_grid.generate_field(3) == np.zeros((8, 3)))

def test_generate_linear(simple_grid):
    """Test generate_linear method in Grid class"""
    comp = []
    for i in range(simple_grid.num_points):
        comp.append(i/(simple_grid.num_points - 1))
    assert np.ndarray.all(abs(simple_grid.generate_linear() - np.array(comp)) < 0.001)

def test_create_interpolator(simple_grid):
    """Test create_interpolator method in Grid class"""
    field = simple_grid.generate_linear()
    r_val = 0.05
    interp = simple_grid.create_interpolator(r_val)
    linear_value = r_val / (simple_grid.r_max - simple_grid.r_min)
    assert np.allclose(interp(field), linear_value)
