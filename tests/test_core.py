"""Tests for turbopy/core.py"""
import pytest
import numpy as np
from turbopy.core import Grid

#Grid class test methods
@pytest.fixture(name='conf')
def grid_conf():
    """Pytest fixture for grid configuration dictionary"""
    grid_dict = {"N": 8,
                 "r_min": 0,
                 "r_max": 0.1}
    return grid_dict

def test_grid(conf):
    """Test initialization of the Grid class"""
    grid = Grid(conf)
    assert grid.grid_data == conf
    assert grid.r_min == 0.0
    assert grid.r_max == 0.1

def test_parse_grid_data(conf):
    """Test parse_grid_data method in Grid class"""
    grid1 = Grid(conf)
    assert grid1.num_points == 8
    assert grid1.dr == 0.1/7
    grid_conf2 = {"r_min": 0,
                  "r_max": 0.1,
                  "dr": 0.1/7}
    grid2 = Grid(grid_conf2)
    assert grid2.dr == 0.1/7
    assert grid2.num_points == 8

def test_set_value_from_keys(conf):
    """Test set_value_from_keys method in Grid class"""
    grid1 = Grid(conf)
    assert grid1.r_min == 0
    assert grid1.r_max == 0.1
    grid_conf1 = {"N": 8,
                  "r_min": 0}
    with pytest.raises(Exception):
        assert Grid(grid_conf1)

def test_generate_field(conf):
    """Test generate_field method in Grid class"""
    grid1 = Grid(conf)
    assert np.ndarray.all(grid1.generate_field() == np.zeros(8))
    grid2 = Grid(conf)
    assert np.ndarray.all(grid2.generate_field(3) == np.zeros((8, 3)))

def test_generate_linear(conf):
    """Test generate_linear method in Grid class"""
    grid = Grid(conf)
    comp = []
    for i in range(grid.num_points):
        comp.append(i/(grid.num_points - 1))
    assert np.ndarray.all(abs(grid.generate_linear() - np.array(comp)) < 0.001)

def test_create_interpolator(conf):
    """Test create_interpolator method in Grid class"""
    grid = Grid(conf)
    field = grid.generate_linear()
    r_val = 0.05
    interp = grid.create_interpolator(r_val)
    linear_value = r_val / (grid.r_max - grid.r_min)
    assert np.allclose(interp(field), linear_value)
