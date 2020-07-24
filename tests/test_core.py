"""Tests for turbopy/core.py"""
import pytest
import numpy as np
from turbopy.core import Grid

def test_grid():
    """Test initialization of the Grid class"""
    n_grid = 8
    grid_conf = {"N": n_grid,
                 "r_min": 0,
                 "r_max": 0.1}
    grid = Grid(grid_conf)
    assert grid.grid_data == grid_conf
    assert grid.r_min == 0.0
    assert grid.r_max == 0.1

def test_parse_grid_data():
    """Test parse_grid_data method in Grid class"""
    n_grid = 8
    grid_conf = {"N": n_grid,
                 "r_min": 0,
                 "r_max": 0.1}
    grid1 = Grid(grid_conf)
    assert grid1.num_points == 8
    assert grid1.dr == 0.1/7
    grid_conf = {"r_min": 0,
                 "r_max": 0.1,
                 "dr": 0.1/7}
    grid2 = Grid(grid_conf)
    assert grid2.dr == 0.1/7
    assert grid2.num_points == 8

def test_set_value_from_keys():
    """Test set_value_from_keys method in Grid class"""
    n_grid = 8
    grid_conf = {"N": n_grid,
                 "r_min": 0,
                 "r_max": 0.1}
    grid1 = Grid(grid_conf)
    assert grid1.r_min == 0
    assert grid1.r_max == 0.1
    grid_conf = {"N": n_grid,
                 "r_min": 0,}
    with pytest.raises(Exception):
        assert Grid(grid_conf)

def test_generate_field():
    """Test generate_field method in Grid class"""
    n_grid = 8
    grid_conf = {"N": n_grid,
                 "r_min": 0,
                 "r_max": 0.1}
    grid1 = Grid(grid_conf)
    assert np.ndarray.all(grid1.generate_field() == np.zeros(8))
    grid_conf = {"N": n_grid,
                 "r_min": 0,
                 "r_max": 0.1}
    grid2 = Grid(grid_conf)
    assert np.ndarray.all(grid2.generate_field(3) == np.zeros((8, 3)))

def test_generate_linear():
    """Test generate_linear method in Grid class"""
    n_grid = 8
    grid_conf = {"N": n_grid,
                 "r_min": 0,
                 "r_max": 0.1}
    grid = Grid(grid_conf)
    comp = []
    for i in range(grid.num_points):
        comp.append(i/(grid.num_points - 1))
    assert np.ndarray.all(abs(grid.generate_linear() - np.array(comp)) < 0.001)

def test_create_interpolator():
    """Test create_interpolator method in Grid class"""
    n_grid = 8
    grid_conf = {"N": n_grid,
                 "r_min": 0,
                 "r_max": 0.1}
    grid = Grid(grid_conf)
    field = grid.generate_linear()
    r0 = 0.05
    interp = grid.create_interpolator(r0)
    assert abs(interp(field) - (r0 / (grid.r_max - grid.r_min))) < 0.001
    
