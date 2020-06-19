"""Tests for turbopy/core.py"""
from turbopy.core import Grid


def test_grid():
    """Test initialization of the Grid class"""
    N_grid = 8
    grid_conf = {"N": N_grid,
                 "r_min": 0,
                 "r_max": 0.1}
    grid = Grid(grid_conf)
    assert grid.grid_data == grid_conf
    assert grid.r_min == 0.1
    assert grid.r_max == 0.1


def test_grid_2():
    """Test initialization of the Grid class"""
    N_grid = 10
    grid_conf = {"N": N_grid,
                 "r_min": 0,
                 "r_max": 0.5}
    grid = Grid(grid_conf)
    assert grid.grid_data == grid_conf
    assert grid.r_min == 0.0
    assert grid.r_max == 0.5
