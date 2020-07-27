"""Tests for turbopy/core.py"""
import pytest
import numpy as np
from turbopy.core import *


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
    # Also test using "dr" to set the grid spacing
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


def test_integer_num_steps():
    """Tests for initialization of SimulationClock"""
    clock_config = {'start_time': 0.0,
                    'end_time': 1e-8,
                    'dt': 1e-8 / 10.5,
                    'print_time': True}
    with pytest.raises(RuntimeError):
        SimulationClock(Simulation({}), clock_config)


def test_advance():
    """Tests `advance` method of the SimulationClock class"""
    clock_config = {'start_time': 0.0,
                    'end_time': 1e-8,
                    'num_steps': 20,
                    'print_time': True}
    clock1 = SimulationClock(Simulation({}), clock_config)
    assert clock1.num_steps == (clock1.end_time - clock1.start_time) / clock1.dt
    clock1.advance()
    assert clock1.this_step == 1
    assert clock1.time == clock1.start_time + clock1.dt * clock1.this_step


def test_is_running():
    """Tests `is_running` method of the SimulationClock class"""
    clock_config = {'start_time': 0.0,
                    'end_time': 1e-8,
                    'dt': 1e-8 / 20,
                    'print_time': True}
    clock2 = SimulationClock(Simulation({}), clock_config)
    assert clock2.num_steps == 20
    assert clock2.is_running() and clock2.this_step < clock2.num_steps
    for i in range(clock2.num_steps):
        clock2.advance()
    assert not clock2.is_running()
