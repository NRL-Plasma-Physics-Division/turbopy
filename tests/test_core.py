"""Tests for turbopy/core.py"""
import pytest
from turbopy.core import *


def test_grid():
    """Test initialization of the Grid class"""
    N_grid = 8
    grid_conf = {"N": N_grid,
                 "r_min": 0,
                 "r_max": 0.1}
    grid = Grid(grid_conf)
    assert grid.grid_data == grid_conf
    assert grid.r_min == 0.0
    assert grid.r_max == 0.1


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
