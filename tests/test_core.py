"""Tests for turbopy/core.py"""
import pytest
import numpy as np
from turbopy.core import *


class ExampleTool(ComputeTool):
    """Example ComputeTool subclass for tests"""


class ExampleModule(PhysicsModule):
    """Example PhysicModule subclass for tests"""
    def update(self):
        pass


# Simulation class test methods
@pytest.fixture(name='simple_sim')
def sim_fixt():
    """Pytest fixture for basic simulation class"""
    dic = {"Grid": {"N": 2, "r_min": 0, "r_max": 1},
           "Clock": {"start_time": 0,
                     "end_time": 10,
                     "num_steps": 100},
           "Tools": {"ExampleTool": {}},
           "PhysicsModules": {"ExampleModule": {}},
           }
    return Simulation(dic)


def test_simulation_init_should_create_class_instance_when_called(simple_sim):
    """Test init method for Simulation class"""
    assert simple_sim.physics_modules == []
    assert simple_sim.compute_tools == []
    assert simple_sim.diagnostics == []
    assert simple_sim.grid is None
    assert simple_sim.clock is None
    assert simple_sim.units is None
    dic = {"Grid": {"N": 2, "r_min": 0, "r_max": 1},
           "Clock": {"start_time": 0,
                     "end_time": 10,
                     "num_steps": 100},
           "Tools": {"ExampleTool": {}},
           "PhysicsModules": {"ExampleModule": {}}
           }
    assert simple_sim.input_data == dic


def test_read_grid_from_input_should_set_grid_attr_when_called(simple_sim):
    """Test read_grid_from_input method in Simulation class"""
    simple_sim.read_grid_from_input()
    assert simple_sim.grid.num_points == 2
    assert simple_sim.grid.r_min == 0
    assert simple_sim.grid.r_max == 1


def test_read_clock_from_input_should_set_clock_attr_when_called(simple_sim):
    """Test read_clock_from_input method in Simulation class"""
    simple_sim.read_clock_from_input()
    assert simple_sim.clock.owner == simple_sim
    assert simple_sim.clock.start_time == 0
    assert simple_sim.clock.time == 0
    assert simple_sim.clock.end_time == 10
    assert simple_sim.clock.this_step == 0
    assert simple_sim.clock.print_time is False
    assert simple_sim.clock.num_steps == 100
    assert simple_sim.clock.dt == 0.1
    dic = {"Grid": {"N": 2, "r_min": 0, "r_max": 1},
           "Clock": {"start_time": 0,
                     "end_time": 10,
                     "dt": 0.2,
                     "print_time": True}}
    other_sim = Simulation(dic)
    other_sim.read_clock_from_input()
    assert other_sim.clock.dt == 0.2
    assert other_sim.clock.num_steps == 50
    assert other_sim.clock.print_time is True


def test_read_tools_from_input_should_set_tools_attr_when_called(simple_sim):
    """Test read_tools_from_input method in Simulation class"""
    ComputeTool.register("ExampleTool", ExampleTool)
    simple_sim.read_tools_from_input()
    assert simple_sim.compute_tools[0].owner == simple_sim
    assert simple_sim.compute_tools[0].input_data == {"type": "ExampleTool"}


def test_fundamental_cycle_should_advance_clock_when_called(simple_sim):
    """Test fundamental_cycle method in Simulation class"""
    simple_sim.read_clock_from_input()
    simple_sim.fundamental_cycle()
    assert simple_sim.clock.this_step == 1
    assert simple_sim.clock.time == 0.1


def test_run_should_run_simulation_while_clock_is_running(simple_sim):
    """Test run method in Simulation class"""
    PhysicsModule.register("ExampleModule", ExampleModule)
    simple_sim.run()
    assert simple_sim.clock.this_step == 100
    assert simple_sim.clock.time == 10


def test_read_modules_from_input_should_set_modules_attr_when_called(simple_sim):
    """Test read_modules_from_input method in Simulation calss"""
    simple_sim.read_modules_from_input()
    assert simple_sim.physics_modules[0].owner == simple_sim
    assert simple_sim.physics_modules[0].input_data == {"name": "ExampleModule"}


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
    assert np.allclose(simple_grid.generate_field(), np.zeros(8))
    assert np.allclose(simple_grid.generate_field(3), np.zeros((8, 3)))

    
def test_generate_linear(simple_grid):
    """Test generate_linear method in Grid class"""
    comp = []
    for i in range(simple_grid.num_points):
        comp.append(i/(simple_grid.num_points - 1))
    assert np.allclose(simple_grid.generate_linear(), np.array(comp))

    
def test_create_interpolator(simple_grid):
    """Test create_interpolator method in Grid class"""
    field = simple_grid.generate_linear()
    r_val = 0.05
    interp = simple_grid.create_interpolator(r_val)
    linear_value = r_val / (simple_grid.r_max - simple_grid.r_min)
    assert np.allclose(interp(field), linear_value)


#SimulationClock class test methods
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
    
