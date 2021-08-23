"""Tests for turbopy/core.py"""
import pytest
import warnings
from pathlib import Path
import numpy as np
from turbopy.core import (
    ComputeTool,
    PhysicsModule,
    Diagnostic,
    Simulation,
    Grid,
    SimulationClock)


class ExampleTool(ComputeTool):
    """Example ComputeTool subclass for tests"""


class ExampleModule(PhysicsModule):
    """Example PhysicsModule subclass for tests"""
    def update(self):
        pass

    def inspect_resource(self, resource: dict):
        for attribute in resource:
            self.__setattr__(attribute, resource[attribute])


class ExampleDiagnostic(Diagnostic):
    """Example Diagnostic subclass for tests"""
    def diagnose(self):
        pass


PhysicsModule.register("ExampleModule", ExampleModule)
ComputeTool.register("ExampleTool", ExampleTool)
Diagnostic.register("ExampleDiagnostic", ExampleDiagnostic)


# Simulation class test methods
@pytest.fixture(name='simple_sim')
def sim_fixt(tmp_path):
    """Pytest fixture for basic simulation class"""
    dic = {"Grid": {"N": 2, "r_min": 0, "r_max": 1},
           "Clock": {"start_time": 0,
                     "end_time": 10,
                     "num_steps": 100},
           "Tools": {"ExampleTool": [
                        {"custom_name": "example"},
                        {"custom_name": "example2"}]},
           "PhysicsModules": {"ExampleModule": {}},
           "Diagnostics": {
               # default values come first
               "directory": f"{tmp_path}/default_output",
               "clock": {},
               "ExampleDiagnostic": [
                   {},
                   {}
                   ]
               }
           }
    return Simulation(dic)


def test_simulation_init_should_create_class_instance_when_called(simple_sim, tmp_path):
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
           "Tools": {"ExampleTool": [
                        {"custom_name": "example"},
                        {"custom_name": "example2"}]},
           "PhysicsModules": {"ExampleModule": {}},
           "Diagnostics": {
               "directory": f"{tmp_path}/default_output",
               "clock": {},
               "ExampleDiagnostic": [
                   {},
                   {}
                   ]
               }
           }
    assert simple_sim.input_data == dic


def test_read_grid_from_input_should_set_grid_attr_when_called(simple_sim):
    """Test read_grid_from_input method in Simulation class"""
    simple_sim.read_grid_from_input()
    assert simple_sim.grid.num_points == 2
    assert simple_sim.grid.r_min == 0
    assert simple_sim.grid.r_max == 1


# Test the old sharing API
class ReceivingModule(PhysicsModule):
    """Example PhysicsModule subclass for tests"""
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.data = None

    def inspect_resource(self, resource: dict):
        if 'shared' in resource:
            self.data = resource['shared']

    def initialize(self):
        # if resources are shared correctly, then this list will be accessible
        print(f'The first data item is {self.data[0]}')

    def update(self):
        pass


class SharingModule(PhysicsModule):
    """Example PhysicsModule subclass for tests"""
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.data = ['test']

    def exchange_resources(self):
        self.publish_resource({'shared': self.data})

    def update(self):
        pass


PhysicsModule.register("Receiving", ReceivingModule)
PhysicsModule.register("Sharing", SharingModule)


@pytest.fixture(name='share_sim')
def shared_simulation_fixture():
    """Pytest fixture for basic simulation class"""
    dic = {"Grid": {"N": 2, "r_min": 0, "r_max": 1},
           "Clock": {"start_time": 0,
                     "end_time": 10,
                     "num_steps": 1},
           "PhysicsModules": {
               "Receiving": {},
               "Sharing": {}
           },
           }
    return Simulation(dic)


def test_that_simulation_is_created(share_sim):
    assert share_sim.physics_modules == []


def test_that_v1_sharing_is_deprecated(share_sim):
    with pytest.deprecated_call():
        share_sim.prepare_simulation()


def test_that_shared_resource_is_available_in_initialize(share_sim):
    share_sim.prepare_simulation()
    assert len(share_sim.physics_modules) == 2
    assert len(share_sim.physics_modules[0].data) == 1
    assert (id(share_sim.physics_modules[0].data)
            == id(share_sim.physics_modules[1].data))


# Test the new sharing API
class ReceivingModuleV2(PhysicsModule):
    """Example PhysicsModule subclass for tests"""
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.data = None
        self._needed_resources = {'shared': 'data'}

    def initialize(self):
        # if resources are shared correctly, then this list will be accessible
        print(f'The first data item is {self.data[0]}')

    def update(self):
        pass


class SharingModuleV2(PhysicsModule):
    """Example PhysicsModule subclass for tests"""
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.data = ['test']
        self._resources_to_share = {'shared': self.data}

    def update(self):
        pass


PhysicsModule.register("ReceivingV2", ReceivingModuleV2)
PhysicsModule.register("SharingV2", SharingModuleV2)


# Still need to add tests for the Diagnostics with the new API


@pytest.fixture(name='share_sim_V2')
def shared_simulation_V2_fixture():
    """Pytest fixture for basic simulation class"""
    dic = {"Grid": {"N": 2, "r_min": 0, "r_max": 1},
           "Clock": {"start_time": 0,
                     "end_time": 10,
                     "num_steps": 1},
           "PhysicsModules": {
               "ReceivingV2": {},
               "SharingV2": {}
           },
           }
    return Simulation(dic)


def test_that_V2_shared_resource_is_available_in_initialize(share_sim_V2):
    share_sim_V2.prepare_simulation()
    assert len(share_sim_V2.physics_modules) == 2
    assert len(share_sim_V2.physics_modules[0].data) == 1
    assert (id(share_sim_V2.physics_modules[0].data)
            == id(share_sim_V2.physics_modules[1].data))


def test_gridless_simulation(tmp_path):
    """Test a gridless simulation"""
    dic = {"Clock": {"start_time": 0,
                     "end_time": 10,
                     "num_steps": 100},
           "Tools": {"ExampleTool": [
               {"custom_name": "example"},
               {"custom_name": "example2"}]},
           "PhysicsModules": {"ExampleModule": {}},
           "Diagnostics": {
               # default values come first
               "directory": f"{tmp_path}/default_output",
               "clock": {},
               "ExampleDiagnostic": [
                   {},
                   {}
               ]
           }
           }
    with warnings.catch_warnings(record=True) as w:
        sim = Simulation(dic)
        sim.run()
        assert sim.clock is not None
        assert sim.grid is None
        assert len(w) == 1
        assert str(w[-1].message) == "No Grid Found."


def test_subclass(simple_sim):
    """Test if subclasses are contained in Simulation"""
    assert issubclass(ExampleModule, PhysicsModule)
    assert issubclass(ExampleDiagnostic, Diagnostic)
    assert issubclass(ExampleTool, ComputeTool)


def test_read_clock_from_input_should_set_clock_attr_when_called(simple_sim):
    """Test read_clock_from_input method in Simulation class"""
    simple_sim.read_clock_from_input()
    assert simple_sim.clock._owner == simple_sim
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
    simple_sim.read_tools_from_input()
    assert simple_sim.compute_tools[0]._owner == simple_sim
    assert simple_sim.compute_tools[0]._input_data == {"type": "ExampleTool", "custom_name": "example"}
    assert simple_sim.compute_tools[1]._owner == simple_sim
    assert simple_sim.compute_tools[1]._input_data == {"type": "ExampleTool", "custom_name": "example2"}


def test_fundamental_cycle_should_advance_clock_when_called(simple_sim):
    """Test fundamental_cycle method in Simulation class"""
    simple_sim.read_clock_from_input()
    simple_sim.fundamental_cycle()
    assert simple_sim.clock.this_step == 1
    assert simple_sim.clock.time == 0.1


def test_run_should_run_simulation_while_clock_is_running(simple_sim):
    """Test run method in Simulation class"""
    simple_sim.run()
    assert simple_sim.clock.this_step == 100
    assert simple_sim.clock.time == 10


def test_turn_back_should_turn_back_time_when_called(simple_sim):
    """Test fundamental_cycle method in Simulation class"""
    simple_sim.read_clock_from_input()
    simple_sim.fundamental_cycle()
    assert simple_sim.clock.this_step == 1
    assert simple_sim.clock.time == 0.1
    simple_sim.clock.turn_back()
    assert simple_sim.clock.this_step == 0
    assert simple_sim.clock.time == 0


def test_read_modules_from_input_should_set_modules_attr_when_called(simple_sim):
    """Test read_modules_from_input method in Simulation class"""
    simple_sim.read_modules_from_input()
    assert simple_sim.physics_modules[0]._owner == simple_sim
    assert simple_sim.physics_modules[0]._input_data == {"name": "ExampleModule"}


def test_find_tool_by_name_should_identify_one_tool(simple_sim):
    simple_sim.read_tools_from_input()
    tool = simple_sim.find_tool_by_name("ExampleTool", "example")
    tool2 = simple_sim.find_tool_by_name("ExampleTool", "example2")

    assert tool._input_data["type"] == "ExampleTool"
    assert tool._input_data["custom_name"] == "example"
    assert tool2._input_data["type"] == "ExampleTool"
    assert tool2._input_data["custom_name"] == "example2"


def test_default_diagnostic_filename_is_generated_if_no_name_specified(simple_sim, tmp_path):
    """Test read_diagnostic_from_input method in Simulation class"""
    simple_sim.read_diagnostics_from_input()
    input_data = simple_sim.diagnostics[0]._input_data
    assert input_data["directory"] == str(Path(f"{tmp_path}/default_output"))
    assert input_data["filename"] == str(Path(f"{tmp_path}/default_output")
                                         / Path("clock0.out"))


def test_default_diagnostic_filename_increments_for_multiple_diagnostics(simple_sim, tmp_path):
    """Test read_diagnostic_from_input method in Simulation class"""
    simple_sim.read_diagnostics_from_input()
    assert simple_sim.diagnostics[0]._input_data["directory"] == str(Path(f"{tmp_path}/default_output"))
    assert simple_sim.diagnostics[0]._input_data["filename"] == str(Path(f"{tmp_path}/default_output")
                                                                    / Path("clock0.out"))
    input_data = simple_sim.diagnostics[2]._input_data
    assert input_data["directory"] == str(Path(f"{tmp_path}/default_output"))
    assert input_data["filename"] == str(Path(f"{tmp_path}/default_output")
                                         / Path("ExampleDiagnostic1.out"))


# Grid class test methods
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
    r_val = -0.1
    with pytest.raises(AssertionError):
        interp = simple_grid.create_interpolator(r_val)
    r_val = 0.2
    with pytest.raises(AssertionError):
        interp = simple_grid.create_interpolator(r_val)


def test_set_cartesian_volumes():
    """Test that cell volumes are set properly."""
    grid_conf2 = {"r_min": 0,
                  "r_max": 1,
                  "dr": 0.1,
                  "coordinate_system": "cartesian"}
    grid2 = Grid(grid_conf2)
    edges = grid2.cell_edges
    volumes = edges[1:] - edges[0:-1]
    assert grid2.cell_volumes.size == volumes.size
    assert np.allclose(grid2.cell_volumes, volumes)
    # Test edge-centered volumes
    volumes = np.zeros_like(edges)
    volumes[0] = edges[1] - edges[0]
    for i in range(edges.size-2):
        volumes[i+1] = 0.5 * (edges[i+2] - edges[i])
    volumes[-1] = edges[-1] - edges[-2]
    assert grid2.interface_volumes.size == volumes.size
    assert np.allclose(grid2.interface_volumes, volumes)


def test_set_cylindrical_volumes():
    """Test that cell volumes are set properly."""
    grid_conf2 = {"r_min": 0,
                  "r_max": 1,
                  "dr": 0.1,
                  "coordinate_system": "cylindrical"}
    grid2 = Grid(grid_conf2)
    edges = grid2.cell_edges
    volumes = np.pi*(edges[1:]**2 - edges[0:-1]**2)
    assert grid2.cell_volumes.size == volumes.size
    assert np.allclose(grid2.cell_volumes, volumes)
    # Test edge-centered volumes
    volumes = np.zeros_like(edges)
    volumes[0] = np.pi * (edges[1]**2 - edges[0]**2)
    for i in range(edges.size-2):
        volumes[i+1] = 0.5 * np.pi * (edges[i+2]**2 - edges[i]**2)
    volumes[-1] = np.pi * (edges[-1]**2 - edges[-2]**2)

    assert grid2.interface_volumes.size == volumes.size
    assert np.allclose(grid2.interface_volumes, volumes)


def test_set_spherical_volumes():
    """Test that cell volumes are set properly."""
    grid_conf2 = {"r_min": 0,
                  "r_max": 1,
                  "dr": 0.1,
                  "coordinate_system": "spherical"}
    grid2 = Grid(grid_conf2)
    edges = grid2.cell_edges
    volumes = 4/3 * np.pi*(edges[1:]**3 - edges[0:-1]**3)
    assert grid2.cell_volumes.size == volumes.size
    assert np.allclose(grid2.cell_volumes, volumes)
    # Test edge-centered volumes
    volumes = np.zeros_like(edges)
    volumes[0] = 4/3 * np.pi * (edges[1]**3 - edges[0]**3)
    for i in range(edges.size-2):
        volumes[i+1] = 0.5 * 4/3 * np.pi * (edges[i+2]**3 - edges[i]**3)
    volumes[-1] = 4/3 * np.pi * (edges[-1]**3 - edges[-2]**3)

    assert grid2.interface_volumes.size == volumes.size
    assert np.allclose(grid2.interface_volumes, volumes)


def test_set_cartesian_areas():
    """Test that cell areas are set properly."""
    grid_conf2 = {"r_min": 0,
                  "r_max": 1,
                  "dr": 0.1,
                  "coordinate_system": "cartesian"}
    grid2 = Grid(grid_conf2)
    areas = np.ones_like(grid2.interface_areas)
    assert grid2.interface_areas.size == areas.size
    assert np.allclose(grid2.interface_areas, areas)


def test_set_cylindrical_areas():
    """Test that cell areas are set properly."""
    grid_conf2 = {"r_min": 0,
                  "r_max": 1,
                  "dr": 0.1,
                  "coordinate_system": "cylindrical"}
    grid2 = Grid(grid_conf2)
    edges = grid2.cell_edges
    areas = 2.0*np.pi*edges
    assert grid2.interface_areas.size == areas.size
    assert np.allclose(grid2.interface_areas, areas)


def test_set_spherical_areas():
    """Test that cell areas are set properly."""
    grid_conf2 = {"r_min": 0,
                  "r_max": 1,
                  "dr": 0.1,
                  "coordinate_system": "spherical"}
    grid2 = Grid(grid_conf2)
    edges = grid2.cell_edges
    areas = 4.0 * np.pi * edges * edges
    assert grid2.interface_areas.size == areas.size
    assert np.allclose(grid2.interface_areas, areas)


# SimulationClock class test methods
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
