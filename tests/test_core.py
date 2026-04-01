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
    GridBase,
    Grid2DCartesian,
    Grid2DCylindrical,
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


# ---------------------------------------------------------------------------
# Grid2DCartesian tests
# ---------------------------------------------------------------------------

@pytest.fixture(name='cart2d_grid')
def cart2d_grid_fixture():
    """2D Cartesian grid fixture: 5 x 4 points"""
    return Grid2DCartesian({
        "coordinate_system": "cartesian2d",
        "Nx": 5, "x_min": 0.0, "x_max": 1.0,
        "Ny": 4, "y_min": -1.0, "y_max": 1.0,
    })


def test_grid2d_cartesian_init(cart2d_grid):
    assert cart2d_grid.Nx == 5
    assert cart2d_grid.Ny == 4
    assert cart2d_grid.x_min == pytest.approx(0.0)
    assert cart2d_grid.x_max == pytest.approx(1.0)
    assert cart2d_grid.y_min == pytest.approx(-1.0)
    assert cart2d_grid.y_max == pytest.approx(1.0)
    assert cart2d_grid.num_points == (5, 4)
    assert cart2d_grid.shape == (5, 4)
    assert cart2d_grid.dx == pytest.approx(0.25)
    assert cart2d_grid.dy == pytest.approx(2.0 / 3)


def test_grid2d_cartesian_axes(cart2d_grid):
    assert cart2d_grid.x.shape == (5,)
    assert cart2d_grid.y.shape == (4,)
    assert cart2d_grid.x_centers.shape == (4,)
    assert cart2d_grid.y_centers.shape == (3,)
    assert cart2d_grid.x_widths.shape == (4,)
    assert cart2d_grid.y_widths.shape == (3,)
    assert np.allclose(cart2d_grid.x, np.linspace(0, 1, 5))
    assert np.allclose(cart2d_grid.y, np.linspace(-1, 1, 4))
    assert cart2d_grid.x_edges is cart2d_grid.x
    assert cart2d_grid.y_edges is cart2d_grid.y


def test_grid2d_cartesian_meshgrid(cart2d_grid):
    assert cart2d_grid.XX.shape == (5, 4)
    assert cart2d_grid.YY.shape == (5, 4)
    assert cart2d_grid.XX[0, 0] == pytest.approx(0.0)
    assert cart2d_grid.YY[0, 0] == pytest.approx(-1.0)
    assert cart2d_grid.XX[-1, 0] == pytest.approx(1.0)
    assert cart2d_grid.YY[0, -1] == pytest.approx(1.0)


def test_grid2d_cartesian_cell_volumes(cart2d_grid):
    assert cart2d_grid.cell_volumes.shape == (4, 3)
    expected = 0.25 * (2.0 / 3)
    assert np.allclose(cart2d_grid.cell_volumes, expected)
    assert np.allclose(cart2d_grid.inverse_cell_volumes, 1.0 / expected)


def test_grid2d_cartesian_generate_field_edge_centered(cart2d_grid):
    f = cart2d_grid.generate_field()
    assert f.shape == (5, 4)
    assert np.allclose(f, 0.0)


def test_grid2d_cartesian_generate_field_cell_centered(cart2d_grid):
    f = cart2d_grid.generate_field(placement_of_points="cell-centered")
    assert f.shape == (4, 3)


def test_grid2d_cartesian_generate_field_staggered(cart2d_grid):
    fx = cart2d_grid.generate_field(placement_of_points="x-edge-y-cell")
    fy = cart2d_grid.generate_field(placement_of_points="x-cell-y-edge")
    assert fx.shape == (5, 3)
    assert fy.shape == (4, 4)


def test_grid2d_cartesian_generate_field_with_components(cart2d_grid):
    f = cart2d_grid.generate_field(num_components=3)
    assert f.shape == (5, 4, 3)


def test_grid2d_cartesian_generate_field_unknown_placement_raises(cart2d_grid):
    with pytest.raises(ValueError):
        cart2d_grid.generate_field(placement_of_points="unknown")


def test_grid2d_cartesian_create_interpolator(cart2d_grid):
    """Bilinear interpolation of f(x,y) = x + y should be exact on a uniform grid."""
    f = cart2d_grid.generate_field()
    for i, xv in enumerate(cart2d_grid.x):
        for j, yv in enumerate(cart2d_grid.y):
            f[i, j] = xv + yv
    interp = cart2d_grid.create_interpolator((0.5, 0.0))
    assert np.isclose(interp(f), 0.5 + 0.0)


def test_grid2d_cartesian_interpolator_at_corner(cart2d_grid):
    f = cart2d_grid.generate_field()
    for i, xv in enumerate(cart2d_grid.x):
        for j, yv in enumerate(cart2d_grid.y):
            f[i, j] = xv + yv
    interp = cart2d_grid.create_interpolator((1.0, 1.0))
    assert np.isclose(interp(f), 2.0)


def test_grid2d_cartesian_interpolator_out_of_bounds(cart2d_grid):
    with pytest.raises(AssertionError):
        cart2d_grid.create_interpolator((-0.1, 0.0))
    with pytest.raises(AssertionError):
        cart2d_grid.create_interpolator((0.5, 2.0))


def test_grid2d_cartesian_is_gridbase(cart2d_grid):
    assert isinstance(cart2d_grid, GridBase)
    assert not isinstance(cart2d_grid, Grid)


def test_grid2d_cartesian_from_dx_dy():
    g = Grid2DCartesian({
        "coordinate_system": "cartesian2d",
        "dx": 0.25, "x_min": 0.0, "x_max": 1.0,
        "dy": 0.5,  "y_min": 0.0, "y_max": 1.0,
    })
    assert g.Nx == 5
    assert g.Ny == 3
    assert g.dx == pytest.approx(0.25)
    assert g.dy == pytest.approx(0.5)


def test_grid2d_cartesian_missing_key_raises():
    with pytest.raises(KeyError):
        Grid2DCartesian({
            "coordinate_system": "cartesian2d",
            "Nx": 5, "x_min": 0.0, "x_max": 1.0,
            # missing Ny / dy
        })


def test_grid2d_cartesian_non_integer_points_raises():
    with pytest.raises(RuntimeError):
        Grid2DCartesian({
            "coordinate_system": "cartesian2d",
            "dx": 0.3, "x_min": 0.0, "x_max": 1.0,
            "Ny": 4, "y_min": 0.0, "y_max": 1.0,
        })


# ---------------------------------------------------------------------------
# Grid2DCylindrical tests
# ---------------------------------------------------------------------------

@pytest.fixture(name='cyl2d_grid')
def cyl2d_grid_fixture():
    """2D cylindrical grid fixture: 6 x 5 points"""
    return Grid2DCylindrical({
        "coordinate_system": "cylindrical2d",
        "Nr": 6, "r_min": 0.0, "r_max": 0.1,
        "Nz": 5, "z_min": -0.1, "z_max": 0.1,
    })


def test_grid2d_cylindrical_init(cyl2d_grid):
    assert cyl2d_grid.Nr == 6
    assert cyl2d_grid.Nz == 5
    assert cyl2d_grid.r_min == pytest.approx(0.0)
    assert cyl2d_grid.r_max == pytest.approx(0.1)
    assert cyl2d_grid.z_min == pytest.approx(-0.1)
    assert cyl2d_grid.z_max == pytest.approx(0.1)
    assert cyl2d_grid.num_points == (6, 5)
    assert cyl2d_grid.shape == (6, 5)
    assert cyl2d_grid.dr == pytest.approx(0.02)
    assert cyl2d_grid.dz == pytest.approx(0.05)


def test_grid2d_cylindrical_axes(cyl2d_grid):
    assert cyl2d_grid.r.shape == (6,)
    assert cyl2d_grid.z.shape == (5,)
    assert cyl2d_grid.r_centers.shape == (5,)
    assert cyl2d_grid.z_centers.shape == (4,)
    assert np.allclose(cyl2d_grid.r, np.linspace(0.0, 0.1, 6))
    assert np.allclose(cyl2d_grid.z, np.linspace(-0.1, 0.1, 5))
    assert cyl2d_grid.r_edges is cyl2d_grid.r
    assert cyl2d_grid.z_edges is cyl2d_grid.z


def test_grid2d_cylindrical_meshgrid(cyl2d_grid):
    assert cyl2d_grid.RR.shape == (6, 5)
    assert cyl2d_grid.ZZ.shape == (6, 5)
    assert cyl2d_grid.RR[0, 0] == pytest.approx(0.0)
    assert cyl2d_grid.ZZ[0, 0] == pytest.approx(-0.1)


def test_grid2d_cylindrical_r_inv(cyl2d_grid):
    assert cyl2d_grid.r_inv[0] == pytest.approx(0.0)
    assert cyl2d_grid.r_inv[1] == pytest.approx(1.0 / cyl2d_grid.r[1])


def test_grid2d_cylindrical_r_inv_2d_shape(cyl2d_grid):
    assert cyl2d_grid.r_inv_2d.shape == (6, 5)
    assert cyl2d_grid.r_inv_2d[0, 0] == pytest.approx(0.0)
    assert cyl2d_grid.r_inv_2d[1, 2] == pytest.approx(1.0 / cyl2d_grid.r[1])


def test_grid2d_cylindrical_cell_volumes_formula(cyl2d_grid):
    r = cyl2d_grid.r
    z = cyl2d_grid.z
    expected = np.pi * np.outer(r[1:] ** 2 - r[:-1] ** 2, z[1:] - z[:-1])
    assert cyl2d_grid.cell_volumes.shape == (5, 4)
    assert np.allclose(cyl2d_grid.cell_volumes, expected)
    assert np.allclose(cyl2d_grid.inverse_cell_volumes, 1.0 / expected)


def test_grid2d_cylindrical_generate_field(cyl2d_grid):
    f = cyl2d_grid.generate_field()
    assert f.shape == (6, 5)
    assert np.allclose(f, 0.0)


def test_grid2d_cylindrical_generate_field_cell_centered(cyl2d_grid):
    f = cyl2d_grid.generate_field(placement_of_points="cell-centered")
    assert f.shape == (5, 4)


def test_grid2d_cylindrical_generate_field_staggered(cyl2d_grid):
    fr = cyl2d_grid.generate_field(placement_of_points="r-edge-z-cell")
    fz = cyl2d_grid.generate_field(placement_of_points="r-cell-z-edge")
    assert fr.shape == (6, 4)
    assert fz.shape == (5, 5)


def test_grid2d_cylindrical_generate_field_with_components(cyl2d_grid):
    f = cyl2d_grid.generate_field(num_components=2)
    assert f.shape == (6, 5, 2)


def test_grid2d_cylindrical_generate_field_unknown_placement_raises(cyl2d_grid):
    with pytest.raises(ValueError):
        cyl2d_grid.generate_field(placement_of_points="unknown")


def test_grid2d_cylindrical_create_interpolator(cyl2d_grid):
    """Bilinear interpolation of f(r,z) = r + z should be exact on a uniform grid."""
    f = cyl2d_grid.generate_field()
    for i, rv in enumerate(cyl2d_grid.r):
        for j, zv in enumerate(cyl2d_grid.z):
            f[i, j] = rv + zv
    interp = cyl2d_grid.create_interpolator((0.05, 0.0))
    assert np.isclose(interp(f), 0.05 + 0.0)


def test_grid2d_cylindrical_interpolator_out_of_bounds(cyl2d_grid):
    with pytest.raises(AssertionError):
        cyl2d_grid.create_interpolator((-0.01, 0.0))
    with pytest.raises(AssertionError):
        cyl2d_grid.create_interpolator((0.05, 0.5))


def test_grid2d_cylindrical_is_gridbase(cyl2d_grid):
    assert isinstance(cyl2d_grid, GridBase)
    assert not isinstance(cyl2d_grid, Grid)


def test_grid2d_cylindrical_from_dr_dz():
    g = Grid2DCylindrical({
        "coordinate_system": "cylindrical2d",
        "dr": 0.02, "r_min": 0.0, "r_max": 0.1,
        "dz": 0.05, "z_min": -0.1, "z_max": 0.1,
    })
    assert g.Nr == 6
    assert g.Nz == 5


def test_grid2d_cylindrical_missing_key_raises():
    with pytest.raises(KeyError):
        Grid2DCylindrical({
            "coordinate_system": "cylindrical2d",
            "Nr": 6, "r_min": 0.0, "r_max": 0.1,
            # missing Nz / dz
        })


# ---------------------------------------------------------------------------
# Simulation factory tests
# ---------------------------------------------------------------------------

def _minimal_sim_config(grid_conf):
    return {
        "Grid": grid_conf,
        "Clock": {"start_time": 0, "end_time": 1, "num_steps": 10},
        "PhysicsModules": {},
    }


def test_simulation_factory_creates_grid2d_cartesian():
    sim = Simulation(_minimal_sim_config({
        "coordinate_system": "cartesian2d",
        "Nx": 5, "x_min": 0.0, "x_max": 1.0,
        "Ny": 4, "y_min": 0.0, "y_max": 1.0,
    }))
    sim.read_grid_from_input()
    assert isinstance(sim.grid, Grid2DCartesian)
    assert sim.grid.Nx == 5
    assert sim.grid.Ny == 4


def test_simulation_factory_creates_grid2d_cylindrical():
    sim = Simulation(_minimal_sim_config({
        "coordinate_system": "cylindrical2d",
        "Nr": 6, "r_min": 0.0, "r_max": 0.1,
        "Nz": 5, "z_min": -0.1, "z_max": 0.1,
    }))
    sim.read_grid_from_input()
    assert isinstance(sim.grid, Grid2DCylindrical)
    assert sim.grid.Nr == 6


def test_simulation_factory_1d_default_unchanged():
    """Backward compatibility: no coordinate_system → 1D Grid."""
    sim = Simulation(_minimal_sim_config({
        "N": 8, "r_min": 0, "r_max": 1,
    }))
    sim.read_grid_from_input()
    assert isinstance(sim.grid, Grid)
    assert sim.grid.num_points == 8


def test_simulation_factory_1d_cylindrical_unchanged():
    """Backward compatibility: 1D 'cylindrical' still produces Grid."""
    sim = Simulation(_minimal_sim_config({
        "coordinate_system": "cylindrical",
        "N": 10, "r_min": 0, "r_max": 1,
    }))
    sim.read_grid_from_input()
    assert isinstance(sim.grid, Grid)
    assert not isinstance(sim.grid, Grid2DCylindrical)


def test_simulation_factory_1d_cartesian_unchanged():
    """Backward compatibility: explicit 'cartesian' still produces Grid."""
    sim = Simulation(_minimal_sim_config({
        "coordinate_system": "cartesian",
        "N": 10, "r_min": 0, "r_max": 1,
    }))
    sim.read_grid_from_input()
    assert isinstance(sim.grid, Grid)


def test_simulation_factory_case_insensitive():
    """coordinate_system matching is case-insensitive."""
    sim = Simulation(_minimal_sim_config({
        "coordinate_system": "Cartesian2D",
        "Nx": 5, "x_min": 0.0, "x_max": 1.0,
        "Ny": 4, "y_min": 0.0, "y_max": 1.0,
    }))
    sim.read_grid_from_input()
    assert isinstance(sim.grid, Grid2DCartesian)
