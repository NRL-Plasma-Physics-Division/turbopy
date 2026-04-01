"""Use turboPy to compute the motion of a block on an anisotropic 2D spring.

This is the 2D Cartesian extension of the block-on-spring example.
The block oscillates independently in x and y under restoring forces
``Fx = -kx * x`` and ``Fy = -ky * y``.  With kx != ky the trajectory
traces a Lissajous figure.  The Grid2DCartesian defines the spatial
domain over which the block moves.

Key 2D grid feature demonstrated:
    - Grid2DCartesian input format (``"coordinate_system": "cartesian2d"``)
    - ``grid.generate_field()`` to allocate a 2D potential energy array
    - ``grid.XX``, ``grid.YY`` meshgrids used to evaluate the potential on the grid
"""
import numpy as np
import pytest
from turbopy import Simulation, PhysicsModule, Diagnostic
from turbopy import CSVOutputUtility, ComputeTool


class BlockOnSpring2D(PhysicsModule):
    """Block on an anisotropic 2D spring.

    The block moves in the (x, y) plane under forces
    ``Fx = -kx * x`` and ``Fy = -ky * y``.  Different spring constants
    produce a Lissajous figure; equal constants produce an ellipse.

    The 3-vector convention ``(1, 3)`` is preserved for consistency
    with the rest of the framework; only components 0 (x) and 1 (y)
    are active.
    """

    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.position = np.zeros((1, 3))
        self.momentum = np.zeros((1, 3))
        self.mass = input_data.get("mass", 1)
        # Spring constants may differ in x and y
        kx = input_data.get("spring_constant_x", 1)
        ky = input_data.get("spring_constant_y", 1)
        self.spring_constant = np.array([kx, ky, 0.0])
        self.push = owner.find_tool_by_name(input_data["pusher"]).push

        self._resources_to_share = {"Block:position": self.position,
                                    "Block:momentum": self.momentum}

    def initialize(self):
        self.position[:] = np.array(self._input_data["x0"])
        self.momentum[:] = np.array(self._input_data.get("p0", [0, 0, 0]))

    def update(self):
        self.push(self.position, self.momentum,
                  self.mass, self.spring_constant)


class PotentialDiagnostic(Diagnostic):
    """Write the spring potential energy field to a CSV file at t=0.

    The potential energy ``V(x, y) = (1/2)(kx*x^2 + ky*y^2)`` is
    evaluated on the 2D grid using ``grid.XX`` and ``grid.YY``, then
    written out row-by-row as a flat CSV.
    """

    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.filename = input_data["filename"]
        self._written = False

    def diagnose(self):
        if not self._written:
            grid = self._owner.grid
            kx = self._owner.input_data["PhysicsModules"]["BlockOnSpring2D"][
                "spring_constant_x"]
            ky = self._owner.input_data["PhysicsModules"]["BlockOnSpring2D"][
                "spring_constant_y"]
            # Allocate a 2D field and fill with potential energy values
            V = grid.generate_field()                  # shape (Nx, Ny)
            V[:, :] = 0.5 * (kx * grid.XX ** 2
                              + ky * grid.YY ** 2)
            np.savetxt(self.filename, V, delimiter=",")
            self._written = True

    def initialize(self):
        pass

    def finalize(self):
        self.diagnose()


class BlockDiagnostic2D(Diagnostic):
    """Write block position or momentum to a CSV file."""

    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.data = None
        self.component = input_data.get("component", "position")
        self.outputter = None
        self._needed_resources = {"Block:" + self.component: "data"}

    def diagnose(self):
        self.outputter.diagnose(self.data[0, :])

    def initialize(self):
        diagnostic_size = (self._owner.clock.num_steps + 1, 3)
        self.outputter = CSVOutputUtility(self._input_data["filename"],
                                          diagnostic_size)

    def finalize(self):
        self.diagnose()
        self.outputter.finalize()


class Leapfrog2D(ComputeTool):
    """Leapfrog integrator for the 2D anisotropic spring.

    Accepts a vector ``spring_constant = [kx, ky, 0]`` so that the
    restoring force differs in each direction.
    """

    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.dt = None

    def initialize(self):
        self.dt = self._owner.clock.dt

    def push(self, position, momentum, mass, spring_constant):
        position[:] = position + self.dt * momentum / mass
        momentum[:] = momentum - self.dt * spring_constant * position


PhysicsModule.register("BlockOnSpring2D", BlockOnSpring2D)
Diagnostic.register("PotentialDiagnostic", PotentialDiagnostic)
Diagnostic.register("BlockDiagnostic2D", BlockDiagnostic2D)
ComputeTool.register("Leapfrog2D", Leapfrog2D)


@pytest.fixture
def bos_2d_run():
    """Return a problem config for the 2D block-on-spring example.

    The grid is a 20x20 Cartesian domain covering [-2, 2] x [-2, 2].
    With ``kx = 1`` and ``ky = 2`` (ratio sqrt(2):1) the trajectory
    traces a Lissajous figure that fills the spring well.
    """
    problem_config = {
        "Grid": {
            "coordinate_system": "cartesian2d",
            "Nx": 20, "x_min": -2.0, "x_max": 2.0,
            "Ny": 20, "y_min": -2.0, "y_max": 2.0,
        },
        "Clock": {
            "start_time": 0,
            "end_time": 20,
            "num_steps": 200,
        },
        "PhysicsModules": {
            "BlockOnSpring2D": {
                "mass": 1,
                "spring_constant_x": 1,
                "spring_constant_y": 2,
                "pusher": "Leapfrog2D",
                "x0": [1.0, 0.5, 0.0],
                "p0": [0.0, 0.0, 0.0],
            }
        },
        "Tools": {"Leapfrog2D": {}},
        "Diagnostics": {
            "directory": "block_on_spring_2d/output/",
            "output_type": "csv",
            "clock": {"filename": "time.csv"},
            "PotentialDiagnostic": {"filename":
                                    "block_on_spring_2d/output/potential.csv"},
            "BlockDiagnostic2D": [
                {"component": "momentum", "filename":
                 "block_on_spring_2d/output/block_p.csv"},
                {"component": "position", "filename":
                 "block_on_spring_2d/output/block_x.csv"},
            ],
        },
    }
    return problem_config
