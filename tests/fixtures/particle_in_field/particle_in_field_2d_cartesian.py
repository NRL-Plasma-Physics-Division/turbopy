"""Use turboPy to propagate a charged particle in a 2D Cartesian EM wave.

This is the 2D Cartesian extension of the particle-in-field example.

The electric field is a transverse plane wave propagating at angle ``theta``
across the grid.  With wave-vector ``k = k0 * (cos θ, sin θ)`` and
polarization ``ê = (-sin θ, cos θ)`` perpendicular to ``k``:

    E(x, y, t) = E0 * ê * cos(k·r - omega*t)

So the two in-plane field components are:

    Ex(x, y, t) = -E0 * sin(θ) * cos(kx*x + ky*y - omega*t)
    Ey(x, y, t) = +E0 * cos(θ) * cos(kx*x + ky*y - omega*t)

The particle starts at the center of the grid and is pushed in (x, y) by
the in-plane electric field sampled at its initial position.

Key 2D grid features demonstrated:
    - Grid2DCartesian input format (``"coordinate_system": "cartesian2d"``)
    - ``grid.generate_field()`` → zero array of shape ``(Nx, Ny)``
    - ``grid.XX``, ``grid.YY`` → ``(Nx, Ny)`` meshgrids for field evaluation
    - ``grid.create_interpolator((x0, y0))`` → bilinear interpolation
"""
import numpy as np
import pytest
from turbopy import Simulation, PhysicsModule, Diagnostic
from turbopy import CSVOutputUtility, ComputeTool


class EMWave2D(PhysicsModule):
    """Transverse EM plane wave propagating at angle *theta* in the xy plane.

    The field is stored on the 2D Cartesian grid as two separate
    ``(Nx, Ny)`` arrays ``Ex`` and ``Ey``.  They are updated every
    timestep using the analytic solution.

    Parameters (from input_data)
    ----------------------------
    amplitude : float
        Peak electric field amplitude ``E0`` (V/m).
    omega : float
        Angular frequency (rad/s).
    angle_deg : float, optional
        Propagation angle with respect to the x-axis in degrees.
        Defaults to 45°.
    """

    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.c = 2.998e8
        self.E0 = input_data["amplitude"]
        self.omega = input_data["omega"]
        self.k0 = self.omega / self.c

        theta = np.deg2rad(input_data.get("angle_deg", 45.0))
        self.kx = self.k0 * np.cos(theta)
        self.ky = self.k0 * np.sin(theta)
        # Polarization unit vector: perpendicular to k in the xy plane
        self._pol_x = -np.sin(theta)
        self._pol_y = np.cos(theta)

        # Allocate 2D field arrays using the grid — shape (Nx, Ny)
        self.Ex = owner.grid.generate_field()
        self.Ey = owner.grid.generate_field()

        self._resources_to_share = {"EMField:Ex": self.Ex,
                                    "EMField:Ey": self.Ey}

    def _phase(self, t):
        """Phase field k·r - omega*t evaluated at every grid point."""
        # grid.XX and grid.YY are (Nx, Ny) meshgrids
        return (self.kx * self._owner.grid.XX
                + self.ky * self._owner.grid.YY
                - self.omega * t)

    def initialize(self):
        phase = self._phase(0.0)
        self.Ex[:, :] = self.E0 * self._pol_x * np.cos(phase)
        self.Ey[:, :] = self.E0 * self._pol_y * np.cos(phase)

    def update(self):
        phase = self._phase(self._owner.clock.time)
        self.Ex[:, :] = self.E0 * self._pol_x * np.cos(phase)
        self.Ey[:, :] = self.E0 * self._pol_y * np.cos(phase)


class ChargedParticle2D(PhysicsModule):
    """Charged particle pushed by the in-plane electric field.

    The particle starts at ``position = (x0, y0)`` and accumulates
    momentum from the ``(Ex, Ey)`` field components sampled at its
    initial location via bilinear interpolation.

    The 3-vector convention ``(1, 3)`` is preserved for compatibility
    with the rest of the framework; components 0 (x) and 1 (y) are used.

    Parameters (from input_data)
    ----------------------------
    position : [float, float]
        Initial ``(x, y)`` position of the particle.
    pusher : str
        Name of the registered ComputeTool to use for time-stepping.
    """

    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.Ex = None
        self.Ey = None

        x0, y0 = input_data["position"]
        self.position = np.zeros((1, 3))
        self.position[0, 0] = x0
        self.position[0, 1] = y0
        self.momentum = np.zeros((1, 3))

        # Create bilinear interpolators fixed to the initial position.
        # For a more physically accurate moving-particle simulation the
        # interpolator would be rebuilt each timestep as the position updates.
        self._interp = owner.grid.create_interpolator((x0, y0))

        self.eoverm = 1.7588e11
        self.charge = 1.6022e-19
        self.mass = 9.1094e-31
        self.push = owner.find_tool_by_name(input_data["pusher"]).push

        self._needed_resources = {"EMField:Ex": "Ex",
                                  "EMField:Ey": "Ey"}
        self._resources_to_share = {
            "ChargedParticle:position": self.position,
            "ChargedParticle:momentum": self.momentum,
        }

    def update(self):
        # Sample both field components at the (fixed) particle location
        Ex_at_particle = self._interp(self.Ex)
        Ey_at_particle = self._interp(self.Ey)
        E = np.array([Ex_at_particle, Ey_at_particle, 0.0])
        self.push(self.position, self.momentum, self.charge, self.mass, E)


class ParticleDiagnostic2D(Diagnostic):
    """Write particle position or momentum to a CSV file."""

    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.data = None
        self.component = input_data["component"]
        self.outputter = None
        self._needed_resources = {"ChargedParticle:" + self.component: "data"}

    def diagnose(self):
        self.outputter.diagnose(self.data[0, :])

    def initialize(self):
        diagnostic_size = (self._owner.clock.num_steps + 1, 3)
        self.outputter = CSVOutputUtility(self._input_data["filename"],
                                          diagnostic_size)

    def finalize(self):
        self.diagnose()
        self.outputter.finalize()


class ForwardEuler2D(ComputeTool):
    """Forward Euler integrator for in-plane particle motion.

    Updates momentum from the full 3-vector E field but only the x and y
    momentum components are driven (Ez = 0 by construction of EMWave2D).
    """

    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.dt = None

    def initialize(self):
        self.dt = self._owner.clock.dt

    def push(self, position, momentum, charge, mass, E):
        p0 = momentum.copy()
        momentum[:] = momentum + self.dt * E * charge
        position[:] = position + self.dt * p0 / mass


PhysicsModule.register("EMWave2D", EMWave2D)
PhysicsModule.register("ChargedParticle2D", ChargedParticle2D)
Diagnostic.register("ParticleDiagnostic2D", ParticleDiagnostic2D)
ComputeTool.register("ForwardEuler2D", ForwardEuler2D)


@pytest.fixture
def pif_2d_sim():
    """Return a configured Simulation for the 2D particle-in-field problem.

    The domain is a 30x30 Cartesian grid on [0, 1] x [0, 1] m.
    A plane wave propagates at 45° with omega = 2e8 rad/s.
    The particle starts at the grid centre (0.5, 0.5).
    """
    problem_config = {
        "Grid": {
            "coordinate_system": "cartesian2d",
            "Nx": 30, "x_min": 0.0, "x_max": 1.0,
            "Ny": 30, "y_min": 0.0, "y_max": 1.0,
        },
        "Clock": {
            "start_time": 0.0,
            "end_time": 1e-8,
            "num_steps": 20,
        },
        "PhysicsModules": {
            "EMWave2D": {
                "amplitude": 1.0,
                "omega": 2e8,
                "angle_deg": 45.0,
            },
            "ChargedParticle2D": {
                "position": [0.5, 0.5],
                "pusher": "ForwardEuler2D",
            },
        },
        "Tools": {"ForwardEuler2D": {}},
        "Diagnostics": {
            "directory": "particle_in_field_2d/output/",
            "output_type": "csv",
            "clock": {"filename": "time.csv"},
            "ParticleDiagnostic2D": [
                {"component": "momentum",
                 "filename": "particle_in_field_2d/output/particle_p.csv"},
                {"component": "position",
                 "filename": "particle_in_field_2d/output/particle_x.csv"},
            ],
        },
    }

    sim = Simulation(problem_config)
    sim.prepare_simulation()
    return sim
