import numpy as np
import xarray as xr
import pytest

from turbopy import Simulation, PhysicsModule, Diagnostic
from turbopy import CSVOutputUtility, ComputeTool
from turbopy import construct_simulation_from_toml


class EMWave(PhysicsModule):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.c = 2.998e8
        self.E0 = input_data["amplitude"]

        self.E = xr.DataArray(owner.grid.generate_field(),
                              dims="x", coords={"x": owner.grid.r})
        self.E.x.attrs["long_name"] = "Position"
        self.E.x.attrs["units"] = "m"
        self.E.attrs["long_name"] = "Electric Field"
        self.E.attrs["units"] = "V/m"

        self.omega = input_data["omega"]
        self.k = self.omega / self.c

        self._resources_to_share = {"EMField:E": self.E}

    def initialize(self):
        phase = - self.omega * 0 + self.k * (self._owner.grid.r - 0.5)
        self.E[:] = self.E0 * np.cos(2 * np.pi * phase)

    def update(self):
        phase = (- self.omega * self._owner.clock.time
                 + self.k * (self._owner.grid.r - 0.5))
        self.E[:] = self.E0 * np.cos(2 * np.pi * phase)


class ChargedParticle(PhysicsModule):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.E = None
        self.x = input_data["position"]
        self.interp_field = owner.grid.create_interpolator(self.x)
        self.position = np.zeros((1, 3))
        self.momentum = np.zeros((1, 3))
        self.eoverm = 1.7588e11
        self.charge = 1.6022e-19
        self.mass = 9.1094e-31
        self.push = owner.find_tool_by_name(input_data["pusher"]).push
        self._needed_resources = {"EMField:E": "E"}
        self._resources_to_share = {"ChargedParticle:position": self.position,
                                    "ChargedParticle:momentum": self.momentum}

    def update(self):
        E = np.array([0, self.interp_field(self.E), 0])
        self.push(self.position, self.momentum, self.charge, self.mass, E, B=0)


class ParticleDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.data = None
        self.component = input_data["component"]
        self.output_function = None
        self.outputter = None
        self._needed_resources = {"ChargedParticle:"+self.component: "data"}

    def diagnose(self):
        self.outputter.diagnose(self.data[0, :])

    def initialize(self):
        # setup output method
        diagnostic_size = (self._owner.clock.num_steps + 1, 3)
        self.outputter = CSVOutputUtility(self._input_data["filename"],
                                          diagnostic_size)

    def finalize(self):
        self.diagnose()
        self.outputter.finalize()

    def print_diagnose(self, data):
        print(data)


class ForwardEuler(ComputeTool):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.dt = None

    def initialize(self):
        self.dt = self._owner.clock.dt

    def push(self, position, momentum, charge, mass, E, B):
        p0 = momentum.copy()
        momentum[:] = momentum + self.dt * E * charge
        position[:] = position + self.dt * p0 / mass


@pytest.fixture
def pif_sim():
    PhysicsModule.register("EMWave", EMWave)
    PhysicsModule.register("ChargedParticle", ChargedParticle)
    Diagnostic.register("ParticleDiagnostic", ParticleDiagnostic)
    ComputeTool.register("ForwardEuler", ForwardEuler)
    input_file = "tests/fixtures/particle_in_field/particle_in_field.toml"
    sim = construct_simulation_from_toml(input_file)

    return sim
