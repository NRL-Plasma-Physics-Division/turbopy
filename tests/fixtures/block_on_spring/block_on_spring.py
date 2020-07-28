"""Use turboPy to compute the motion of a block on a spring"""
import numpy as np
import sys
import pytest
# import xarray as xr
# import matplotlib.pyplot as plt

from turbopy import Simulation, PhysicsModule, Diagnostic
from turbopy import CSVOutputUtility, ComputeTool
from turbopy import construct_simulation_from_toml


class BlockOnSpring(PhysicsModule):
    """Use turboPy to compute the motion of a block on a spring"""
    
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.position = np.zeros((1, 3))
        self.momentum = np.zeros((1, 3))
        self.mass = input_data.get('mass', 1)
        self.spring_constant = input_data.get('spring_constant', 1)
        self.push = owner.find_tool_by_name(input_data["pusher"]).push
    
    def initialize(self):
        self.position[:] = np.array(self.input_data["x0"])
    
    def exchange_resources(self):
        self.publish_resource({"Block:position": self.position})
        self.publish_resource({"Block:momentum": self.momentum})
    
    def update(self):
        self.push(self.position, self.momentum,
                  self.mass, self.spring_constant)


class BlockDiagnostic(Diagnostic):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.data = None
        self.component = input_data.get("component", 1)
        self.output_function = None
    
    def inspect_resource(self, resource):
        if "Block:" + self.component in resource:
            self.data = resource["Block:" + self.component]
    
    def diagnose(self):
        self.output_function(self.data[0, :])
    
    def initialize(self):
        # setup output method
        functions = {"stdout": self.print_diagnose,
                     "csv": self.csv_diagnose,
                     }
        self.output_function = functions[self.input_data["output_type"]]
        if self.input_data["output_type"] == "csv":
            diagnostic_size = (self.owner.clock.num_steps + 1, 3)
            self.csv = CSVOutputUtility(self.input_data["filename"], diagnostic_size)
    
    def finalize(self):
        self.diagnose()
        if self.input_data["output_type"] == "csv":
            self.csv.finalize()
    
    def print_diagnose(self, data):
        print(data)
    
    def csv_diagnose(self, data):
        self.csv.append(data)


class ForwardEuler(ComputeTool):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.dt = None
    
    def initialize(self):
        self.dt = self.owner.clock.dt
    
    def push(self, position, momentum, mass, spring_constant):
        p0 = momentum.copy()
        momentum[:] = momentum - self.dt * spring_constant * position
        position[:] = position + self.dt * p0 / mass


class BackwardEuler(ComputeTool):
    """Implementation of the backward Euler algorithm
        y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
        Since the position and momentum are separable for this problem, this
        algorithm can be rearranged to give
        alpha = (1 + h^2 * k / m)
        alpha * x_{n+1} = x_n + h * p_n / m
                p_{n+1} = p_n + h * (-k * x_{n+1})
        """
    
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.dt = None
    
    def initialize(self):
        self.dt = self.owner.clock.dt
    
    def push(self, position, momentum, mass, spring_constant):
        factor = 1.0 / (1 + self.dt ** 2 * spring_constant / mass)
        position[:] = (position + self.dt * momentum / mass) * factor
        momentum[:] = momentum - self.dt * spring_constant * position


class LeapFrog(ComputeTool):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.dt = None
    
    def initialize(self):
        self.dt = self.owner.clock.dt
    
    def push(self, position, momentum, mass, spring_constant):
        # momentum[:] = momentum - self.dt * spring_constant * position / 2
        # position[:] = position + self.dt * momentum / mass
        # momentum[:] = momentum - self.dt * spring_constant * position / 2
        position[:] = position + self.dt * momentum / mass
        momentum[:] = momentum - self.dt * spring_constant * position


@pytest.fixture
def bos_run():
    PhysicsModule.register("BlockOnSpring", BlockOnSpring)
    Diagnostic.register("BlockDiagnostic", BlockDiagnostic)
    # ComputeTool.register("ForwardEuler", ForwardEuler)
    ComputeTool.register("BackwardEuler", BackwardEuler)
    ComputeTool.register("LeapFrog", LeapFrog)
    
    input_file = "tests/fixtures/block_on_spring/block_on_spring.toml"
    sim = construct_simulation_from_toml(input_file)
    sim.run()
