"""Use turboPy to compute the motion of a block on a spring"""
import numpy as np
import pytest
from turbopy import Simulation, PhysicsModule, Diagnostic
from turbopy import CSVOutputUtility, ComputeTool


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
        self.position[:] = np.array(self._input_data["x0"])

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
        self.csv = None

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
        self.output_function = functions[self._input_data["output_type"]]
        if self._input_data["output_type"] == "csv":
            diagnostic_size = (self._owner.clock.num_steps + 1, 3)
            self.csv = CSVOutputUtility(self._input_data["filename"], diagnostic_size)

    def finalize(self):
        self.diagnose()
        if self._input_data["output_type"] == "csv":
            self.csv.finalize()

    def print_diagnose(self, data):
        print(data)

    def csv_diagnose(self, data):
        self.csv.append(data)


class ForwardEuler(ComputeTool):
    """Implementation of the forward Euler algorithm

    y_{n+1} = y_n + h * f(t_n, y_n)
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.dt = None

    def initialize(self):
        self.dt = self._owner.clock.dt

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
        self.dt = self._owner.clock.dt

    def push(self, position, momentum, mass, spring_constant):
        factor = 1.0 / (1 + self.dt ** 2 * spring_constant / mass)
        position[:] = (position + self.dt * momentum / mass) * factor
        momentum[:] = momentum - self.dt * spring_constant * position


class Leapfrog(ComputeTool):
    """Implementation of the leapfrog algorithm

    x_{n+1} = x_n + h * fx(t_{n}, p_{n})
    p_{n+1} = p_n + h * fp(t_{n+1}, x_{n+1})
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.dt = None

    def initialize(self):
        self.dt = self._owner.clock.dt

    def push(self, position, momentum, mass, spring_constant):
        position[:] = position + self.dt * momentum / mass
        momentum[:] = momentum - self.dt * spring_constant * position


PhysicsModule.register("BlockOnSpring", BlockOnSpring)
Diagnostic.register("BlockDiagnostic", BlockDiagnostic)
ComputeTool.register("BlockForwardEuler", ForwardEuler)
ComputeTool.register("BackwardEuler", BackwardEuler)
ComputeTool.register("Leapfrog", Leapfrog)


@pytest.fixture
def bos_run():
    # Note: grid isn't used, but "gridless" sims aren't an option yet
    problem_config = {
        "Grid": {"N": 2, "x_min": 0, "x_max": 1},
        "Clock": {"start_time": 0,
                  "end_time": 10,
                  "num_steps": 10},
        "PhysicsModules": {
            "BlockOnSpring": {
                "mass": 1,
                "spring_constant": 1,
                "pusher": "Leapfrog",
                "x0": [0, 1, 0],
                "p0": [0, 0, 0]
            }
        },
        "Tools": {
            "Leapfrog": {},
            "BlockForwardEuler": {},
            "BackwardEuler":{}
        },
        "Diagnostics": {
            # default values come first
            "directory": "tmp/block_on_spring/output_leapfrog/",
            "output_type": "csv",
            "clock": {"filename": "time.csv"},
            "BlockDiagnostic": [
                {'component': 'momentum', 'filename': 'block_p.csv'},
                {'component': 'position', 'filename': 'block_x.csv'}
            ]
        }
    }

    sim = Simulation(problem_config)
    sim.run()

    problem_config["PhysicsModules"]["BlockOnSpring"]["pusher"] = "BlockForwardEuler"
    problem_config["Diagnostics"]["directory"] = "tmp/block_on_spring/output_forwardeuler/"
    sim = Simulation(problem_config)
    sim.run()

    problem_config["PhysicsModules"]["BlockOnSpring"]["pusher"] = "BackwardEuler"
    problem_config["Diagnostics"]["directory"] = "tmp/block_on_spring/output_backwardeuler/"
    sim = Simulation(problem_config)
    sim.run()
