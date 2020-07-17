"""Helper functions for constructing Simulations"""
import qtoml as toml

from .core import Simulation

def construct_simulation_from_toml(filename: str) -> Simulation:
    """Construct a Simulation instance from a toml input file

    Parameters
    ----------
    filename : `str`
        The name of the file which contains the input specification, in
        `toml` format.

    Returns
    -------
    simulation_instance : `Simulation`
        An instance of the Simulation class, initialized using the data
        in the input file, which was converted into a python dictionary.
    """
    with open(filename) as f:
        input_data = toml.load(f)

    return Simulation(input_data)