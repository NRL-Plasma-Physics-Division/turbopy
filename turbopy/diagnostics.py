"""
Diagnostics module for the turboPy computational physics simulation framework.

Diagnostics can access :class:`PhysicsModule` data. 
They are called every time step, or every N steps.
They can write to file, cache for later, update plots, etc, and they
can halt the simulation if conditions require.
"""
from abc import ABC, abstractmethod
import numpy as np
import xarray as xr

from .core import Diagnostic, Simulation


class OutputUtility(ABC):
    """Abstract base class for output utility

    An instance of an OutputUtility can (optionally) be used by diagnostic
    classes to assist with the implementation details needed for outputing
    the diagnostic information.
    """
    def __init__(self, input_data):
        pass

    @abstractmethod
    def diagnose(self, data):
        """Perform the diagnostic"""
        pass

    @abstractmethod
    def finalize(self):
        """Perform any finalization steps when the simulation is complete"""
        pass

    @abstractmethod
    def write_data(self):
        """Optional function for writting buffer to file etc."""
        pass


class PrintOutputUtility(OutputUtility):
    """OutputUtility which writes to the screen"""
    def diagnose(self, data):
        """
        Prints out data to standard output.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            1D numpy array of values.
        """
        print(data)


class CSVOutputUtility(OutputUtility):
    """Comma separated value (CSV) diagnostic output helper class

    Provides routines for writing data to a file in CSV format. This
    class can be used by Diagnostics subclassses to handle output to
    csv format.

    Parameters
    ----------
    filename : str
       File name for CSV data file.
    diagnostic_size : (int, int)
       Size of data set to be written to CSV file. First value is the
       number of time points. Second value is number of spatial points.

    Attributes
    ----------
    filename: str
        File name for CSV data file.
    buffer: :class:`numpy.ndarray`
        Buffer for storing data before it is written to file.
    buffer_index: int
        Position in buffer.
    """

    def __init__(self, filename, diagnostic_size, **kwargs):
        self._filename = filename
        self._buffer = np.zeros(diagnostic_size)
        self._buffer_index = 0

    def diagnose(self, data):
        """
        Adds 'data' into csv output buffer.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            1D numpy array of values to be added to the buffer.
        """
        self._append(data)

    def finalize(self):
        """Write the CSV data to file.
        """
        self._write_buffer()

    def write_data(self):
        """Write buffer to file"""
        self._write_buffer()

    def append(self, data):
        """Append data to the buffer.

        .. deprecated::
            `append` has been removed from the public API. Use `diagnose`
            instead.
        """
        self._append(data)

    def _append(self, data):
        """Append data to the buffer.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            1D numpy array of values to be added to the buffer.
        """
        self._buffer[self._buffer_index, :] = data
        self._buffer_index += 1

    def _write_buffer(self):
        """Write the CSV data to file.
        """
        with open(self._filename, 'wb') as f:
            np.savetxt(f, self._buffer, delimiter=",")


class NPYOutputUtility(OutputUtility):
    """NumPy formatted binary file (.npy) diagnostic output helper class

    Provides routines for writing data to a file in NumPy format. This
    class can be used by Diagnostics subclassses to handle output to
    .npy format.

    Parameters
    ----------
    filename : str
       File name for .npy data file.
    diagnostic_size : (int, int)
       Size of data set to be written to .npy file. First value is the
       number of time points. Second value is number of spatial points.

    Attributes
    ----------
    filename: str
        File name for .npy data file.
    buffer: :class:`numpy.ndarray`
        Buffer for storing data before it is written to file.
    buffer_index: int
        Position in buffer.
    """

    def __init__(self, filename, diagnostic_size, **kwargs):
        self._filename = filename
        self._buffer = np.zeros(diagnostic_size)
        self._buffer_index = 0

    def diagnose(self, data):
        """
        Adds 'data' into npy output buffer.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            1D numpy array of values to be added to the buffer.
        """
        self._append(data)

    def finalize(self):
        """Write the npy data to file.
        """
        self._write_buffer()

    def write_data(self):
        """Write buffer to file"""
        self._write_buffer()

    def _append(self, data):
        """Append data to the buffer.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            1D numpy array of values to be added to the buffer.
        """
        self._buffer[self._buffer_index, :] = data
        self._buffer_index += 1
    
    def _write_buffer(self):
        """Write the npy data to file.
        """
        with open(self._filename, 'wb') as f:
            np.save(f, self._buffer)


utilities = {
    "stdout": PrintOutputUtility,
    "csv": CSVOutputUtility,
    "npy": NPYOutputUtility
}


class IntervalHandler:
    """Calls a function (action) if a given interval has passed

    Parameters
    ----------
    interval : float, None
        The time interval to wait in between actions. If interval is None,
        then the action will be called every time.
    action : callable
        The function to call when the interval has passed
    """
    def __init__(self, interval, action):
        self._interval = interval
        self._action = action
        self._last_action = None
        self.current_step = 0

        if interval is None:
            self.perform_action = self._action_every_time

    def _action_every_time(self, time):
        self._action()
        self.current_step += 1

    def perform_action(self, time):
        """Perform the action if an interval has passed"""
        if self._check_step(time):
            self._action()
            self._last_action = time
            self.current_step += 1

    def _check_step(self, time):
        """Check if an interval has passed since last action"""
        if self._last_action is None:
            # Always run the action the first time
            return True
        return time >= self._last_action + self._interval


class PointDiagnostic(Diagnostic):
    """
    Parameters
    ----------
    owner : Simulation
       Simulation object containing current object.
    input_data : dict
       Dictionary that contains information regarding location, field,
       and output type.

    Attributes
    ----------
    location : str
        Location.
    field_name : str
        Field name.
    output : str
        Output type.
    get_value : function, None
        Function to get value given the field.
    field : str, None
        Field as dictated by resource.
    output_function : function, None
        Function for assigned output method: standard output or csv.
    csv : :class:`numpy.ndarray`, None
        numpy.ndarray being written as a csv file.
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.location = input_data["location"]
        self.field_name = input_data["field"]
        self.output = input_data["output_type"]  # "stdout"
        self.get_value = None
        self.field = None
        self.outputter = None
        self.interval = self._input_data.get('write_interval', None)
        self.handler = None
        self._needed_resources = {self.field_name: "field"}

    def diagnose(self):
        """
        Run output function given the value of the field.
        """
        self.outputter.diagnose(self.get_value(self.field))
        if self.handler:
            self.handler.perform_action(self._owner.clock.time)

    def initialize(self):
        """
        Initialize output function if provided as csv, and self.csv
        as an instance of the :class:`CSVOuputUtility` class.
        """
        # set up function to interpolate the field value
        super().initialize()
        self.get_value = self._owner.grid.create_interpolator(
                                self.location)

        # setup output method
        diagnostic_size = (self._owner.clock.num_steps + 1, 1)
        self._input_data["diagnostic_size"] = diagnostic_size

        # Use composition to provide i/o functionality
        self.outputter = utilities[self._input_data["output_type"]](**self._input_data)

        # set up interval handler
        if self.interval:
            self.handler = IntervalHandler(self.interval, self.outputter.write_data)

    def finalize(self):
        """
        Write the CSV data to file if CSV is the proper output type.
        """
        self.diagnose()
        self.outputter.finalize()


class FieldDiagnostic(Diagnostic):
    """
    Parameters
    ----------
    owner : Simulation
       Simulation object containing current object.
    input_data : dict
       Dictionary that contains information regarding location, field,
       and output type.

    Attributes
    ----------
    component : str
    field_name : str
        Field.
    output : str
        Output type.
    field : str, None
        Field as dictated by resource.
    dump_interval : int, None
        Time interval at which the diagnostic is run.
    write_interval : int, None
        Time interval at which the diagnostic buffer is written to file. If
        this is None, then the buffer is not written out until the end of
        the simulation.
    diagnose : method
        Uses the dump and write handlers to perform the diagnostic actions.
    diagnostic_size : (int, int), None
        Size of data set to be written to CSV file. First value is the
        number of time points. Second value is number of spatial points.
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)

        self.component = input_data["component"]
        self.field_name = input_data["field"]
        self.output = input_data["output_type"]  # "stdout"
        self.field = None

        # Set up handler for the diagnostic interval
        self.dump_handler = None
        self.dump_interval = self._input_data.get('dump_interval', None)

        self.outputter = None
        self.diagnostic_size = None

        # Set up handler for writing to file during the simulation
        self.write_handler = None
        self.write_interval = self._input_data.get('write_interval', None)

        # Set up resource sharing
        self._needed_resources = {self.field_name: "field"}

    def diagnose(self):
        self.dump_handler.perform_action(self._owner.clock.time)
        if self.write_handler:
            self.write_handler.perform_action(self._owner.clock.time)

    def do_diagnostic(self):
        """
        Run output_function depending on field.shape.
        """
        if len(self.field.shape) > 1:
            self.outputter.diagnose(self.field[:, self.component])
        else:
            self.outputter.diagnose(self.field)

    def initialize(self):
        """
        Initialize diagnostic_size and output function if provided as
        csv, and self.csv as an instance of the
        :class:`CSVOutputUtility` class.
        """
        super().initialize()
        self.diagnostic_size = (self._owner.clock.num_steps + 1,
                                self.field.shape[0])

        if "dump_interval" in self._input_data:
            dump_interval = self._input_data["dump_interval"]
            self.diagnostic_size = (int(np.ceil(
                self._owner.clock.end_time / dump_interval) + 1),
                self.field.shape[0])

        self._input_data['diagnostic_size'] = self.diagnostic_size

        # Use composition to provide i/o functionality
        self.outputter = utilities[self._input_data["output_type"]](**self._input_data)

        # Set up write interval handler
        if self.write_interval:
            self.write_handler = IntervalHandler(
                self.write_interval,
                self.outputter.write_data)

        # Set up the dump handler:
        self.dump_handler = IntervalHandler(
            self.dump_interval,
            self.do_diagnostic)

    def finalize(self):
        """
        Write the CSV data to file if CSV is the proper output type.
        """
        self.do_diagnostic()
        self.outputter.finalize()


class GridDiagnostic(Diagnostic):
    """Diagnostic subclass used to store and save grid data
    into a CSV file

    Parameters
    ----------
    owner : Simulation
        The 'Simulation' object that contains this object
    input_data : dict
        Dictionary containing information about this diagnostic such as
        its name

    Attributes
    ----------
    owner : Simulation
        The 'Simulation' object that contains this object
    input_data : dict
        Dictionary containing information about this diagnostic such as
        its name
    filename : str
        File name for CSV grid file
    """

    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.filename = input_data["filename"]

    def diagnose(self):
        """Grid diagnotic only runs at startup"""
        pass

    def initialize(self):
        """Save grid data into CSV file"""
        super().initialize()
        with open(self.filename, 'wb') as f:
            np.savetxt(f, self._owner.grid.r, delimiter=",")


class ClockDiagnostic(Diagnostic):
    """Diagnostic subclass used to store and save time data into a CSV
    file using the CSVOutputUtility class.

    Parameters
    ----------
    owner : Simulation
        The :class:`Simulation` object that contains this object
    input_data : dict
        Dictionary containing information about this diagnostic such as
        its name

    Attributes
    ----------
    owner : Simulation
        The :class:`Simulation` object that contains this object
    input_data : dict
        Dictionary containing information about this diagnostic such as
        its name
    filename : str
        File name for CSV time file
    csv : :class:`numpy.ndarray`
        Array to store values to be written into a CSV file
    interval : float, None
        The time interval to wait in between writing to output file. If interval is None,
        then the outputs are written only at the end of the simulation.
    handler : IntervalHandler
        The :class:`IntervalHandler` object that handles writing to output files while
        the simulation is running. Is None if the interval parameter is not specified
    """

    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.filename = input_data["filename"]
        self.csv = None
        self.interval = self._input_data.get('write_interval', None)
        self.handler = None

    def diagnose(self):
        """Append time into the csv buffer."""
        if self.handler:
            self.handler.perform_action(self._owner.clock.time)
        self.csv.diagnose(self._owner.clock.time)

    def initialize(self):
        """Initialize `self.csv` as an instance of the
        :class:`CSVOuputUtility` class."""
        super().initialize()
        diagnostic_size = (self._owner.clock.num_steps + 1, 1)
        self.csv = CSVOutputUtility(self._input_data["filename"],
                                    diagnostic_size)
        if self.interval:
            self.handler = IntervalHandler(self.interval, self.csv.write_data)

    def finalize(self):
        """Write time into self.csv and saves as a CSV file."""
        self.diagnose()
        self.csv.finalize()


class HistoryDiagnostic(Diagnostic):
    """Outputs histories/traces as functions of time

    This diagnostic assists in outputting 1D history traces. Multiple time-
    dependant quantities can be selected, and are output to a NetCDF file
    using the xarray python package.

    Examples
    --------
    When using a python dictionary to define the turboPy simulation, the
    history diagnostics can be added as in this example. Each item in the
    "traces" list has several key: value pairs. The "name" key corresponds
    to a turboPy resource that is shared by another module. The "coords"
    key is used in cases where the shared resource is more than just a
    scalar quantitiy. In this example, the position and momentum are
    length-3 vectors, with the three entries corresponding to the three
    vector components. In the case where a resources is a quantity on the
    grid, then something like ``'coords': ['x'], 'units': 'm'`` might be
    appropriate.

    Note that the 'coords' list has two items, because the shape of the
    shared numpy array is ``(1, 3)`` in this example. The first item is
    basically just a placeholder, and is called "dim0".

    >>> simulation_parameters = {"Diagnostics": {
                "histories": {
                    "filename": "output.nc",
                    "traces": [
                        {'name': 'EMField:E'},
                        {'name': 'ChargedParticle:momentum',
                        'units': 'kg m/s',
                        'coords': ["dim0", "vector component"],
                        'long_name': 'Particle Momentum'
                        },
                        {'name': 'ChargedParticle:position',
                        'units': 'm',
                        'coords': ["dim0", "vector component"],
                        'long_name': 'Particle Position'
                        },
                    ]
                }
            }
        }

    This is another example of a similar history setup, but in the format
    expected for a ``toml`` input file. ::

        [Diagnostics.histories]
        filename = "history.nc"

        [[Diagnostics.histories.traces]]
        name = 'ChargedParticle:momentum'
        units = 'kg m/s'
        coords = ["dim0", "vector component"]
        long_name = 'Particle Momentum'

        [[Diagnostics.histories.traces]]
        name = 'ChargedParticle:position'
        units = 'm'
        coords = ["dim0", "vector component"]
        long_name = 'Particle Position'

        [[Diagnostics.histories.traces]]
        name = 'EMField:E'


    References
    ----------
    [1] C. Birdsall and A. Langdon. Plasma Physics via Computer Simulation.
    Institute of Physics Series in Plasma Physics and Fluid Dynamics.
    Taylor & Francis, 2004. Page 382.
    """
    def __init__(self, owner: Simulation, input_data: dict) -> None:
        super().__init__(owner, input_data)
        self._filename = input_data['filename']
        self._traces = xr.Dataset()
        self._history_key_list = [t['name'] for t in input_data['traces']]
        self._handler = None

        # set up the interval handler
        self._interval = self._input_data.get('interval', None)
        self._handler = IntervalHandler(self._interval,
                                        self.do_diagnostic)
        self._num_outputs = self._owner.clock.num_steps
        if self._interval:
            self._num_outputs = int(np.ceil(
                self._owner.clock.end_time / self._interval))

        # get shared resources
        self._needed_resources = {k: f'_data_{k}' for k in self._history_key_list}

    def diagnose(self):
        self._handler.perform_action(self._owner.clock.time)

    def do_diagnostic(self):
        this_step = self._handler.current_step
        self._traces['time']._variable._data[this_step] = self._owner.clock.time

        for name in self._history_key_list:
            # Note, use the ellipsis here to handle multidimensional data
            self._traces[name]._variable._data[this_step, ...] = self.__dict__[f'_data_{name}']

    def initialize(self):
        # set up the time coordinate
        self._traces.coords['time'] = ('timestep', np.zeros(self._num_outputs))
        self._traces.coords['time'].attrs['units'] = 's'
        self._traces.coords['time'].attrs['long_name'] = 'Time'

        # set up the grid coordinate
        self._traces.coords['r'] = ('grid', self._owner.grid.r)
        self._traces.coords['r'].attrs['units'] = 'm'
        self._traces.coords['r'].attrs['long_name'] = 'Radius'

        # set up the history traces
        for trace in self._input_data['traces']:
            trace_data = self.__dict__[f'_data_{trace["name"]}']

            if isinstance(trace_data, xr.Dataset):
                for item in trace_data:
                    # use the xarray API to add this to the dataset
                    self._traces[item] = trace_data[item].expand_dims(
                        {'timestep': self._traces.coords['timestep']}).copy(deep=True)
                    self.__dict__[f'_data_{item}'] = trace_data[item]
                    self._history_key_list.append(item)
                self._history_key_list.remove(trace['name'])
            else:
                # Convert data into DataArray
                if not isinstance(trace_data, xr.DataArray):
                    trace_data = xr.DataArray(trace_data, dims=trace['coords'])

                # use the xarray API to add this to the dataset
                self._traces[trace['name']] = trace_data.expand_dims(
                    {'timestep': self._traces.coords['timestep']}).copy(deep=True)

                # add attributes
                if 'units' in trace:
                    self._traces[trace['name']].attrs['units'] = trace['units']
                if 'long_name' in trace:
                    self._traces[trace['name']].attrs['long_name'] = trace['long_name']

    def finalize(self):
        self._traces = self._traces.squeeze()  # remove unused dimensions
        self._traces.to_netcdf(self._filename, 'w')


Diagnostic.register("point", PointDiagnostic)
Diagnostic.register("field", FieldDiagnostic)
Diagnostic.register("grid", GridDiagnostic)
Diagnostic.register("clock", ClockDiagnostic)
Diagnostic.register("histories", HistoryDiagnostic)



# TODO: add tests for plotting
# class FieldPlottingDiagnostic(FieldDiagnostic):
#     """Extend the FieldDiagnostic to also create plots of the data"""
#     def __init__(self, owner: Simulation, input_data: dict):
#         super().__init__(owner, input_data)
# 
#     def do_diagnostic(self):
#         super().do_diagnostic()
#         plt.clf()
#         self.field.plot()
#         plt.title(f"Time: {self._owner.clock.time:0.3e} s")
#         plt.pause(0.01)
# 
#     def finalize(self):
#         super().finalize()
#         # Call show to keep the plot open
#         plt.show()



# sample = {"Diagnostics": {
#         # default values come first
#         "directory": "block_on_spring/output_leapfrog/",
#         "output_type": "netcdf",
#         "histories": {
#             "filename": "test.nc",
#             "traces": [
#             {'name': 'ChargedParticle:momentum',
#              'units': 'kg m/s',
#              'coords': ["vector component"],
#              'long_name': 'Particle Momentum'
#             },
#             {'name': 'ChargedParticle:position', 
#              'units': 'm',
#              'coords': ["vector component"],
#              'long_name': 'Particle Position'
#             },
#             {'name': 'EMField:E'}
#             ]
#         }
#     }
# }
