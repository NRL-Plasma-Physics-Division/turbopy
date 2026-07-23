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

from .core import (Diagnostic, Grid, Grid2DCartesian, Grid2DCylindrical,
                   Simulation)


class OutputUtility(ABC):
    """Abstract base class for output utility

    An instance of an OutputUtility can (optionally) be used by diagnostic
    classes to assist with the implementation details needed for outputing
    the diagnostic information.
    """
    def __init__(self, input_data):
        """Base constructor for the output utility.

        Parameters
        ----------
        input_data : dict
            Configuration passed straight from the owning diagnostic.
            The base class does not use it, but subclasses (e.g.,
            :class:`CSVOutputUtility`) accept keyword arguments such
            as ``filename`` and ``diagnostic_size``.
        """
        pass

    @abstractmethod
    def diagnose(self, data):
        """Consume one data sample.

        Subclasses implement how a sample is stored, printed, or
        forwarded.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            The value to record.  Typically a 1D array, but the
            exact shape depends on the subclass.
        """
        pass

    @abstractmethod
    def finalize(self):
        """Complete any outstanding output work.

        Called once from :meth:`turbopy.core.Diagnostic.finalize`
        when the simulation ends.  Subclasses typically flush any
        remaining buffered data to disk here.
        """
        pass

    @abstractmethod
    def write_data(self):
        """Flush the current in-memory buffer to persistent storage.

        Called intermittently by :class:`IntervalHandler` during a
        long-running simulation so that output is not lost if the
        process is killed before :meth:`finalize` runs.
        """
        pass


class PrintOutputUtility(OutputUtility):
    """OutputUtility that writes samples directly to standard output.

    Every call to :meth:`diagnose` prints immediately, so there is no
    in-memory buffer to flush; :meth:`finalize` and :meth:`write_data`
    are implemented as no-ops to satisfy the :class:`OutputUtility`
    abstract contract.
    """
    def __init__(self, **kwargs):
        """Accept and ignore any diagnostic config keys.

        Instantiated via ``utilities["stdout"](**diagnostic_input_data)``,
        so the signature must tolerate whatever keys the diagnostic passes
        (``filename``, ``diagnostic_size``, ``output_type``, ...) even
        though none of them are needed.
        """

    def diagnose(self, data):
        """
        Prints out data to standard output.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            1D numpy array of values.
        """
        print(data)

    def finalize(self):
        """No-op — output is written eagerly in :meth:`diagnose`."""

    def write_data(self):
        """No-op — output is written eagerly in :meth:`diagnose`."""


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
        """Invoke the stored action if enough time has elapsed.

        Parameters
        ----------
        time : float
            The current simulation time (typically
            ``self._owner.clock.time``).  The action is called only
            when ``time >= last_action_time + interval`` (or on the
            very first invocation).
        """
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
    """Sample a field at a single spatial point on every step.

    An interpolator is built with
    :meth:`turbopy.core.GridBase.create_interpolator` for the
    requested location, and each sample is forwarded to an
    :class:`OutputUtility` (``"stdout"``, ``"csv"``, or ``"npy"``).

    Parameters
    ----------
    owner : Simulation
        Simulation object containing this diagnostic.
    input_data : dict
        Diagnostic configuration.  Expected keys:

        - ``"location"`` : coordinate at which to sample.  Scalar for
          1D grids; a tuple for 2D grids.
        - ``"field"`` : name of the shared resource to sample.
        - ``"output_type"`` : one of ``"stdout"``, ``"csv"``,
          ``"npy"``.
        - ``"write_interval"`` (optional) : time interval between
          flushes to disk.

    Attributes
    ----------
    location : float or tuple of float
        Spatial coordinate being sampled.
    field_name : str
        Name of the shared resource to sample.
    output : str
        The ``"output_type"`` string.
    get_value : callable or None
        Interpolator built from
        :meth:`turbopy.core.GridBase.create_interpolator`; ``None``
        until :meth:`initialize` runs.
    field : :class:`numpy.ndarray` or None
        Shared field, bound by :meth:`inspect_resources`.
    outputter : :class:`OutputUtility` or None
        Output sink (``PrintOutputUtility``, ``CSVOutputUtility``,
        or ``NPYOutputUtility``).
    interval : float or None
        The ``"write_interval"`` value from the input dict.
    handler : :class:`IntervalHandler` or None
        Wraps ``outputter.write_data`` and calls it every
        ``interval`` seconds.
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
        """Build the field interpolator and the :attr:`outputter`.

        Constructs :attr:`get_value` by calling
        :meth:`turbopy.core.GridBase.create_interpolator` on the
        simulation grid, then instantiates the
        :class:`OutputUtility` selected by
        ``input_data["output_type"]`` and, if a ``write_interval``
        was configured, wraps its ``write_data`` in an
        :class:`IntervalHandler`.
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
    """Sample an entire shared field on the main-loop cadence.

    Every call to :meth:`diagnose` records the field via the
    configured :class:`OutputUtility` and (optionally) flushes the
    buffer to disk on a separate ``write_interval`` cadence.

    Parameters
    ----------
    owner : Simulation
        Simulation object containing this diagnostic.
    input_data : dict
        Diagnostic configuration.  Expected keys:

        - ``"field"`` : name of the shared field.
        - ``"component"`` : integer index into the vector component
          axis, used only when the field has more than one
          dimension.
        - ``"output_type"`` : ``"stdout"``, ``"csv"``, or ``"npy"``.
        - ``"dump_interval"`` (optional) : time interval between
          samples; if omitted, the diagnostic samples every step.
        - ``"write_interval"`` (optional) : time interval between
          disk flushes.

    Attributes
    ----------
    component : int
        Vector component index selected on multi-component fields.
    field_name : str
        Name of the shared field.
    output : str
        The ``"output_type"`` string.
    field : :class:`numpy.ndarray` or None
        The shared field, bound by :meth:`inspect_resources`.
    outputter : :class:`OutputUtility` or None
        Output sink instance.
    dump_handler : :class:`IntervalHandler` or None
        Controls how often :meth:`do_diagnostic` is invoked.
    dump_interval : float or None
        Sample cadence from the input dict.
    write_handler : :class:`IntervalHandler` or None
        Controls how often the outputter flushes to disk.
    write_interval : float or None
        Flush cadence from the input dict.  If ``None``, the buffer
        is only written at the end of the simulation.
    diagnostic_size : tuple of (int, int) or None
        ``(num_time_samples, num_spatial_points)`` used to
        preallocate the outputter buffer.
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
        """Fire the dump and write handlers if their intervals have elapsed.

        The dump handler wraps :meth:`do_diagnostic` and the write
        handler wraps ``outputter.write_data``; each is invoked at
        most once per call, and only when its :class:`IntervalHandler`
        reports that the corresponding interval has passed.
        """
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
        """Size the output buffer and construct :attr:`outputter`.

        Computes :attr:`diagnostic_size` from the number of clock
        steps (or from ``end_time / dump_interval`` when a dump
        cadence is configured) and the shape of the shared field,
        then instantiates the :class:`OutputUtility` selected by
        ``input_data["output_type"]``.  Also creates the dump and
        (optionally) write :class:`IntervalHandler` instances.
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
    """Write the grid coordinate arrays once, at startup.

    On :meth:`initialize` the grid coordinates are written to the
    configured filename as CSV.  Behavior depends on the grid type:

    - 1D :class:`~turbopy.core.Grid` -- write the ``r`` array as a
      single column.
    - :class:`~turbopy.core.Grid2DCartesian` -- write both axes
      (``x`` and ``y``) as two padded columns with an ``x,y`` header.
    - :class:`~turbopy.core.Grid2DCylindrical` -- write both axes
      (``r`` and ``z``) as two padded columns with an ``r,z`` header.

    Parameters
    ----------
    owner : Simulation
        The :class:`~turbopy.core.Simulation` that owns this
        diagnostic.
    input_data : dict
        Diagnostic configuration; must include ``"filename"``.

    Attributes
    ----------
    filename : str
        Absolute path of the CSV grid file.
    """

    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.filename = input_data["filename"]

    def diagnose(self):
        """No-op: this diagnostic writes its output once at startup."""
        pass

    def initialize(self):
        """Write the grid coordinate arrays to :attr:`filename`.

        For 2D grids both axes are written as two padded columns
        (shorter axis padded with ``NaN``); for 1D grids a single
        column of ``grid.r`` values is written.
        """
        super().initialize()
        grid = self._owner.grid
        with open(self.filename, 'wb') as f:
            if isinstance(grid, Grid2DCartesian):
                n = max(grid.Nx, grid.Ny)
                data = np.full((n, 2), np.nan)
                data[:grid.Nx, 0] = grid.x
                data[:grid.Ny, 1] = grid.y
                np.savetxt(f, data, delimiter=",", header="x,y")
            elif isinstance(grid, Grid2DCylindrical):
                n = max(grid.Nr, grid.Nz)
                data = np.full((n, 2), np.nan)
                data[:grid.Nr, 0] = grid.r
                data[:grid.Nz, 1] = grid.z
                np.savetxt(f, data, delimiter=",", header="r,z")
            else:
                np.savetxt(f, grid.r, delimiter=",")


class ClockDiagnostic(Diagnostic):
    """Record the simulation clock's ``time`` on every step.

    A :class:`CSVOutputUtility` buffer is written to disk either at
    the configured ``write_interval`` or once at the end of the run.

    Parameters
    ----------
    owner : Simulation
        The :class:`~turbopy.core.Simulation` that owns this
        diagnostic.
    input_data : dict
        Diagnostic configuration.  Expected keys:

        - ``"filename"`` : path of the CSV time file.
        - ``"write_interval"`` (optional) : cadence at which the
          buffer is flushed to disk.

    Attributes
    ----------
    filename : str
        Absolute path of the CSV file.
    outputter : :class:`CSVOutputUtility` or None
        Output helper; ``None`` until :meth:`initialize` runs.
    interval : float or None
        The ``"write_interval"`` value from the input dict.  If
        ``None``, the buffer is only written at end of run.
    handler : :class:`IntervalHandler` or None
        Wraps :meth:`CSVOutputUtility.write_data`.  ``None`` when
        ``interval`` was not configured.
    """

    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.filename = input_data["filename"]
        self.outputter = None
        self.interval = self._input_data.get('write_interval', None)
        self.handler = None

    def diagnose(self):
        """Append time into the outputter buffer."""
        if self.handler:
            self.handler.perform_action(self._owner.clock.time)
        self.outputter.diagnose(self._owner.clock.time)

    def initialize(self):
        """Initialize ``self.outputter`` as a :class:`CSVOutputUtility`."""
        super().initialize()
        diagnostic_size = (self._owner.clock.num_steps + 1, 1)
        self.outputter = CSVOutputUtility(self._input_data["filename"],
                                          diagnostic_size)
        if self.interval:
            self.handler = IntervalHandler(self.interval,
                                           self.outputter.write_data)

    def finalize(self):
        """Write time into ``self.outputter`` and save as a CSV file."""
        self.diagnose()
        self.outputter.finalize()


class HistoryDiagnostic(Diagnostic):
    """Outputs histories/traces as functions of time

    This diagnostic aggregates multiple time-dependent quantities and
    outputs them to a single NetCDF file using the :mod:`xarray` package.
    Supports 1D grids and both 2D grid types
    (:class:`~turbopy.core.Grid2DCartesian` and
    :class:`~turbopy.core.Grid2DCylindrical`).  On 2D grids the resulting
    dataset carries spatial coordinates in addition to ``time``: ``x`` and
    ``y`` for Cartesian grids, or ``r`` and ``z`` for cylindrical grids.

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
        """Delegate to the interval handler; runs :meth:`do_diagnostic` on cadence."""
        self._handler.perform_action(self._owner.clock.time)

    def do_diagnostic(self):
        """Record one time slice into the xarray dataset.

        Writes the current clock time into the ``time`` coordinate
        and copies every configured trace's current value into the
        appropriate row of the dataset.  Uses Ellipsis indexing so
        multi-dimensional traces are handled uniformly.
        """
        this_step = self._handler.current_step
        self._traces['time']._variable._data[this_step] = self._owner.clock.time

        for name in self._history_key_list:
            # Note, use the ellipsis here to handle multidimensional data
            self._traces[name]._variable._data[this_step, ...] = self.__dict__[f'_data_{name}']

    def initialize(self):
        """Allocate the xarray dataset and register per-trace metadata.

        Sets up the ``time`` and grid coordinates, then walks
        ``input_data["traces"]`` to expand every trace as a new
        variable with a ``timestep`` dimension.  Supports 1D grids and
        both 2D grid types.  On a
        :class:`~turbopy.core.Grid2DCartesian` grid the dataset gains
        ``x`` and ``y`` coordinates; on a
        :class:`~turbopy.core.Grid2DCylindrical` grid it gains ``r``
        and ``z`` coordinates.  On a 1D
        :class:`~turbopy.core.Grid` the dataset gains an ``r``
        coordinate along the ``"grid"`` dimension.
        """
        # set up the time coordinate
        self._traces.coords['time'] = ('timestep', np.zeros(self._num_outputs))
        self._traces.coords['time'].attrs['units'] = 's'
        self._traces.coords['time'].attrs['long_name'] = 'Time'

        # set up the grid coordinate(s)
        grid = self._owner.grid
        if isinstance(grid, Grid2DCartesian):
            self._traces.coords['x'] = ('x', grid.x)
            self._traces.coords['x'].attrs['units'] = 'm'
            self._traces.coords['x'].attrs['long_name'] = 'x'
            self._traces.coords['y'] = ('y', grid.y)
            self._traces.coords['y'].attrs['units'] = 'm'
            self._traces.coords['y'].attrs['long_name'] = 'y'
        elif isinstance(grid, Grid2DCylindrical):
            self._traces.coords['r'] = ('r', grid.r)
            self._traces.coords['r'].attrs['units'] = 'm'
            self._traces.coords['r'].attrs['long_name'] = 'Radius'
            self._traces.coords['z'] = ('z', grid.z)
            self._traces.coords['z'].attrs['units'] = 'm'
            self._traces.coords['z'].attrs['long_name'] = 'z'
        else:
            self._traces.coords['r'] = ('grid', grid.r)
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
        """Squeeze the dataset and write it to a NetCDF file.

        Removes size-1 dimensions from the trace dataset and writes
        the result to :attr:`_filename` using
        :meth:`xarray.Dataset.to_netcdf`.
        """
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
