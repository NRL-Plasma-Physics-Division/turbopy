"""
Diagnostics module for the turboPy computational physics simulation framework.

Diagnostics can access :class:`PhysicsModule` data. 
They are called every time step, or every N steps.
They can write to file, cache for later, update plots, etc, and they
can halt the simulation if conditions require.
"""
import numpy as np

from .core import Diagnostic, Simulation


class CSVOutputUtility:
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

    def __init__(self, filename, diagnostic_size):
        self.filename = filename
        self.buffer = np.zeros(diagnostic_size)
        self.buffer_index = 0

    def append(self, data):
        """Append data to the buffer.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            1D numpy array of values to be added to the buffer.
        """
        self.buffer[self.buffer_index, :] = data
        self.buffer_index += 1

    def finalize(self):
        """Write the CSV data to file.
        """
        with open(self.filename, 'wb') as f:
            np.savetxt(f, self.buffer, delimiter=",")


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
        self.output_function = None
        self.csv = None

    def diagnose(self):
        """
        Run output function given the value of the field.
        """
        self.output_function(self.get_value(self.field))

    def inspect_resource(self, resource):
        """
        Assign attribute field if field_name given in resource.

        Parameters
        ----------
        resource : dict
            Dictionary containing information of field_name to resource.
        """
        if self.field_name in resource:
            self.field = resource[self.field_name]

    def print_diagnose(self, data):
        """
        Prints out data to standard output.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            1D numpy array of values.
        """
        print(data)

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
        functions = {"stdout": self.print_diagnose,
                     "csv": self.csv_diagnose,
                     }
        self.output_function = functions[self._input_data["output_type"]]

        if self._input_data["output_type"] == "csv":
            diagnostic_size = (self._owner.clock.num_steps + 1, 1)
            # Use composition to provide csv i/o functionality
            self.csv = CSVOutputUtility(self._input_data["filename"],
                                        diagnostic_size)

    def csv_diagnose(self, data):
        """
        Adds 'data' into csv output buffer.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            1D numpy array of values.
        """
        self.csv.append(data)

    def finalize(self):
        """
        Write the CSV data to file if CSV is the proper output type.
        """
        self.diagnose()
        if self._input_data["output_type"] == "csv":
            self.csv.finalize()


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
    dump_interval : SimulationClock, None
        Time interval between two diagnostic runs.
    last_dump : SimulationClock, None
        Time of last diagnostic run.
    diagnose : method
        Run `do_diagnostic` or `check_step` method depending on
        configuration parameters.
    diagnostic_size : (int, int), None
        Size of data set to be written to CSV file. First value is the
        number of time points. Second value is number of spatial points.
    field_was_found : bool
        Boolean representing if field was found in inspect_resource.
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)

        self.component = input_data["component"]
        self.field_name = input_data["field"]
        self.output = input_data["output_type"]  # "stdout"
        self.field = None

        self.dump_interval = None
        self.last_dump = None
        self.diagnose = self.do_diagnostic
        self.diagnostic_size = None

        self.field_was_found = False

    def check_step(self):
        """
        Run diagnostic if dump_interval time has passed since last_dump
        and update last_dump with current time if run.
        """
        if self._owner.clock.time >= self.last_dump + self.dump_interval:
            self.do_diagnostic()
            self.last_dump = self._owner.clock.time

    def do_diagnostic(self):
        """
        Run output_function depending on field.shape.
        """
        if len(self.field.shape) > 1:
            self.output_function(self.field[:, self.component])
        else:
            self.output_function(self.field)

    def inspect_resource(self, resource):
        """
        Assign attribute field if field_name given in resource and
        update boolean if field was found.

        Parameters
        ----------
        resource : dict
            Dictionary containing information of field_name to resource.
        """
        if self.field_name in resource:
            self.field_was_found = True
            self.field = resource[self.field_name]

    def print_diagnose(self, data):
        """
        Print field_name and data onto standard output.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            1D numpy array of values.
        """
        print(self.field_name, data)

    def initialize(self):
        """
        Initialize diagnostic_size and output function if provided as
        csv, and self.csv as an instance of the
        :class:`CSVOutputUtility` class.
        """
        super().initialize()
        if not self.field_was_found:
            raise (RuntimeError(f"Diagnostic field {self.field_name}"
                                " was not found"))
        self.diagnostic_size = (self._owner.clock.num_steps + 1,
                                self.field.shape[0])
        if "dump_interval" in self._input_data:
            self.dump_interval = self._input_data["dump_interval"]
            self.diagnose = self.check_step
            self.last_dump = 0
            self.diagnostic_size = (int(np.ceil(
                self._owner.clock.end_time / self.dump_interval) + 1),
                self.field.shape[0])

        # setup output method
        functions = {"stdout": self.print_diagnose,
                     "csv": self.csv_diagnose,
                     }
        self.output_function = functions[self._input_data["output_type"]]
        if self._input_data["output_type"] == "csv":
            self.csv = CSVOutputUtility(self._input_data["filename"],
                                        self.diagnostic_size)

    def csv_diagnose(self, data):
        """
        Adds 'data' into csv output buffer.

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            1D numpy array of values.
        """
        self.csv.append(data)

    def finalize(self):
        """
        Write the CSV data to file if CSV is the proper output type.
        """
        self.do_diagnostic()
        if self._input_data["output_type"] == "csv":
            self.csv.finalize()


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
    """

    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.filename = input_data["filename"]
        self.csv = None

    def diagnose(self):
        """Append time into the csv buffer."""
        self.csv.append(self._owner.clock.time)

    def initialize(self):
        """Initialize `self.csv` as an instance of the
        :class:`CSVOuputUtility` class."""
        super().initialize()
        diagnostic_size = (self._owner.clock.num_steps + 1, 1)
        self.csv = CSVOutputUtility(self._input_data["filename"],
                                    diagnostic_size)

    def finalize(self):
        """Write time into self.csv and saves as a CSV file."""
        self.diagnose()
        self.csv.finalize()


Diagnostic.register("point", PointDiagnostic)
Diagnostic.register("field", FieldDiagnostic)
Diagnostic.register("grid", GridDiagnostic)
Diagnostic.register("clock", ClockDiagnostic)
