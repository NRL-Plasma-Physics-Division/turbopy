"""
Core base classes of the turboPy framework

Notes
-----
The published paper for Turbopy: A lightweight python framework for computational physics \
 can be found in the link below [1]_.

References
----------
.. [1] 1 A.S. Richardson, D.F. Gordon, S.B. Swanekamp, I.M. Rittersdorf, P.E. Adamson, \
O.S. Grannis, G.T. Morgan, A. Ostenfeld, K.L. Phlips, C.G. Sun, G. Tang, and D.J. Watkins, \
Comput. Phys. Commun. 258, 107607 (2021). \
https://doi.org/10.1016/j.cpc.2020.107607

"""
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
import warnings


class Simulation:
    """Main turboPy simulation class

    This Class "owns" all the physics modules, compute tools, and
    diagnostics. It also coordinates them. The main simulation loop is
    driven by an instance of this class.

    Parameters
    ----------
    input_data : `dict`
        This dictionary contains all parameters needed to set up a
        turboPy simulation. Each key describes a section, and the value
        is another dictionary with the needed parameters for that
        section.

        Expected keys are:

        ``"Grid"``, optional
            Dictionary containing parameters needed to define the grid.
            Currently only 1D grids are defined in turboPy.

            The expected parameters are:

            - ``"N"`` | {``"dr"`` | ``"dx"``} :
                The number of grid points (`int`) | the grid spacing
                (`float`)
            - ``"min"`` | ``"x_min"`` | ``"r_min"`` :
                The coordinate value of the minimum grid point (`float`)
            - ``"max"`` | ``"x_max"`` | ``"r_max"`` :
                The coordinate value of the maximum grid point (`float`)

        ``"Clock"``
            Dictionary of parameters needed to define the simulation
            clock.

            The expected parameters are:

            - ``"start_time"`` :
                The time for the start of the simulation (`float`)
            - ``"end_time"`` :
                The time for the end of the simulation (`float`)
            - ``"num_steps"`` | ``"dt"`` :
                The number of time steps (`int`) | the size of the time
                step (`float`)
            - ``"print_time"`` :
                `bool`, optional, default is ``False``

        ``"PhysicsModules"`` : `dict` [`str`, `dict`]
            Dictionary of :class:`PhysicsModule` items needed for the
            simulation.

            Each key in the dictionary should map to a
            :class:`PhysicsModule` subclass key in the
            :class:`PhysicsModule` registry.

            The value is a dictionary of parameters which is passed to
            the constructor for the :class:`PhysicsModule`.

        ``"Diagnostics"`` : `dict` [`str`, `dict`], optional
            Dictionary of :class:`Diagnostic` items needed for the
            simulation.

            Each key in the dictionary should map to a
            :class:`Diagnostic` subclass key in the :class:`Diagnostic`
            registry.

            The value is a dictionary of parameters which is passed to
            the constructor for the :class:`Diagnostic`.

            If the key is not found in the registry, then the key/value
            pair is interpreted as a default parameter value, and is
            added to dictionary of parameters for all of the
            :class:`Diagnostic` constructors.

            If the directory and filename keys are not specified,
            default values are created in the 
            :meth:`read_diagnostics_from_input` method.
            The default name for the directory is "default_output" and 
            the default filename is the name of the Diagnostic subclass 
            followed by a number.

        ``"Tools"`` : `dict` [`str`, `dict`], optional
            Dictionary of :class:`ComputeTool` items needed for the
            simulation.

            Each key in the dictionary should map to a
            :class:`ComputeTool` subclass key in the
            :class:`ComputeTool` registry.

            The value is a dictionary of parameters which is passed to
            the constructor for the :class:`ComputeTool`.

    Attributes
    ----------
    physics_modules : list of :class:`PhysicsModule` subclass objects
        A list of :class:`PhysicsModule` objects for this simulation.
    diagnostics : list of :class:`Diagnostic` subclass objects
        A list of :class:`Diagnostic` objects for this simulation.
    compute_tools : list of :class:`ComputeTool` subclass objects
        A list of :class:`ComputeTool` objects for this simulation.
    """

    def __init__(self, input_data: dict):
        self.physics_modules = []
        self.compute_tools = []
        self.diagnostics = []

        self.grid = None
        self.clock = None
        self.units = None

        self.input_data = input_data

    def run(self):
        """
        Runs the simulation

        This initializes the simulation, runs the main loop, and then
        finalizes the simulation.
        """
        print("Simulation is initializing")
        self.prepare_simulation()
        print("Initialization complete")

        print("Simulation is started")
        while self.clock.is_running():
            self.fundamental_cycle()

        self.finalize_simulation()
        print("Simulation complete")

    def fundamental_cycle(self):
        """
        Perform one step of the main time loop

        Executes each diagnostic and physics module, and advances
        the clock.
        """
        for d in self.diagnostics:
            d.diagnose()
        for m in self.physics_modules:
            m.reset()
        for m in self.physics_modules:
            m.update()
        self.clock.advance()

    def prepare_simulation(self):
        """
        Prepares the simulation by reading the input and initializing
        physics modules and diagnostics.
        """
        if 'Grid' in self.input_data:
            print("Reading Grid...")
            self.read_grid_from_input()
        else:
            warnings.warn('No Grid Found.')
            print("Initializing Gridless Simulation...")

        print("Initializing Simulation Clock...")
        self.read_clock_from_input()

        print("Reading Tools...")
        self.read_tools_from_input()

        print("Reading PhysicsModules...")
        self.read_modules_from_input()

        print("Reading Diagnostics...")
        self.read_diagnostics_from_input()

        print("Initializing Tools...")
        for t in self.compute_tools:
            t.initialize()

        print("Initializing PhysicsModules...")
        for m in self.physics_modules:
            m.exchange_resources()
        for m in self.physics_modules:
            m.initialize()

        print("Initializing Diagnostics...")
        for d in self.diagnostics:
            d.initialize()

    def finalize_simulation(self):
        """
        Close out the simulation

        Runs the :class:`Diagnostic.finalize()` method for each
        diagnostic.
        """
        for d in self.diagnostics:
            d.finalize()

    def read_grid_from_input(self):
        """Construct the grid based on input parameters"""
        self.grid = Grid(self.input_data["Grid"])

    def read_clock_from_input(self):
        """Construct the clock based on input parameters"""
        self.clock = SimulationClock(self, self.input_data["Clock"])

    def read_tools_from_input(self):
        """Construct :class:`ComputeTools` based on input"""
        if "Tools" in self.input_data:
            for tool_name, params in self.input_data["Tools"].items():
                tool_class = ComputeTool.lookup(tool_name)
                if not isinstance(params, list):
                    params = [params]
                for tool in params:
                    tool["type"] = tool_name
                    self.compute_tools.append(tool_class(owner=self, 
                                                         input_data=tool)) 

    def read_modules_from_input(self):
        """Construct :class:`PhysicsModule` instances based on input"""
        for physics_module_name, physics_module_data in \
                self.input_data["PhysicsModules"].items():
            print(f"Loading physics module: {physics_module_name}...")
            physics_module_class = PhysicsModule.lookup(physics_module_name)
            physics_module_data["name"] = physics_module_name
            self.physics_modules.append(physics_module_class(
                owner=self, input_data=physics_module_data))
        self.sort_modules()

    def read_diagnostics_from_input(self):
        """Construct :class:`Diagnostic` instances based on input"""
        if "Diagnostics" in self.input_data:
            # This dictionary has two types of keys:
            #    keys that are valid diagnostic types
            #    other keys, which should be passed along
            #    as "default" parameters
            diags = {k: v for k, v in
                     self.input_data["Diagnostics"].items()
                     if Diagnostic.is_valid_name(k)}
            params = {k: v for k, v in
                      self.input_data["Diagnostics"].items()
                      if not Diagnostic.is_valid_name(k)}

            if "directory" not in params:
                params["directory"] = str(Path("default_output"))

            for diag_type, d in diags.items():
                diagnostic_class = Diagnostic.lookup(diag_type)
                if not type(d) is list:
                    d = [d]
                file_num = 0
                for di in d:
                    # Values in di supersede values in params because
                    # of the order in which these are combined
                    di = {**params, **di, "type": diag_type}
                    if "filename" not in di:
                        # Set a default output filename
                        file_end = di.get("output_type", "out")
                        di["filename"] = (f"{diag_type}{file_num}"
                                          f".{file_end}")
                        file_num += 1
                    di["filename"] = str(Path(di["directory"])
                                         / Path(di["filename"]))
                    self.diagnostics.append(
                        diagnostic_class(owner=self, input_data=di))

    def sort_modules(self):
        """Sort :class:`Simulation.physics_modules` by some logic

        Unused stub for future implementation"""
        pass

    def find_tool_by_name(self, tool_name: str, custom_name: str = None):
        """Returns the :class:`ComputeTool` associated with the
        given name"""
        tools = [t for t in self.compute_tools if t.name == tool_name 
                 and t.custom_name == custom_name]
        if len(tools) == 1:
            return tools[0]
        return None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.input_data})"


class DynamicFactory(ABC):
    """Abstract class which provides dynamic factory functionality

    This base class provides a dynamic factory pattern functionality to
    classes that derive from this.
    """
    @property
    @abstractmethod
    def _factory_type_name(self):
        """Override this in derived classes with a string that
        describes type of the derived factory"""
        pass

    @property
    @abstractmethod
    def _registry(self):
        """Override this in derived classes with a dictionary that
        holds references to derived subclasses"""
        pass

    @classmethod
    def register(cls, name_to_register: str, class_to_register,
                 override=False):
        """Add a derived class to the registry"""
        if name_to_register in cls._registry and not override:
            raise ValueError("{0} '{1}' already registered".format(
                cls._factory_type_name, name_to_register))
        if not issubclass(class_to_register, cls):
            raise TypeError("{0} is not a subclass of {1}".format(
                class_to_register, cls))
        cls._registry[name_to_register] = class_to_register

    @classmethod
    def lookup(cls, name: str):
        """Look up a name in the registry, and return the associated
        derived class"""
        try:
            return cls._registry[name]
        except KeyError:
            raise KeyError("{0} '{1}' not found in registry".format(
                cls._factory_type_name, name))

    @classmethod
    def is_valid_name(cls, name: str):
        """Check if the name is in the registry"""
        return name in cls._registry


class PhysicsModule(DynamicFactory):
    """This is the base class for all physics modules

    By default, a subclass will share any public attributes as turboPy
    resources. The default resource name for these automatically shared
    attributes is the string form by combining the class name and the
    attribute name: `<class_name>_<attribute_name>`.

    If there are attributes that should not be automatically
    shared, then use the python "private" naming convention, and give
    the attribute a name which starts with an underscore.

    Parameters
    ----------
    owner : :class:`Simulation`
        Simulation class that :class:`PhysicsModule` belongs to.
    input_data : `dict`
       Dictionary that contains user defined parameters about this
       object such as its name.

    Attributes
    ----------
    _owner : :class:`Simulation`
        Simulation class that PhysicsModule belongs to.
    _module_type : `str`, ``None``
        Module type.
    _input_data : `dict`
       Dictionary that contains user defined parameters about this
       object such as its name.
    _registry : `dict`
        Registered derived ComputeTool classes.
    _factory_type_name : `str`
        Type of PhysicsModule child class.

    Notes
    -----
    This class is based on Module class in TurboWAVE.
    Because python mutable/immutable is different than C++ pointers, the
    implementation here is different. Here, a "resource" is a
    dictionary, and can have more than one thing being shared. Note that
    the value stored in the dictionary needs to be mutable. Make sure
    not to reinitialize it, because other physics modules will be
    holding a reference to it.
    """
    _factory_type_name = "Physics Module"
    _registry = {}

    def __init__(self, owner: Simulation, input_data: dict):
        self._owner = owner
        self._module_type = None
        self._input_data = input_data

    def publish_resource(self, resource: dict):
        """
        Method which implements the details of sharing resources

        Parameters
        ----------
        resource : `dict`
            resource dictionary to be shared
        """
        for k in resource.keys():
            print(f"Module {self.__class__.__name__} is sharing {k}")
        for physics_module in self._owner.physics_modules:
            physics_module.inspect_resource(resource)
        for diagnostic in self._owner.diagnostics:
            diagnostic.inspect_resource(resource)

    def inspect_resource(self, resource: dict):
        """Method for accepting resources shared by other PhysicsModules
        If your subclass needs the data described by the key, now's
        their chance to save a pointer to the data.

        Parameters
        ----------
        resource : `dict`
            resource dictionary to be shared
        """
        pass

    def exchange_resources(self):
        """Main method for sharing resources with other
        :class:`PhysicsModule` objects.

        This is the function where you call :meth:`publish_resource`,
        to tell other physics modules about data you want to share.

        By default, any "public" attributes (those with names that do
        not start with an underscore) will be shared with the key
        `<class_name>_<attribute_name>`.
        """
        shared = {f'{self.__class__.__name__}_{attribute}': value for attribute, value
                  in self.__dict__.items()
                  if not attribute.startswith('_')}
        self.publish_resource(shared)

    def update(self):
        """Do the main work of the :class:`PhysicsModule`

        This is called at every time step in the main loop.
        """
        raise NotImplementedError

    def reset(self):
        """Perform any needed reset operations

        This is called at every time step in the main loop, before any
        of the calls to `update`.
        """
        pass

    def initialize(self):
        """Perform initialization operations for this
        :class:`PhysicsModule`

        This is called before the main simulation loop
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self._input_data})"


class ComputeTool(DynamicFactory):
    """This is the base class for compute tools

    These are the compute-heavy functions, which have implementations
    of numerical methods which can be shared between physics modules.

    Parameters
    ----------
    owner : :class:`Simulation`
        Simulation class that ComputeTool belongs to.
    input_data : `dict`
        Dictionary that contains user defined parameters about this
        object such as its name.

    Attributes
    ----------
    _registry : `dict`
        Registered derived ComputeTool classes.
    _factory_type_name : `str`
        Type of ComputeTool child class
    _owner : :class:`Simulation`
        Simulation class that ComputeTool belongs to.
    _input_data : `dict`
        Dictionary that contains user defined parameters about this
        object such as its name.
    name : `str`
        Type of ComputeTool.
    custom_name: `str`
        Name given to individual instance of tool, optional.
        Used when multiple tools of the same type exist in one 
        :class:`Simulation`.
    """

    _factory_type_name = "Compute Tool"
    _registry = {}

    def __init__(self, owner: Simulation, input_data: dict):
        self._owner = owner
        self._input_data = input_data
        self.name = input_data["type"]
        self.custom_name = None
        if "custom_name" in input_data:
            self.custom_name = input_data["custom_name"]

    def initialize(self):
        """Perform any initialization operations needed for this tool"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self._input_data})"


class SimulationClock:
    """
    Clock class for turboPy

    Parameters
    ----------
    owner : :class:`Simulation`
        Simulation class that SimulationClock belongs to.
    input_data : `dict`
        Dictionary of parameters needed to define the simulation
        clock.

        The expected parameters are:

        - ``"start_time"`` :
            The time for the start of the simulation (`float`)
        - ``"end_time"`` :
            The time for the end of the simulation (`float`)
        - ``"num_steps"`` | ``"dt"`` :
            The number of time steps (`int`) | the size of the time
            step (`float`)
        - ``"print_time"`` :
            `bool`, optional, default is ``False``

    Attributes
    ----------
    _owner : :class:`Simulation`
        Simulation class that SimulationClock belongs to.
    _input_data : `dict`
        Dictionary of parameters needed to define the simulation
        clock.
    
    start_time : `float`
        Clock start time.
    time : `float`
        Current time on clock.
    end_time : `float`
        Clock end time.
    this_step : `int`
        Current time step since start.
    print_time : `bool`
        If True will print current time after each increment.
    num_steps : `int`
        Number of steps clock will take in the interval.
    dt : `float`
        Time passed at each increment.
    """

    def __init__(self, owner: Simulation, input_data: dict):
        self._owner = owner
        self._input_data = input_data
        self.start_time = input_data["start_time"]
        self.time = self.start_time
        self.end_time = input_data["end_time"]
        self.this_step = 0
        self.print_time = False
        if "print_time" in input_data:
            self.print_time = input_data["print_time"]

        if "num_steps" in input_data:
            self.num_steps = input_data["num_steps"]
            self.dt = (
                (input_data["end_time"] - input_data["start_time"])
                / input_data["num_steps"])
        elif "dt" in input_data:
            self.dt = input_data["dt"]
            self.num_steps = (self.end_time - self.start_time) / self.dt
            if not np.isclose(self.num_steps, np.rint(self.num_steps)):
                raise RuntimeError("Simulation interval is not an "
                                   "integer multiple of timestep dt")
            self.num_steps = np.int(np.rint(self.num_steps))

    def advance(self):
        """Increment the time"""
        self.this_step += 1
        self.time = self.start_time + self.dt * self.this_step
        if self.print_time:
            print(f"t = {self.time:0.4e}")
    
    def turn_back(self, num_steps=1):
        """Set the time back `num_steps` time steps"""
        self.this_step = self.this_step - num_steps
        self.time = self.start_time + self.dt * self.this_step
        if self.print_time:
            print(f"t = {self.time}")        

    def is_running(self):
        """Check if time is less than end time"""
        return self.this_step < self.num_steps

    def __repr__(self):
        return f"{self.__class__.__name__}({self._input_data})"


class Grid:
    """Grid class

    Parameters
    ----------
    input_data : `dict`
        Dictionary containing parameters needed to defined the grid.
        Currently only 1D grids are defined in turboPy.

        The expected parameters are:

        - ``"N"`` | {``"dr"`` | ``"dx"``} :
            The number of grid points (`int`) | the grid spacing
            (`float`)
        - ``"min"`` | ``"x_min"`` | ``"r_min"`` :
            The coordinate value of the minimum grid point (`float`)
        - ``"max"`` | ``"x_max"`` | ``"r_max"`` :
            The coordinate value of the maximum grid point (`float`)

    Attributes
    ----------
    _input_data : `dict`
        Dictionary containing parameters needed to defined the grid.
        Currently only 1D grids are defined in turboPy.
    r_min: `float`, ``None``
        Min of the Grid range.
    r_max : `float`, ``None``
        Max of the Grid range.
    num_points: `int`, ``None``
        Number of points on Grid.
    dr : `float`, ``None``
        Grid spacing.
    r, cell_edges : :class:`numpy.ndarray`
        Array of evenly spaced Grid values.
    cell_centers : `float`
        Value of the coordinate in the middle of each Grid cell.
    cell_widths : :class:`numpy.ndarray`
        Width of each cell in the Grid.
    r_inv : `float`
        Inverse of coordinate values at each Grid point,
        1/:class:`Grid.r`.
    """
    def __init__(self, input_data: dict):
        self._input_data = input_data
        self.r_min = None
        self.r_max = None
        self.num_points = None
        self.dr = None
        self.coordinate_system = "cartesian"
        self.r = None
        self.cell_edges = None
        self.cell_centers = None
        self.cell_widths = None
        self.r_inv = None

        self.cell_volumes = None
        self.inverse_cell_volumes = None
        self.interface_areas = None
        self.interface_volumes = None
        self.inverse_interface_volumes = None

        self.parse_grid_data()
        self.set_grid_points()
        self.set_volume_and_area_elements()

    def parse_grid_data(self):
        """
        Initializes the grid spacing, range, and number of points on the
        grid from :class:`Grid._input_data`.

        Raises
        ------
        RuntimeError
            If the range and step size causes a non-integer number of
            grid points.
        """
        self.set_value_from_keys("r_min", {"min", "x_min", "r_min"})
        self.set_value_from_keys("r_max", {"max", "x_max", "r_max"})
        if "N" in self._input_data:
            self.num_points = self._input_data["N"]
            self.dr = (self.r_max - self.r_min) / (self.num_points - 1)
        else:
            self.set_value_from_keys("dr", {"dr", "dx"})
            self.num_points = 1 + (self.r_max - self.r_min) / self.dr
            if not self.num_points % 1 == 0:
                raise (RuntimeError("Invalid grid spacing: "
                                    "configuration does not imply "
                                    "integer number of grid points"))
            self.num_points = np.int(self.num_points)

        # set the coordinate system
        if "coordinate_system" in self._input_data:
            self.coordinate_system = self._input_data["coordinate_system"]
        self.coordinate_system = self.coordinate_system.lower().strip()

    def set_value_from_keys(self, var_name, options):
        """
        Initializes a specified attribute to a value provided in
        :class:`Grid._input_data`.

        Parameters
        ----------
        var_name : `str`
            Attribute name to be initialized.
        options : `set`
            Set of keys in :class:`Grid._input_data` to search
            for values.

        Raises
        ------
        KeyError
            If none of the keys in `options` are present in
            :class:`Grid._input_data`.
        """
        for name in options:
            if name in self._input_data:
                setattr(self, var_name, self._input_data[name])
                return
        raise (KeyError("Grid configuration for " + var_name
                        + " not found."))

    def set_grid_points(self):
        self.r = (self.r_min + (self.r_max - self.r_min) *
                  self.generate_linear())
        self.cell_edges = self.r
        self.cell_centers = (self.r[1:] + self.r[:-1]) / 2
        self.cell_widths = (self.r[1:] - self.r[:-1])
        with np.errstate(divide='ignore'):
            self.r_inv = 1 / self.r
            self.r_inv[self.r_inv == np.inf] = 0

    def generate_field(self, num_components=1,
                       placement_of_points="edge-centered"):
        """Returns squeezed :class:`numpy.ndarray` of zeros with
        dimensions :class:`Grid.num_points` and `num_components`.

        Parameters
        ----------
        num_components : int, defaults to 1
            Number of vector components at each point.
        placement_of_points : str, defaults to "edge-centered"
            Designate position of points on grid
        Returns
        -------
        :class:`numpy.ndarray`
            Squeezed array of zeros.
        """
        number_of_field_points = None
        if placement_of_points == "edge-centered":
            number_of_field_points = self.num_points
        elif placement_of_points == "cell-centered":
            number_of_field_points = self.num_points - 1
        else:
            raise ValueError("Unknown placement option specified")
        return np.squeeze(np.zeros((number_of_field_points, num_components)))

    def generate_linear(self):
        """Returns :class:`numpy.ndarray` with :class:`Grid.num_points`
        evenly spaced in the interval between 0 and 1.

         Returns
         -------
         :class:`numpy.ndarray`
            Evenly spaced array.
         """
        return np.linspace(0, 1, self.num_points)

    def create_interpolator(self, r0):
        """Return a function which linearly interpolates any field on
        this grid, to the point ``r0``.

        Parameters
        ----------
        r0 : `float`
            The requested point on the grid.

        Returns
        -------
        function
            A function which takes a grid quantity ``y`` and returns the
            interpolated value of ``y`` at the point ``r0``.
        """
        assert (r0 >= self.r_min), "Requested point is not in the grid"
        assert (r0 <= self.r_max), "Requested point is not in the grid"
        i, = np.where((r0 - self.dr < self.r) & (self.r < r0 + self.dr))
        assert (len(i) in [1, 2]), ("Error finding requested point"
                                    "in the grid")
        if len(i) == 1:
            return lambda y: y[i]
        if len(i) == 2:
            # linearly interpolate
            def interpval(yvec):
                """A function which takes a grid quantity ``y`` and
                returns the interpolated value of ``y`` at the
                point ``r0``.

                Parameters
                ----------
                yvec : :class:`numpy.ndarray`
                    A vector describing a quantity ``y`` on the grid

                Returns
                -------
                `float`
                    Value of ``y`` linearly interpolated to the
                    point ``r0``
                """
                rvals = self.r[i]
                y = yvec[i]
                return y[0] + ((r0 - rvals[0]) * (y[1] - y[0])
                               / (rvals[1] - rvals[0]))

            return interpval

    def set_volume_and_area_elements(self):
        if self.coordinate_system == 'cartesian':
            self.set_cartesian_volumes()
            self.set_cartesian_areas()
        elif self.coordinate_system == 'cylindrical':
            self.set_cylindrical_volumes()
            self.set_cylindrical_areas()
        elif self.coordinate_system == 'spherical':
            self.set_spherical_volumes()
            self.set_spherical_areas()
        else:
            raise ValueError(f'Coordinate system '
                             f'{self.coordinate_system} is undefined')
        self.set_interface_volumes()

    def set_cartesian_volumes(self):
        self.cell_volumes = self.cell_edges[1:] - self.cell_edges[:-1]
        self.inverse_cell_volumes = 1./self.cell_volumes

    def set_cylindrical_volumes(self):
        scratch = self.cell_edges ** 2
        self.cell_volumes = np.pi * (scratch[1:] - scratch[:-1])
        self.inverse_cell_volumes = 1./self.cell_volumes

    def set_spherical_volumes(self):
        scratch = self.cell_edges ** 3
        self.cell_volumes = 4/3 * np.pi * (scratch[1:] - scratch[:-1])
        self.inverse_cell_volumes = 1./self.cell_volumes

    def set_cartesian_areas(self):
        self.interface_areas = np.ones_like(self.cell_edges)

    def set_cylindrical_areas(self):
        self.interface_areas = 2.0 * np.pi * self.cell_edges

    def set_spherical_areas(self):
        self.interface_areas = 4.0 * np.pi * self.cell_edges ** 2

    def set_interface_volumes(self):
        self.interface_volumes = np.zeros_like(self.cell_edges)
        self.inverse_interface_volumes = np.zeros_like(self.interface_volumes)

        self.interface_volumes[0] = self.cell_volumes[0]
        self.interface_volumes[1:-1] = 0.5 * (self.cell_volumes[1:]
                                              + self.cell_volumes[0:-1])
        self.interface_volumes[-1] = self.cell_volumes[-1]

        self.inverse_interface_volumes[0] = self.inverse_cell_volumes[0]
        self.inverse_interface_volumes[1:-1] = 0.5 * (self.inverse_cell_volumes[1:]
                                                      + self.inverse_cell_volumes[0:-1])
        self.inverse_interface_volumes[-1] = self.inverse_cell_volumes[-1]

    def __repr__(self):
        return f"{self.__class__.__name__}({self._input_data})"


class Diagnostic(DynamicFactory):
    """Base diagnostic class.

    Parameters
    ----------
    owner: Simulation
        The Simulation object that owns this object
    input_data: `dict`
        Dictionary that contains user defined parameters about this
        object such as its name.

    Attributes
    ----------
    _factory_type_name: `str`
        Type of DynamicFactory child class
    _registry: `dict`
        Registered derived Diagnostic classes
    _owner: Simulation
        The Simulation object that contains this object
    _input_data: `dict`
        Dictionary that contains user defined parameters about this
        object such as its name.
    """

    _factory_type_name = "Diagnostic"
    _registry = {}

    def __init__(self, owner: Simulation, input_data: dict):
        self._owner = owner
        self._input_data = input_data

    def inspect_resource(self, resource: dict):
        """Save references to data from other PhysicsModules

        If your subclass needs the data described by the key, now's
        their chance to save a reference to the data

        Parameters
        ----------
        resource: `dict`
            A dictionary containing references to data shared by other
            PhysicsModules.
        """
        pass

    def diagnose(self):
        """Perform diagnostic step

        This gets called on every step of the main simulation loop.

        Raises
        ------
        NotImplementedError
            Method or function hasn't been implemented yet. This is an
            abstract base class. Derived classes must implement this
            method in order to be a concrete child class of
            :class:`Diagnostic`.
        """
        raise NotImplementedError

    def initialize(self):
        """Perform any initialization operations

        This gets called once before the main simulation loop. Base class
        definition creates output directory if it does not already exist. If
        subclass overrides this function, call `super().initialize()`
        """
        d = Path(self._input_data["directory"])
        d.mkdir(parents=True, exist_ok=True)

    def finalize(self):
        """Perform any finalization operations

        This gets called once after the main simulation loop is
        complete.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self._input_data})"
