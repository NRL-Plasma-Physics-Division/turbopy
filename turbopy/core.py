"""
Core base classes of the turboPy framework

Notes
-----
The published paper for Turbopy: A lightweight python framework for \
 computational physics can be found in the link below [1]_.


References
----------
.. [1] 1 A.S. Richardson, D.F. Gordon, S.B. Swanekamp, I.M. Rittersdorf, \
P.E. Adamson, O.S. Grannis, G.T. Morgan, A. Ostenfeld, K.L. Phlips, C.G. Sun, \
G. Tang, and D.J. Watkins, Comput. Phys. Commun. 258, 107607 (2021). \
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
            The ``"coordinate_system"`` key selects the grid type:

            **1D grids** (default; ``"coordinate_system"`` absent, or
            ``"cartesian"``, ``"cylindrical"``, ``"spherical"``):

            - ``"N"`` | {``"dr"`` | ``"dx"``} :
                The number of grid points (`int`) | the grid spacing
                (`float`)
            - ``"min"`` | ``"x_min"`` | ``"r_min"`` :
                The coordinate value of the minimum grid point (`float`)
            - ``"max"`` | ``"x_max"`` | ``"r_max"`` :
                The coordinate value of the maximum grid point (`float`)

            **2D Cartesian grid** (``"coordinate_system": "cartesian2d"``):

            - ``"Nx"`` | ``"dx"`` : x-axis point count | spacing
            - ``"x_min"``, ``"x_max"`` : x-axis bounds
            - ``"Ny"`` | ``"dy"`` : y-axis point count | spacing
            - ``"y_min"``, ``"y_max"`` : y-axis bounds

            **2D Cylindrical grid** (``"coordinate_system": "cylindrical2d"``):

            - ``"Nr"`` | ``"dr"`` : radial point count | spacing
            - ``"r_min"``, ``"r_max"`` : radial bounds
            - ``"Nz"`` | ``"dz"`` : axial point count | spacing
            - ``"z_min"``, ``"z_max"`` : axial bounds

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

        self.all_shared_resources = {}

        # set default values for optional
        self.input_data.setdefault('Tools', {})
        self.input_data.setdefault('Diagnostics', {})

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
            m.inspect_resources()
        for m in self.physics_modules:
            m.initialize()

        print("Initializing Diagnostics...")
        for d in self.diagnostics:
            d.inspect_resources()
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
        """Construct the grid based on input parameters.

        Reads the ``"coordinate_system"`` key from the Grid input to
        dispatch to the correct :class:`GridBase` subclass.
        Values ``"cartesian2d"`` and ``"cylindrical2d"`` produce 2D grids.
        All other values (including the default ``"cartesian"``,
        ``"cylindrical"``, and ``"spherical"``) produce the existing
        1D :class:`Grid` for backward compatibility.
        """
        grid_data = self.input_data["Grid"]
        coord_sys = grid_data.get(
            "coordinate_system", "cartesian").lower().strip()
        if coord_sys == "cartesian2d":
            self.grid = Grid2DCartesian(grid_data)
        elif coord_sys == "cylindrical2d":
            self.grid = Grid2DCylindrical(grid_data)
        else:
            self.grid = Grid(grid_data)

    def read_clock_from_input(self):
        """Construct the clock based on input parameters"""
        self.clock = SimulationClock(self, self.input_data["Clock"])

    def read_tools_from_input(self):
        """Construct :class:`ComputeTools` based on input"""
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
        diagnostics, default_params = self.parse_diagnostic_input_dictionary()

        diagnostics = make_values_into_lists(diagnostics)
        default_params.setdefault('directory', 'default_output')

        for diag_type, list_of_diagnostics in diagnostics.items():
            diagnostic_class = Diagnostic.lookup(diag_type)

            file_num = 0
            for params in list_of_diagnostics:
                params['type'] = diag_type
                params = self.combine_dictionaries(default_params, params)
                if "filename" not in params:
                    # Set a default output filename
                    file_end = params.get("output_type", "out")
                    params["filename"] = (f"{diag_type}{file_num}"
                                          f".{file_end}")
                    file_num += 1
                params["filename"] = str(Path(params["directory"])
                                         / Path(params["filename"]))
                self.diagnostics.append(
                    diagnostic_class(owner=self, input_data=params))

    def combine_dictionaries(self, defaults, custom):
        # Values in "custom" dictionary supersede "defaults" because of
        # the order in which they are combined here
        return {**defaults, **custom}

    def parse_diagnostic_input_dictionary(self):
        # The input_data["Diagnostics"] dictionary has two types of keys:
        #    1) keys that are valid diagnostic types
        #    2) other keys, which should be passed along
        #    as "default" parameters
        diagnostics = {k: v for k, v in
                       self.input_data["Diagnostics"].items()
                       if Diagnostic.is_valid_name(k)}
        default_params = {k: v for k, v in
                          self.input_data["Diagnostics"].items()
                          if not Diagnostic.is_valid_name(k)}
        return diagnostics, default_params

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

    def gather_shared_resources(self, shared):
        for k, v in shared.items():
            if k in self.all_shared_resources:
                warnings.warn(f'Shared resource {k} has been overwritten')
            self.all_shared_resources[k] = v


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
    _needed_resources: `dict`
        Dictionary that lists shared resources that this module
        needs. Format is `{shared_key: variable_name}`, where
        `shared_key` is a string with the name of needed resource,
        and `variable_name` is a string to use when saving this
        variable. For example: {"Fields:E": "E"} will make `self.E`.
    _resources_to_share: `dict`
        Dictionary that lists shared resources that this module
        is sharing to others. Format is `{shared_key: variable}`, where
        `shared_key` is a string with the name of resource to share,
        and `variable` is the data to be shared.

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

        # By default, share "public" attributes
        shared = {f'{self.__class__.__name__}_{attribute}': value
                  for attribute, value
                  in self.__dict__.items()
                  if not attribute.startswith('_')}
        self._resources_to_share = shared

        # Items should have key "shared_name", and value is the variable
        # name for the "pointer".
        # For example: {"Fields:E": "E"} will make self.E
        self._needed_resources = {}

    def publish_resource(self, resource: dict):
        """**Deprecated**

        *This method is only here for backwards compatability. New
        code should use the ``_resources_to_share`` dictionary.*

        Method which implements the details of sharing resources
        Parameters
        ----------
        resource : `dict`
            resource dictionary to be shared
        """
        warnings.warn("The resource-sharing API has changed. "
                      "Add to `self._resources_to_share` instead of "
                      "calling `publish_resource`.",
                      DeprecationWarning)
        for k in resource.keys():
            print(f"Module {self.__class__.__name__} is sharing {k}")
        for physics_module in self._owner.physics_modules:
            physics_module.inspect_resource(resource)
        for diagnostic in self._owner.diagnostics:
            diagnostic.inspect_resource(resource)

    def inspect_resource(self, resource: dict):
        """**Deprecated**

        *This method is only here for backwards compatability. New
        code should use the ``_needed_resources`` dictionary.*

        Method for accepting resources shared by other PhysicsModules
        If your subclass needs the data described by the key, now's
        their chance to save a pointer to the data.
        Parameters
        ----------
        resource : `dict`
            resource dictionary to be shared
        """
        pass

    def inspect_resources(self):
        for shared_name, var_name in self._needed_resources.items():
            if shared_name not in self._owner.all_shared_resources:
                warnings.warn(f"Module {self.__class__.__name__} can't find "
                              f"needed resource {shared_name}")
            else:
                self.__dict__[var_name] = self._owner.all_shared_resources[
                                              shared_name
                                          ]

    def exchange_resources(self):
        """Main method for sharing resources with other
        :class:`PhysicsModule` objects.

        This is the function where you call :meth:`publish_resource`,
        to tell other physics modules about data you want to share.

        By default, any "public" attributes (those with names that do
        not start with an underscore) will be shared with the key
        `<class_name>_<attribute_name>`.
        """

        for k in self._resources_to_share.keys():
            print(f"Module {self.__class__.__name__} is sharing {k}")

        self._owner.gather_shared_resources(self._resources_to_share)

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
            self.num_steps = np.int64(np.rint(self.num_steps))

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


class GridBase(ABC):
    """Abstract base class for all turboPy grid types.

    All concrete grid classes inherit from this base and must implement
    :meth:`generate_field` and :meth:`create_interpolator`. The
    ``coordinate_system`` class attribute identifies the grid type.

    Use ``isinstance(grid, GridBase)`` to accept any grid type.
    Use ``isinstance(grid, Grid)`` to identify 1D grids specifically.
    """

    coordinate_system: str = ""

    @abstractmethod
    def generate_field(self, num_components=1,
                       placement_of_points="edge-centered"):
        """Return a zero-filled :class:`numpy.ndarray` shaped for this grid.

        Parameters
        ----------
        num_components : int, optional
            Number of vector components at each point. Default is 1.
        placement_of_points : str, optional
            Designates position of points on the grid.

        Returns
        -------
        :class:`numpy.ndarray`
        """

    @abstractmethod
    def create_interpolator(self, location):
        """Return a callable that interpolates a field to ``location``.

        Parameters
        ----------
        location : float or tuple of float
            The coordinate(s) of the requested point. Scalar for 1D
            grids; a tuple ``(coord1, coord2)`` for 2D grids.

        Returns
        -------
        callable
        """

    def __repr__(self):
        return f"{self.__class__.__name__}({self._input_data})"


class Grid(GridBase):
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
            self.num_points = np.int64(self.num_points)

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
        else:
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
        self.inverse_cell_volumes = 1. / self.cell_volumes

    def set_cylindrical_volumes(self):
        scratch = self.cell_edges ** 2
        self.cell_volumes = np.pi * (scratch[1:] - scratch[:-1])
        self.inverse_cell_volumes = 1. / self.cell_volumes

    def set_spherical_volumes(self):
        scratch = self.cell_edges ** 3
        self.cell_volumes = 4 / 3 * np.pi * (scratch[1:] - scratch[:-1])
        self.inverse_cell_volumes = 1. / self.cell_volumes

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
        self.inverse_interface_volumes[1:-1] = 0.5 * \
            (self.inverse_cell_volumes[1:] + self.inverse_cell_volumes[0:-1])
        self.inverse_interface_volumes[-1] = self.inverse_cell_volumes[-1]


class Grid2DCartesian(GridBase):
    """2D Cartesian grid on (x, y) axes.

    Parameters
    ----------
    input_data : `dict`
        Dictionary containing parameters needed to define the grid.

        The expected parameters are:

        - ``"Nx"`` | ``"dx"`` :
            The number of x-axis edge points (`int`) | the x spacing
            (`float`)
        - ``"Ny"`` | ``"dy"`` :
            The number of y-axis edge points (`int`) | the y spacing
            (`float`)
        - ``"x_min"`` :
            Minimum x coordinate (`float`)
        - ``"x_max"`` :
            Maximum x coordinate (`float`)
        - ``"y_min"`` :
            Minimum y coordinate (`float`)
        - ``"y_max"`` :
            Maximum y coordinate (`float`)

    Attributes
    ----------
    coordinate_system : str
        Always ``"cartesian2d"``.
    x, x_edges : :class:`numpy.ndarray`, shape (Nx,)
        Edge-point coordinates along the x axis.
    y, y_edges : :class:`numpy.ndarray`, shape (Ny,)
        Edge-point coordinates along the y axis.
    x_centers : :class:`numpy.ndarray`, shape (Nx-1,)
        Cell-center coordinates along x.
    y_centers : :class:`numpy.ndarray`, shape (Ny-1,)
        Cell-center coordinates along y.
    x_widths : :class:`numpy.ndarray`, shape (Nx-1,)
        Cell widths along x.
    y_widths : :class:`numpy.ndarray`, shape (Ny-1,)
        Cell widths along y.
    dx, dy : float
        Uniform grid spacings.
    Nx, Ny : int
        Number of edge points along each axis.
    num_points : tuple
        ``(Nx, Ny)``
    shape : tuple
        ``(Nx, Ny)``
    XX, YY : :class:`numpy.ndarray`, shape (Nx, Ny)
        Meshgrid arrays (``indexing='ij'``).
    cell_volumes : :class:`numpy.ndarray`, shape (Nx-1, Ny-1)
        Area of each cell = ``dx_i * dy_j``.
    inverse_cell_volumes : :class:`numpy.ndarray`, shape (Nx-1, Ny-1)
        ``1 / cell_volumes``
    x_min, x_max, y_min, y_max : float
        Domain bounds.
    """

    coordinate_system = "cartesian2d"

    def __init__(self, input_data: dict):
        self._input_data = input_data
        self._parse_grid_data()
        self._set_grid_points()
        self._set_volume_elements()

    def _parse_grid_data(self):
        d = self._input_data
        self.x_min = d["x_min"]
        self.x_max = d["x_max"]
        self.y_min = d["y_min"]
        self.y_max = d["y_max"]

        if "Nx" in d:
            self.Nx = int(d["Nx"])
            self.dx = (self.x_max - self.x_min) / (self.Nx - 1)
        elif "dx" in d:
            self.dx = float(d["dx"])
            npts = 1 + (self.x_max - self.x_min) / self.dx
            if not np.isclose(npts, round(npts)):
                raise RuntimeError("x range / dx does not give an integer "
                                   "number of grid points")
            self.Nx = int(round(npts))
        else:
            raise KeyError("Grid2DCartesian requires 'Nx' or 'dx'")

        if "Ny" in d:
            self.Ny = int(d["Ny"])
            self.dy = (self.y_max - self.y_min) / (self.Ny - 1)
        elif "dy" in d:
            self.dy = float(d["dy"])
            npts = 1 + (self.y_max - self.y_min) / self.dy
            if not np.isclose(npts, round(npts)):
                raise RuntimeError("y range / dy does not give an integer "
                                   "number of grid points")
            self.Ny = int(round(npts))
        else:
            raise KeyError("Grid2DCartesian requires 'Ny' or 'dy'")

        self.num_points = (self.Nx, self.Ny)
        self.shape = self.num_points

    def _set_grid_points(self):
        self.x = np.linspace(self.x_min, self.x_max, self.Nx)
        self.x_edges = self.x
        self.y = np.linspace(self.y_min, self.y_max, self.Ny)
        self.y_edges = self.y
        self.x_centers = 0.5 * (self.x[1:] + self.x[:-1])
        self.y_centers = 0.5 * (self.y[1:] + self.y[:-1])
        self.x_widths = self.x[1:] - self.x[:-1]
        self.y_widths = self.y[1:] - self.y[:-1]
        # Use 'ij' indexing so XX.shape == (Nx, Ny)
        self.XX, self.YY = np.meshgrid(self.x, self.y, indexing='ij')

    def _set_volume_elements(self):
        # cell_volumes[i, j] = dx_i * dy_j
        dx2d = self.x_widths[:, np.newaxis]   # (Nx-1, 1)
        dy2d = self.y_widths[np.newaxis, :]   # (1, Ny-1)
        self.cell_volumes = dx2d * dy2d        # (Nx-1, Ny-1)
        self.inverse_cell_volumes = 1.0 / self.cell_volumes

    def generate_field(self, num_components=1,
                       placement_of_points="edge-centered"):
        """Return a zero-filled :class:`numpy.ndarray` for a field on this grid.

        Parameters
        ----------
        num_components : int, optional
            Number of vector components at each point. Default is 1.
        placement_of_points : str, optional
            One of:

            - ``"edge-centered"`` — shape ``(Nx, Ny)``
            - ``"cell-centered"`` — shape ``(Nx-1, Ny-1)``
            - ``"x-edge-y-cell"`` — shape ``(Nx, Ny-1)``
            - ``"x-cell-y-edge"`` — shape ``(Nx-1, Ny)``

        Returns
        -------
        :class:`numpy.ndarray`
        """
        placement_map = {
            "edge-centered":  (self.Nx,     self.Ny),
            "cell-centered":  (self.Nx - 1, self.Ny - 1),
            "x-edge-y-cell":  (self.Nx,     self.Ny - 1),
            "x-cell-y-edge":  (self.Nx - 1, self.Ny),
        }
        if placement_of_points not in placement_map:
            raise ValueError(f"Unknown placement option: {placement_of_points!r}")
        nx, ny = placement_map[placement_of_points]
        if num_components == 1:
            return np.zeros((nx, ny))
        return np.zeros((nx, ny, num_components))

    def create_interpolator(self, location):
        """Return a bilinear interpolation function for an edge-centered field.

        Parameters
        ----------
        location : tuple of float
            ``(x0, y0)`` coordinates of the requested point.

        Returns
        -------
        callable
            A function ``f(field_2d) -> float`` that bilinearly interpolates
            a 2D edge-centered field (shape ``(Nx, Ny)``) to ``location``.
        """
        x0, y0 = location
        assert x0 >= self.x_min, "Requested x0 is outside the grid"
        assert x0 <= self.x_max, "Requested x0 is outside the grid"
        assert y0 >= self.y_min, "Requested y0 is outside the grid"
        assert y0 <= self.y_max, "Requested y0 is outside the grid"

        ix = int(np.searchsorted(self.x, x0, side='right')) - 1
        iy = int(np.searchsorted(self.y, y0, side='right')) - 1
        ix = min(ix, self.Nx - 2)
        iy = min(iy, self.Ny - 2)

        tx = (x0 - self.x[ix]) / (self.x[ix + 1] - self.x[ix])
        ty = (y0 - self.y[iy]) / (self.y[iy + 1] - self.y[iy])

        def interpolate(field):
            f00 = field[ix,     iy    ]
            f10 = field[ix + 1, iy    ]
            f01 = field[ix,     iy + 1]
            f11 = field[ix + 1, iy + 1]
            return ((1 - tx) * (1 - ty) * f00
                    + tx * (1 - ty) * f10
                    + (1 - tx) * ty * f01
                    + tx * ty * f11)

        return interpolate


class Grid2DCylindrical(GridBase):
    """2D cylindrical grid on (r, z) axes.

    This represents a full 2D ``(r, z)`` domain, distinct from the
    existing 1D ``"cylindrical"`` :class:`Grid` which is a 1D radial
    coordinate with cylindrical volume elements.

    Parameters
    ----------
    input_data : `dict`
        Dictionary containing parameters needed to define the grid.

        The expected parameters are:

        - ``"Nr"`` | ``"dr"`` :
            The number of radial edge points (`int`) | the radial
            spacing (`float`)
        - ``"Nz"`` | ``"dz"`` :
            The number of axial edge points (`int`) | the axial
            spacing (`float`)
        - ``"r_min"`` :
            Minimum radial coordinate (`float`, typically ``0.0``)
        - ``"r_max"`` :
            Maximum radial coordinate (`float`)
        - ``"z_min"`` :
            Minimum axial coordinate (`float`)
        - ``"z_max"`` :
            Maximum axial coordinate (`float`)

    Attributes
    ----------
    coordinate_system : str
        Always ``"cylindrical2d"``.
    r, r_edges : :class:`numpy.ndarray`, shape (Nr,)
        Edge-point radial coordinates.
    z, z_edges : :class:`numpy.ndarray`, shape (Nz,)
        Edge-point axial coordinates.
    r_centers : :class:`numpy.ndarray`, shape (Nr-1,)
        Radial cell centers.
    z_centers : :class:`numpy.ndarray`, shape (Nz-1,)
        Axial cell centers.
    r_widths : :class:`numpy.ndarray`, shape (Nr-1,)
        Cell widths in r.
    z_widths : :class:`numpy.ndarray`, shape (Nz-1,)
        Cell widths in z.
    dr, dz : float
        Uniform grid spacings.
    Nr, Nz : int
        Number of edge points along each axis.
    num_points : tuple
        ``(Nr, Nz)``
    shape : tuple
        ``(Nr, Nz)``
    RR, ZZ : :class:`numpy.ndarray`, shape (Nr, Nz)
        Meshgrid arrays (``indexing='ij'``).
    r_inv : :class:`numpy.ndarray`, shape (Nr,)
        ``1/r`` at each radial edge point; ``0`` where ``r == 0``.
    r_inv_2d : :class:`numpy.ndarray`, shape (Nr, Nz)
        Meshgrid of ``r_inv``, useful for 2D vector operators.
    cell_volumes : :class:`numpy.ndarray`, shape (Nr-1, Nz-1)
        Volume of each annular cell = ``pi * (r_{i+1}^2 - r_i^2) * dz_j``.
    inverse_cell_volumes : :class:`numpy.ndarray`, shape (Nr-1, Nz-1)
        ``1 / cell_volumes``
    r_min, r_max, z_min, z_max : float
        Domain bounds.
    """

    coordinate_system = "cylindrical2d"

    def __init__(self, input_data: dict):
        self._input_data = input_data
        self._parse_grid_data()
        self._set_grid_points()
        self._set_volume_elements()

    def _parse_grid_data(self):
        d = self._input_data
        self.r_min = d["r_min"]
        self.r_max = d["r_max"]
        self.z_min = d["z_min"]
        self.z_max = d["z_max"]

        if "Nr" in d:
            self.Nr = int(d["Nr"])
            self.dr = (self.r_max - self.r_min) / (self.Nr - 1)
        elif "dr" in d:
            self.dr = float(d["dr"])
            npts = 1 + (self.r_max - self.r_min) / self.dr
            if not np.isclose(npts, round(npts)):
                raise RuntimeError("r range / dr does not give an integer "
                                   "number of grid points")
            self.Nr = int(round(npts))
        else:
            raise KeyError("Grid2DCylindrical requires 'Nr' or 'dr'")

        if "Nz" in d:
            self.Nz = int(d["Nz"])
            self.dz = (self.z_max - self.z_min) / (self.Nz - 1)
        elif "dz" in d:
            self.dz = float(d["dz"])
            npts = 1 + (self.z_max - self.z_min) / self.dz
            if not np.isclose(npts, round(npts)):
                raise RuntimeError("z range / dz does not give an integer "
                                   "number of grid points")
            self.Nz = int(round(npts))
        else:
            raise KeyError("Grid2DCylindrical requires 'Nz' or 'dz'")

        self.num_points = (self.Nr, self.Nz)
        self.shape = self.num_points

    def _set_grid_points(self):
        self.r = np.linspace(self.r_min, self.r_max, self.Nr)
        self.r_edges = self.r
        self.z = np.linspace(self.z_min, self.z_max, self.Nz)
        self.z_edges = self.z
        self.r_centers = 0.5 * (self.r[1:] + self.r[:-1])
        self.z_centers = 0.5 * (self.z[1:] + self.z[:-1])
        self.r_widths = self.r[1:] - self.r[:-1]
        self.z_widths = self.z[1:] - self.z[:-1]
        # Use 'ij' indexing so RR.shape == (Nr, Nz)
        self.RR, self.ZZ = np.meshgrid(self.r, self.z, indexing='ij')
        # r_inv: 1/r with 0 where r == 0; errstate suppresses the
        # divide-by-zero warning for the r[0] == 0 case before masking.
        with np.errstate(divide='ignore', invalid='ignore'):
            r_inv_raw = np.where(self.r != 0, 1.0 / self.r, 0.0)
        self.r_inv = r_inv_raw
        self.r_inv_2d, _ = np.meshgrid(self.r_inv, self.z, indexing='ij')

    def _set_volume_elements(self):
        # V[i, j] = pi * (r_{i+1}^2 - r_i^2) * dz_j
        r_sq_diff = np.pi * (self.r[1:] ** 2 - self.r[:-1] ** 2)  # (Nr-1,)
        self.cell_volumes = (r_sq_diff[:, np.newaxis]
                             * self.z_widths[np.newaxis, :])        # (Nr-1, Nz-1)
        self.inverse_cell_volumes = 1.0 / self.cell_volumes

    def generate_field(self, num_components=1,
                       placement_of_points="edge-centered"):
        """Return a zero-filled :class:`numpy.ndarray` for a field on this grid.

        Parameters
        ----------
        num_components : int, optional
            Number of vector components at each point. Default is 1.
        placement_of_points : str, optional
            One of:

            - ``"edge-centered"`` — shape ``(Nr, Nz)``
            - ``"cell-centered"`` — shape ``(Nr-1, Nz-1)``
            - ``"r-edge-z-cell"`` — shape ``(Nr, Nz-1)``
            - ``"r-cell-z-edge"`` — shape ``(Nr-1, Nz)``

        Returns
        -------
        :class:`numpy.ndarray`
        """
        placement_map = {
            "edge-centered":  (self.Nr,     self.Nz),
            "cell-centered":  (self.Nr - 1, self.Nz - 1),
            "r-edge-z-cell":  (self.Nr,     self.Nz - 1),
            "r-cell-z-edge":  (self.Nr - 1, self.Nz),
        }
        if placement_of_points not in placement_map:
            raise ValueError(f"Unknown placement option: {placement_of_points!r}")
        nr, nz = placement_map[placement_of_points]
        if num_components == 1:
            return np.zeros((nr, nz))
        return np.zeros((nr, nz, num_components))

    def create_interpolator(self, location):
        """Return a bilinear interpolation function for an edge-centered field.

        Parameters
        ----------
        location : tuple of float
            ``(r0, z0)`` coordinates of the requested point.

        Returns
        -------
        callable
            A function ``f(field_2d) -> float`` that bilinearly interpolates
            a 2D edge-centered field (shape ``(Nr, Nz)``) to ``location``.
        """
        r0, z0 = location
        assert r0 >= self.r_min, "Requested r0 is outside the grid"
        assert r0 <= self.r_max, "Requested r0 is outside the grid"
        assert z0 >= self.z_min, "Requested z0 is outside the grid"
        assert z0 <= self.z_max, "Requested z0 is outside the grid"

        ir = int(np.searchsorted(self.r, r0, side='right')) - 1
        iz = int(np.searchsorted(self.z, z0, side='right')) - 1
        ir = min(ir, self.Nr - 2)
        iz = min(iz, self.Nz - 2)

        tr = (r0 - self.r[ir]) / (self.r[ir + 1] - self.r[ir])
        tz = (z0 - self.z[iz]) / (self.z[iz + 1] - self.z[iz])

        def interpolate(field):
            f00 = field[ir,     iz    ]
            f10 = field[ir + 1, iz    ]
            f01 = field[ir,     iz + 1]
            f11 = field[ir + 1, iz + 1]
            return ((1 - tr) * (1 - tz) * f00
                    + tr * (1 - tz) * f10
                    + (1 - tr) * tz * f01
                    + tr * tz * f11)

        return interpolate


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
    _needed_resources: `dict`
        Dictionary that lists shared resources that this module
        needs. Format is `{shared_key: variable_name}`, where
        `shared_key` is a string with the name of needed resource,
        and `variable_name` is a string to use when saving this
        variable. For example: {"Fields:E": "E"} will make `self.E`.
    """

    _factory_type_name = "Diagnostic"
    _registry = {}

    def __init__(self, owner: Simulation, input_data: dict):
        self._owner = owner
        self._input_data = input_data

        # Items should have key "shared_name", and value is the variable
        # name for the "pointer"
        # For example: {"Fields:E": "E"} will make self.E
        self._needed_resources = {}

    def inspect_resource(self, resource: dict):
        """**Deprecated**

        *This method is only here for backwards compatability. New
        code should use the ``_needed_resources`` dictionary.*

        Save references to data from other PhysicsModules
        If your subclass needs the data described by the key, now's
        their chance to save a reference to the data
        Parameters
        ----------
        resource: `dict`
            A dictionary containing references to data shared by other
            PhysicsModules.
        """
        pass

    def inspect_resources(self):
        for shared_name, var_name in self._needed_resources.items():
            if shared_name not in self._owner.all_shared_resources:
                warnings.warn(f"Diagnostic {self.__class__.__name__} can't "
                              f"find needed resource {shared_name}")
            else:
                self.__dict__[var_name] = self._owner.all_shared_resources[
                                              shared_name
                                          ]

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


def wrap_item_in_list(item):
    if type(item) is list:
        return item
    else:
        return [item]


def make_values_into_lists(dictionary):
    return {k: wrap_item_in_list(v) for k, v in dictionary.items()}
