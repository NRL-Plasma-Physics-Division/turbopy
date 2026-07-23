Diagnostics
===========

A :class:`turbopy.core.Diagnostic` observes a running simulation and
optionally writes output to disk.  Diagnostics are called every step
from :meth:`turbopy.core.Simulation.fundamental_cycle` (before physics
modules update) and once more from
:meth:`~turbopy.core.Simulation.finalize_simulation` at the end of the
run.

Lifecycle
---------

1. ``__init__(owner, input_data)``  — populate
   :attr:`_needed_resources` and read config from ``input_data``.
2. :meth:`~turbopy.core.Diagnostic.inspect_resources` — binds shared
   resources named in :attr:`_needed_resources` to attributes.
3. :meth:`~turbopy.core.Diagnostic.initialize` — one-time setup.  The
   base class creates the output directory listed in
   ``input_data["directory"]``.  Subclasses that override should call
   ``super().initialize()``.
4. :meth:`~turbopy.core.Diagnostic.diagnose` — called on every step of
   the main loop.  Subclasses **must** override.
5. :meth:`~turbopy.core.Diagnostic.finalize` — called once from
   :meth:`~turbopy.core.Simulation.finalize_simulation`; typically
   flushes buffered output to disk.

Default filenames and directories
---------------------------------

When :meth:`~turbopy.core.Simulation.read_diagnostics_from_input`
processes the ``"Diagnostics"`` dict, entries whose key is *not* a
registered diagnostic name are treated as default parameters and
merged into every constructed diagnostic's ``input_data``.  When no
explicit ``"filename"`` is supplied, the default is
``"{diag_type}{file_num}.{output_type}"`` (with ``file_num``
incrementing per instance of the same type).  When no
``"directory"`` is supplied, the default is ``"default_output"``.

Built-in diagnostics
--------------------

:class:`turbopy.diagnostics.PointDiagnostic`
   Sample a field at a single point (an
   interpolator is built from
   :meth:`~turbopy.core.GridBase.create_interpolator`) and route the
   value to an :class:`~turbopy.diagnostics.OutputUtility`
   (``"stdout"``, ``"csv"``, or ``"npy"``).  ``"write_interval"``
   controls how often the buffer is flushed to disk during the run.

:class:`turbopy.diagnostics.FieldDiagnostic`
   Sample a whole field on every step (or every ``"dump_interval"``
   seconds) and route it to an
   :class:`~turbopy.diagnostics.OutputUtility`.  Distinct from
   ``"write_interval"``, which controls flush cadence to disk.

:class:`turbopy.diagnostics.GridDiagnostic`
   Write the grid coordinates once, at
   :meth:`~turbopy.core.Diagnostic.initialize` time.  For 2D grids
   both axes are written; for 1D grids the radial coordinate array is
   written.

:class:`turbopy.diagnostics.ClockDiagnostic`
   Write the simulation time on every step.

:class:`turbopy.diagnostics.HistoryDiagnostic`
   Aggregate multiple traces into an
   :mod:`xarray` dataset and write a single NetCDF file.  Supports 1D
   grids and both 2D grid types
   (:class:`~turbopy.core.Grid2DCartesian` and
   :class:`~turbopy.core.Grid2DCylindrical`).  On 2D grids the
   resulting NetCDF dataset carries spatial coordinates in addition to
   ``time``: ``x`` and ``y`` for Cartesian grids, or ``r`` and ``z``
   for cylindrical grids.

Output utility helpers
----------------------

:class:`turbopy.diagnostics.OutputUtility` is an abstract helper you
can compose into your own diagnostics.  The built-in implementations
are:

* :class:`~turbopy.diagnostics.PrintOutputUtility` — writes to
  ``stdout``.
* :class:`~turbopy.diagnostics.CSVOutputUtility` — writes CSV.
* :class:`~turbopy.diagnostics.NPYOutputUtility` — writes NumPy
  ``.npy``.

The ``output_type`` string in the input dict maps to one of these:
``"stdout"``, ``"csv"``, ``"npy"``.

Interval handler
----------------

:class:`turbopy.diagnostics.IntervalHandler` calls a bound function
when a specified interval of simulation time has elapsed.  Used
internally by :class:`~turbopy.diagnostics.PointDiagnostic`,
:class:`~turbopy.diagnostics.FieldDiagnostic`, and
:class:`~turbopy.diagnostics.ClockDiagnostic` to control how often
buffers are flushed to disk while the simulation is running.

Example input
-------------

::

    {
        "Diagnostics": {
            "directory": "output/",
            "grid":  {"filename": "grid.csv"},
            "clock": {"filename": "time.csv"},
            "field": {
                "field": "EMField_E",
                "component": 0,
                "output_type": "csv",
                "dump_interval": 0.01,
                "write_interval": 0.1,
            },
        }
    }
