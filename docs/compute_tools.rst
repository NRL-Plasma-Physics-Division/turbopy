Compute Tools
=============

A :class:`turbopy.core.ComputeTool` is a numerical kernel — a Poisson
solver, a finite difference operator, a particle pusher — that can be
shared between :class:`turbopy.core.PhysicsModule` instances.  Tools are
not called by the main loop; instead, modules hold references to them
and call their methods as needed.

When to use a compute tool
--------------------------

Prefer a :class:`~turbopy.core.ComputeTool` over a
:class:`~turbopy.core.PhysicsModule` when the code:

* Implements a numerical method rather than a physical dynamic.
* Might be reused by more than one physics module.
* Has no need to advance state on the main-loop clock.

Anatomy of a tool
-----------------

::

    from turbopy import ComputeTool

    class MySolver(ComputeTool):
        def __init__(self, owner, input_data):
            super().__init__(owner, input_data)
            self.tolerance = input_data.get("tolerance", 1.0e-6)

        def initialize(self):
            # called once by Simulation.prepare_simulation
            self.matrix = build_matrix(self._owner.grid)

        def solve(self, rhs):
            return spsolve(self.matrix, rhs)

    ComputeTool.register("MySolver", MySolver)

Retrieving a tool from a physics module
---------------------------------------

Use :meth:`turbopy.core.Simulation.find_tool_by_name` from inside a
module::

    class MyPhysics(PhysicsModule):
        def initialize(self):
            self.solver = self._owner.find_tool_by_name("MySolver")

If more than one instance of the same tool type is configured in
``"Tools"``, give each one a ``"custom_name"`` in the input dict and
pass that as the second argument to
:meth:`~turbopy.core.Simulation.find_tool_by_name`.

Note that :meth:`~turbopy.core.Simulation.find_tool_by_name` returns
``None`` when zero or more than one match is found.

Built-in compute tools
----------------------

:class:`turbopy.computetools.PoissonSolver1DRadial`
   Solves the 1D radial Poisson equation on a 1D
   :class:`~turbopy.core.Grid` using finite differences.

:class:`turbopy.computetools.FiniteDifference`
   Builds first- and second-derivative sparse matrix operators for a 1D
   :class:`~turbopy.core.Grid`.  Raises :class:`TypeError` in
   ``__init__`` if the simulation grid is not a 1D grid — use
   :class:`~turbopy.computetools.FiniteDifference2D` for 2D grids.  The
   ``"method"`` input key selects between ``"centered"`` and
   ``"upwind_left"`` for :meth:`~turbopy.computetools.FiniteDifference.setup_ddx`.

:class:`turbopy.computetools.FiniteDifference2D`
   Finite difference operators for 2D grids
   (:class:`~turbopy.core.Grid2DCartesian` or
   :class:`~turbopy.core.Grid2DCylindrical`).  Fields must be flattened
   in row-major (C) order before being multiplied by these matrices;
   the result can be ``.reshape((N1, N2))``'d back to 2D.  Operators
   are built via Kronecker products of 1D building blocks.  Raises
   :class:`TypeError` in ``__init__`` if the grid is 1D.  Provides
   ``ddx``, ``ddy``, ``del2_x``, ``del2_y`` for
   :class:`~turbopy.core.Grid2DCartesian`, and ``ddr``, ``ddz``,
   ``del2_r``, ``del2_z`` for
   :class:`~turbopy.core.Grid2DCylindrical`.  The
   :meth:`~turbopy.computetools.FiniteDifference2D.laplacian` method
   dispatches on grid type: it returns ``del2_x + del2_y`` for
   Cartesian grids and ``del2_r + del2_z`` (with the ``(1/r) d/dr``
   term folded into ``del2_r``) for cylindrical grids.

:class:`turbopy.computetools.PoissonSolver2D`
   Solves the 2D Poisson equation on either a
   :class:`~turbopy.core.Grid2DCartesian` or a
   :class:`~turbopy.core.Grid2DCylindrical` grid, imposing homogeneous
   Dirichlet (``φ = 0``) conditions on all four boundaries.  Uses
   :meth:`~turbopy.computetools.FiniteDifference2D.laplacian`
   internally to assemble the sparse system, then calls
   :func:`scipy.sparse.linalg.spsolve`.  The ``solve(source)`` method
   expects ``source`` to be a 2D array whose shape matches
   ``grid.shape``.  Raises :class:`TypeError` if the simulation grid
   is a 1D :class:`~turbopy.core.Grid`.

:class:`turbopy.computetools.BorisPush`
   Boris particle pusher.  ``push`` updates the particle position and
   momentum arrays *in place*.

:class:`turbopy.computetools.Interpolators`
   Thin wrapper over :func:`scipy.interpolate.interp1d`.  The ``kind``
   argument is passed straight through.

Configuring tools from input
----------------------------

::

    {
        "Tools": {
            "FiniteDifference": {"method": "centered"},
            "MySolver": [
                {"custom_name": "coarse", "tolerance": 1e-4},
                {"custom_name": "fine",   "tolerance": 1e-8},
            ],
        },
    }

A tool value that is a list of dicts creates one tool per entry, all of
the same class but with independent configuration.
