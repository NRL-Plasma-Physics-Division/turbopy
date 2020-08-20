"""
Several subclasses of the :class:`turbopy.core.ComputeTool` class for
common scenarios

Included stock subclasses:

- Solver for the 1D radial Poisson's equation
- Helper functions for constructing sparse finite difference matrices
- Charged particle pusher using the Boris method
- Interpolate a function y(x) given y on a grid in x

"""
import numpy as np
import scipy.interpolate as interpolate
from scipy import sparse

from .core import ComputeTool, Simulation


class PoissonSolver1DRadial(ComputeTool):
    """
    Solve 1D radial Poisson's Equation, using finite difference methods

    Parameters
    ----------
    owner : Simulation
        The :class:`turbopy.core.Simulation` object that contains this
        object
    input_data : dict
        There are no custom configuration options for this tool
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
    
    def solve(self, sources):
        """
        Solves Poisson's Equation

        Parameters
        ----------
        sources : :class:`numpy.ndarray`
            Vector containing source terms for the Poisson equation

        Returns
        -------
        :class:`numpy.ndarray`
            Vector containing the finite difference solution
        """
        r = self._owner.grid.r
        dr = np.mean(self._owner.grid.cell_widths)
        I1 = np.cumsum(r * sources * dr)
        integrand = I1 * dr / r
        # linearly extrapolate to r = 0
        i0 = 2 * integrand[1] - integrand[2]
        integrand[0] = i0
        # add constant of integration so derivative = 0 at r = 0
        integrand = integrand - i0
        I2 = np.cumsum(integrand)
        return I2 - I2[-1]


class FiniteDifference(ComputeTool):
    """Helper functions for constructing finite difference matrices

    This class contains functions for constructing finite difference
    approximations to various differential operators. The
    :mod:`scipy.sparse` package from :mod:`scipy` is used since most of
    these are tridiagonal sparse matrices.

    Parameters
    ----------
    owner : Simulation
        The :class:`turbopy.core.Simulation` object that contains this
        object
    input_data : dict
        Dictionary of configuration options.
        The expected parameters are:

        - ``"method"`` | {``"centered"`` | ``"upwind_left"``} :
            Select between centered difference, and left upwind
            difference for the `setup_ddx` member function.
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.dr = self._owner.grid.dr
    
    def setup_ddx(self):
        """Select between centered and upwind finite difference

        Returns
        -------
        function
            Returns a reference to either :meth:`centered_difference` or
            :meth:`upwind_left`, based on the configuration option
            :attr:`input_data["method"]`
        """
        assert (self._input_data["method"] in
                ["centered", "upwind_left"])
        if self._input_data["method"] == "centered":
            return self.centered_difference
        if self._input_data["method"] == "upwind_left":
            return self.upwind_left
    
    def centered_difference(self, y):
        """Centered finite difference estimate for dy/dx

        Parameters
        ----------
        y : :class:`numpy.ndarray`
            Vector of values on the grid

        Returns
        -------
        :class:`numpy.ndarray`
            Estimate of the derivative dy/dx constructed using the
            centered finite difference method
        """
        d = self._owner.grid.generate_field()
        d[1:-1] = (y[2:] - y[:-2]) / (2 * self.dr)
        return d

    def upwind_left(self, y):
        """Left upwind finite difference estimate for dy/dx

        Parameters
        ----------
        y : :class:`numpy.ndarray`
            Vector of values on the grid

        Returns
        -------
        :class:`numpy.ndarray`
            Estimate of the derivative dy/dx constructed using the
            left upwind finite difference method
        """
        d = self._owner.grid.generate_field()
        d[1:] = (y[1:] - y[:-1]) / self._owner.grid.cell_widths
        return d

    def ddx(self):
        """Finite difference matrix for df/dx (centered)

        Returns
        -------
        :class:`scipy.sparse.dia_matrix`
            Matrix which implements the centered finite difference
            approximation to df/dx
        """
        N = self._owner.grid.num_points
        g = 1/(2.0 * self.dr)
        col_below = np.zeros(N) - g
        col_above = np.zeros(N) + g
        D = sparse.dia_matrix(([col_below, col_above], [-1, 1]),
                              shape=(N, N))
        return D

    def radial_curl(self):
        """Finite difference matrix for (rf)'/r = (1/r)(d/dr)(rf)

        Returns
        -------
        :class:`scipy.sparse.dia_matrix`
            Matrix which implements a finite difference approximation
            to (rf)'/r = (1/r)(d/dr)(rf)
        """
        N = self._owner.grid.num_points
        g = 1/(2.0 * self.dr)
        col_below = np.zeros(N)
        col_diag = np.zeros(N)
        col_above = np.zeros(N)
        col_below[:-1] = -g * (self._owner.grid.r[:-1]
                               / self._owner.grid.r[1:])
        col_above[1:] = g * (self._owner.grid.r[1:]
                             / self._owner.grid.r[:-1])

        # Set boundary conditions
        # At r=0, use B~linear, and B=0.
        # for col_above, the first element is dropped
        col_above[1] = 2.0 / self.dr
        # At r=Rw, use rB~const?
        # for col_below, the last element is dropped
        col_diag[-1] = 1.0 / self.dr
        col_below[-2] = 2.0 * col_below[-1]
        # set main columns for finite difference derivative
        D = sparse.dia_matrix(([col_below, col_diag, col_above],
                               [-1, 0, 1]), shape=(N, N))
        return D
    
    def del2_radial(self):
        """Finite difference matrix for (1/r)(d/dr)(r (df/dr))

        Returns
        -------
        :class:`scipy.sparse.dia_matrix`
            Matrix which implements a finite difference approximation
            to (1/r)(d/dr)(r (df/dr))"""
        N = self._owner.grid.num_points
        g1 = 1/(2.0 * self.dr)
        col_below = -g1 * np.ones(N)
        col_above = g1 * np.ones(N)
        
        col_above[1:] = col_above[1:] / self._owner.grid.r[:-1]
        col_below[:-1] = col_below[:-1] / self._owner.grid.r[1:]
        
        # BC at r=0
        col_above[1] = 0
        
        D1 = sparse.dia_matrix(([col_below, col_above], [-1, 1]),
                               shape=(N, N))
        
        g2 = 1/(self.dr**2)
        col_below = g2 * np.ones(N)
        col_diag = g2 * np.ones(N)
        col_above = g2 * np.ones(N)
        
        # BC at r=0, first row of D
        col_above[1] = 2 * col_above[1]
        D2 = sparse.dia_matrix(([col_below, -2*col_diag, col_above],
                                [-1, 0, 1]), shape=(N, N))
        
        # Need to set boundary conditions!
        D = D1 + D2
        return D
    
    def del2(self):
        """Finite difference matrix for d2/dx2

        Returns
        -------
        :class:`scipy.sparse.dia_matrix`
            Matrix which implements a finite difference approximation
            to (d/dx)(df/dx)"""
        N = self._owner.grid.num_points
        
        g2 = 1/(self.dr**2)
        col_below = g2 * np.ones(N)
        col_diag = g2 * np.ones(N)
        col_above = g2 * np.ones(N)
        
        # BC at r=0, first row of D
        col_above[1] = 2 * col_above[1]
        D2 = sparse.dia_matrix(([col_below, -2*col_diag, col_above],
                                [-1, 0, 1]), shape=(N, N))
        return D2

    def ddr(self):
        """Finite difference matrix for (d/dr) f

        Returns
        -------
        :class:`scipy.sparse.dia_matrix`
            Matrix which implements a finite difference approximation
            to df/dr
        """
        N = self._owner.grid.num_points
        g1 = 1/(2.0 * self.dr)
        col_below = -g1 * np.ones(N)
        col_above = g1 * np.ones(N)
        # BC at r=0
        col_above[1] = 0
        D1 = sparse.dia_matrix(([col_below, col_above], [-1, 1]),
                               shape=(N, N))
        return D1

    def BC_left_extrap(self):
        """Sparse matrix to extrapolate solution at left boundary

        Returns
        -------
        :class:`scipy.sparse.dia_matrix`
            Matrix which implements a boundary condition for the left
            boundary such that the solution at the first two internal
            grid points is extrapolated to the boundary point.
        """
        N = self._owner.grid.num_points
        col_diag = np.ones(N)
        col_above = np.zeros(N)
        col_above2 = np.zeros(N)
        
        # for col_above, the first element is dropped
        col_diag[0] = 0
        col_above[1] = 2
        col_above2[2] = -1

        BC = sparse.dia_matrix(([col_diag, col_above, col_above2],
                                [0, 1, 2]), shape=(N, N))
        return BC

    def BC_left_avg(self):
        """Sparse matrix to set average solution at left boundary

        Returns
        -------
        :class:`scipy.sparse.dia_matrix`
            Matrix which implements a boundary condition for the left
            boundary.
        """
        N = self._owner.grid.num_points
        col_diag = np.ones(N)
        col_above = np.zeros(N)
        col_above2 = np.zeros(N)
        
        # for col_above, the first element is dropped
        col_diag[0] = 0
        col_above[1] = 1.5
        col_above2[2] = -0.5

        BC = sparse.dia_matrix(([col_diag, col_above, col_above2],
                                [0, 1, 2]), shape=(N, N))
        return BC        

    def BC_left_quad(self):
        """Sparse matrix for quadratic extrapolation at left boundary

        Returns
        -------
        :class:`scipy.sparse.dia_matrix`
            Matrix which implements a boundary condition for the left
            boundary such that the solution at the first two internal
            grid points is extrapolated to the boundary point.
        """
        N = self._owner.grid.num_points
        r = self._owner.grid.r
        col_diag = np.ones(N)
        col_above = np.zeros(N)
        col_above2 = np.zeros(N)
        
        R2 = (r[1]**2 + r[2]**2)/(r[2]**2 - r[1]**2)/2
        # for col_above, the first element is dropped
        col_diag[0] = 0
        col_above[1] = 0.5 + R2
        col_above2[2] = 0.5 - R2

        BC = sparse.dia_matrix(([col_diag, col_above, col_above2],
                                [0, 1, 2]), shape=(N, N))
        return BC
    
    def BC_left_flat(self):
        """Sparse matrix to set Neumann condition at left boundary

        Returns
        -------
        :class:`scipy.sparse.dia_matrix`
            Matrix which implements a boundary condition for the left
            boundary such that the derivative of the solution is zero
            at the boundary.
        """
        N = self._owner.grid.num_points
        col_diag = np.ones(N)
        col_above = np.zeros(N)
        # for col_above, the first element is dropped
        col_diag[0] = 0
        col_above[1] = 1

        BC = sparse.dia_matrix(([col_diag, col_above], [0, 1]),
                               shape=(N, N))
        return BC        
    
    def BC_right_extrap(self):
        """Sparse matrix to extrapolate solution at right boundary

        Returns
        -------
        :class:`scipy.sparse.dia_matrix`
            Matrix which implements a boundary condition for the right
            boundary such that the solution at the first two internal
            grid points is extrapolated to the boundary point.
        """
        N = self._owner.grid.num_points
        col_diag = np.ones(N)
        col_below = np.zeros(N)
        col_below2 = np.zeros(N)
        
        # for col_below, the last element is dropped
        col_diag[-1] = 0
        col_below[-2] = 2
        col_below2[-3] = -1

        BC_right = sparse.dia_matrix(([col_below2, col_below, col_diag],
                                      [-2, -1, 0]), shape=(N, N))
        return BC_right


class BorisPush(ComputeTool):
    """
    Calculate charged particle motion in electric and magnetic fields

    This is an implementation of the Boris push algorithm.

    Parameters
    ----------
    owner : Simulation
        The :class:`turbopy.core.Simulation` object that contains this
        object
    input_data : dict
        There are no custom configuration options for this tool

    Attributes
    ----------
    c2 : float
        The speed of light squared
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.c2 = 2.9979e8 ** 2

    def push(self, position, momentum, charge, mass, E, B):
        """
        Update the position and momentum of a charged particle in an
        electromagnetic field

        Parameters
        ----------
        position : :class:`numpy.ndarray`
            The initial position of the particle as a vector
        momentum : :class:`numpy.ndarray`
            The initial momentum of the particle as a vector
        charge : float
            The electric charge of the particle
        mass : float
            The mass of the particle
        E : :class:`numpy.ndarray`
            The value of the electric field at the particle
        B: :class:`numpy.ndarray`
            The value of the magnetic field at the particle
        """
        dt = self._owner.clock.dt

        vminus = momentum + dt * E * charge / 2
        m1 = np.sqrt(mass**2 + np.sum(momentum*momentum, axis=-1)
                     / self.c2)

        t = dt * B * charge / m1[:, np.newaxis] / 2
        s = 2 * t / (1 + np.sum(t*t, axis=-1)[:, np.newaxis])
        
        vprime = vminus + np.cross(vminus, t)
        vplus = vminus + np.cross(vprime, s)
        momentum[:] = vplus + dt * E * charge / 2
        m2 = np.sqrt(mass**2 + np.sum(momentum*momentum, axis=-1)
                     / self.c2)
        position[:] = position + dt * momentum / m2[:, np.newaxis]


class Interpolators(ComputeTool):
    """
    Interpolate a function y(x) given y at grid points in x

    Parameters
    ----------
    owner : Simulation
        The :class:`turbopy.core.Simulation` object that contains this
        object
    input_data : dict
        There are no custom configuration options for this tool
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)

    def interpolate1D(self, x, y, kind='linear'):
        """
        Given two datasets, return an interpolating function

        Parameters
        ----------
        x : list
            List of input values to be interpolated
        y : list
            List of output values to be interpolated
        kind : str
            Order of function being used to relate the two datasets,
            defaults to "linear". Passed as a parameter to
            scipy.interpolate.interpolate.interp1d.

        Returns
        -------
        f : scipy.interpolate.interpolate.interp1d
            Function which interpolates y(x) given grid `x` and
            values `y` on the grid.
        """
        f = interpolate.interp1d(x, y, kind)
        return f


ComputeTool.register("BorisPush", BorisPush)
ComputeTool.register("PoissonSolver1DRadial", PoissonSolver1DRadial)
ComputeTool.register("FiniteDifference", FiniteDifference)
ComputeTool.register("Interpolators", Interpolators)
