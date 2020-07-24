"""
Several subclasses of the `ComputeTool` class for common scenarios

Included stock subclasses:
    Solver for Poisson's equation (1-D) to find electric potential fields
    EXPLAIN : FINITEDIFFERENCE CLASS
    Particle pusher for magnetic fields using the Boris method
    Interpolate a function given two datasets
"""
import numpy as np
import scipy.interpolate as interpolate
from scipy import sparse

from .core import ComputeTool, Simulation


class PoissonSolver1DRadial(ComputeTool):
    """
    `ComputeTool` that solves an instance of Poisson's Equation in a radial 1-D context

    Parameters
    ----------
    owner : Simulation
        The `Simulation` object that contains this object
    input_data : dict
        Data containing the name of the tool
    
    Attributes
    ----------
    owner : Simulation
        The `Simulation` object that contains this object
    input_data : dict
        Dictionary containing the name of the `ComputeTool` so it can be called by `PhysicsModules`
    field : ndarray
        The list of points over which the computations are done
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.field = None
        
    def initialize(self):
        """
        The `Grid` that provides the `field` attribute may be instantiated after the computetool.
        In order to ensure that the grid exists when it is called to provide the data for `field`,
        the initialize method is called after all the objects in the `Simulation` are created.

        This method pulls the field data from the `Grid`
        """
        self.field = self.owner.grid.generate_field(1)
    
    def solve(self, sources):
        """
        Solves Poisson's Equation

        Parameters
        ----------
        sources : TODO
            TODO: explain what `sources` is supposed to be

        Returns
        -------
        ndarray
            TODO: explain
        """
        r = self.owner.grid.r
        dr = np.mean(self.owner.grid.cell_widths)
        I1 = np.cumsum(r * sources * dr)
        integrand = I1 * dr / r
        i0 = 2 * integrand[1] - integrand[2]   # linearly extrapolate to r = 0
        integrand[0] = i0
        integrand = integrand - i0     # add const of integer so derivative = 0 at r = 0
        I2 = np.cumsum(integrand)
        return I2 - I2[-1]


class FiniteDifference(ComputeTool):
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.dr = self.owner.grid.dr
    
    def setup_ddx(self):
        assert (self.input_data["method"] in ["centered", "upwind_left"])
        if self.input_data["method"] == "centered":
            return self.centered_difference
        if self.input_data["method"] == "upwind_left":
            return self.upwind_left
    
    def centered_difference(self, y):
        d = self.owner.grid.generate_field()
        d[1:-1] = (y[2:] - y[:-2]) / self.dr_centered
        return d
    
    def ddx(self):
        N = self.owner.grid.num_points
        g = 1/(2.0 * self.dr)
        col_below = np.zeros(N) - g
        col_above = np.zeros(N) + g
        D = sparse.dia_matrix( ([col_below, col_above], [-1, 1]), shape=(N, N) )
        return D
        
    
    def upwind_left(self, y):
        d = self.owner.grid.generate_field()
        d[1:] = (y[1:] - y[:-1]) / self.owner.grid.cell_widths
        return d

    def radial_curl(self):
        # FD matrix for (rB)'/r = (1/r)(d/dr)(rB)
        N = self.owner.grid.num_points
        g = 1/(2.0 * self.dr)
        col_below = np.zeros(N)
        col_diag = np.zeros(N)
        col_above = np.zeros(N)
        col_below[:-1] = -g * (self.owner.grid.r[:-1]/self.owner.grid.r[1:])
        col_above[1:] = g * (self.owner.grid.r[1:]/self.owner.grid.r[:-1])
        # set boundary conditions
        # At r=0, use B~linear, and B=0.
        col_above[1] = 2.0 / self.dr     # for col_above, the first element is dropped
        # At r=Rw, use rB~const?
        col_diag[-1] = 1.0 / self.dr     # for col_below, the last element is dropped
        col_below[-2] = 2.0 * col_below[-1]
        # set main columns for finite difference derivative
        D = sparse.dia_matrix( ([col_below, col_diag, col_above], [-1, 0, 1]), shape=(N, N) )
        return D
    
    def del2_radial(self):
        # FD matrix for (1/r)(d/dr)(r (df/dr))
        N = self.owner.grid.num_points
        g1 = 1/(2.0 * self.dr)
        col_below = -g1 * np.ones(N)
        col_above = g1 * np.ones(N)
        
        col_above[1:] = col_above[1:] / self.owner.grid.r[:-1]
        col_below[:-1] = col_below[:-1] / self.owner.grid.r[1:]
        
        # BC at r=0
        col_above[1] = 0
        
        D1 = sparse.dia_matrix(([col_below, col_above], [-1, 1]), shape=(N, N))
        
        g2 = 1/(self.dr**2)
        col_below = g2 * np.ones(N)
        col_diag = g2 * np.ones(N)
        col_above = g2 * np.ones(N)
        
        # BC at r=0, first row of D
        col_above[1] = 2 * col_above[1]
        D2 = sparse.dia_matrix(([col_below, -2*col_diag, col_above], [-1, 0, 1]), shape=(N, N))
        
        # Need to set boundary conditions!
        D = D1 + D2
        return D
    
    def del2(self):
        # FD matrix for d2/dx2
        N = self.owner.grid.num_points
        
        g2 = 1/(self.dr**2)
        col_below = g2 * np.ones(N)
        col_diag = g2 * np.ones(N)
        col_above = g2 * np.ones(N)
        
        # BC at r=0, first row of D
        col_above[1] = 2 * col_above[1]
        D2 = sparse.dia_matrix(([col_below, -2*col_diag, col_above], [-1, 0, 1]), shape=(N, N))

        return D2
        
    
    def ddr(self):
        # FD matrix for (d/dr) f
        N = self.owner.grid.num_points
        g1 = 1/(2.0 * self.dr)
        col_below = -g1 * np.ones(N)
        col_above = g1 * np.ones(N)
        # BC at r=0
        col_above[1] = 0
        D1 = sparse.dia_matrix(([col_below, col_above], [-1, 1]), shape=(N, N))        
        return D1

    def BC_left_extrap(self):
        N = self.owner.grid.num_points
        col_diag = np.ones(N)
        col_above = np.zeros(N)
        col_above2 = np.zeros(N)
        
        # for col_above, the first element is dropped
        col_diag[0] = 0
        col_above[1] = 2
        col_above2[2] = -1

        BC = sparse.dia_matrix(([col_diag, col_above, col_above2], [0,1,2]), shape=(N, N))
        return BC

    def BC_left_avg(self):
        N = self.owner.grid.num_points
        col_diag = np.ones(N)
        col_above = np.zeros(N)
        col_above2 = np.zeros(N)
        
        # for col_above, the first element is dropped
        col_diag[0] = 0
        col_above[1] = 1.5
        col_above2[2] = -0.5

        BC = sparse.dia_matrix(([col_diag, col_above, col_above2], [0,1,2]), shape=(N, N))
        return BC        

    def BC_left_quad(self):
        N = self.owner.grid.num_points
        r = self.owner.grid.r
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
        N = self.owner.grid.num_points
        col_diag = np.ones(N)
        col_above = np.zeros(N)
        col_above2 = np.zeros(N)
        # for col_above, the first element is dropped
        col_diag[0] = 0
        col_above[1] = 1

        BC = sparse.dia_matrix(([col_diag, col_above], [0,1]), shape=(N, N))
        return BC        
    
    def BC_right_extrap(self):
        N = self.owner.grid.num_points
        col_diag = np.ones(N)
        col_below = np.zeros(N)
        col_below2 = np.zeros(N)
        
        # for col_below, the last element is dropped
        col_diag[-1] = 0
        col_below[-2] = 2
        col_below2[-3] = -1

        BC_right = sparse.dia_matrix(([col_below2, col_below, col_diag], [-2, -1, 0]), shape=(N, N))
        return BC_right


class BorisPush(ComputeTool):
    """
    Calculate the motion of a charged particle in a magnetic field

    Parameters
    ----------
    owner : Simulation
        The `Simulation` object that contains this object
    input_data : dict
        Data including the name of the tool

    Attributes
    ----------
    owner : Simulation
        The `Simulation` object that contains this object
    input_data : dict
        Dictionary containing data concerning the object, namely the class name
    c2 : float
        The speed of light squared
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.c2 = 2.9979e8 ** 2

    def push(self, position, momentum, charge, mass, E, B):
        """
        Update the position and momentum of a charged particle in an electromagnetic field

        Parameters
        ----------
        position : ndarray
            The initial position of the particle as a vector
        momentum : ndarray
            The initial momentum of the particle as a vector
        charge : float
            The electric charge of the particle
        mass : float
            The mass of the particle
        E : ndarray
            The value of the electric field around the particle
        B: ndarray
            The value of the magnetic field around the particle
        """
        dt = self.owner.clock.dt

        vminus = momentum + dt * E * charge / 2
        m1 = np.sqrt(mass**2 + np.sum(momentum*momentum, axis=-1)/self.c2)

        t = dt * B * charge / m1[:, np.newaxis] / 2
        s = 2 * t / (1 + np.sum(t*t, axis=-1)[:, np.newaxis])
        
        vprime = vminus + np.cross(vminus, t)
        vplus = vminus + np.cross(vprime, s)
        momentum[:] = vplus + dt * E * charge / 2
        m2 = np.sqrt(mass**2 + np.sum(momentum*momentum, axis=-1)/self.c2)
        position[:] = position + dt * momentum / m2[:, np.newaxis]


class Interpolators(ComputeTool):
    """
    Interpolate a function given two lists of numbers (input and output)

    Parameters
    ----------
    owner : Simulation
        The `Simulation` object that contains this object
    input_data : dict
        Dictionary containing information about the class, i.e. its name

    Attributes
    ----------
    owner : Simulation
        The `Simulation` object that contains this object
    input_data : dict
        Dictionary containing information about the class, i.e. its name
    """
    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)

    def interpolate1D(self, x, y, kind='linear'):
        """
        Given two datasets, return a function relating them

        Parameters
        ----------
        x : list
            List of input values to be interpolated
        y : list
            List of output values to be interpolated
        kind : str
            Order of function being used to relate the two datasets, defaults to "linear"

        Returns
        -------
        f : scipy.interpolate.interpolate.interp1d
            Function relating `x` and `y`
        """
        f = interpolate.interp1d(x, y, kind)
        return f


ComputeTool.register("BorisPush", BorisPush)
ComputeTool.register("PoissonSolver1DRadial", PoissonSolver1DRadial)
ComputeTool.register("FiniteDifference", FiniteDifference)
ComputeTool.register("Interpolators", Interpolators)



