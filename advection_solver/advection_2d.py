import numpy as np

from numba import jit
from . operator import upwind_tendency, vanleer_tendency


# Apply 1D operator in x, y dimensions separately, and then add up tendencies
# TODO: make those helper functions less verbose


@jit(nopython=True)
def _tend_x_inner(c, u, dx, dt):
    ny, nx = c.shape
    tend = np.empty((ny, nx))
    for i in range(ny):
        tend[i, :] = upwind_tendency(c[i, :], u[i, :], dx, dt)
    return tend


@jit(nopython=True)
def _tend_y_inner(c, v, dy, dt):
    ny, nx = c.shape
    tend = np.empty((ny, nx))
    for j in range(nx):
        tend[:, j] = upwind_tendency(c[:, j], v[:, j], dy, dt)
    return tend


@jit(nopython=True)
def _tend_x_outer(c, u, dx, dt):
    ny, nx = c.shape
    tend = np.empty((ny, nx))
    for i in range(ny):
        tend[i, :] = vanleer_tendency(c[i, :], u[i, :], dx, dt)
    return tend


@jit(nopython=True)
def _tend_y_outer(c, v, dy, dt):
    ny, nx = c.shape
    tend = np.empty((ny, nx))
    for j in range(nx):
        tend[:, j] = vanleer_tendency(c[:, j], v[:, j], dy, dt)
    return tend


@jit(nopython=True)
def tendency_2d_vanleer(c, u, v, dx, dy, dt):
    '''
    2D advection tendency with periodic boundary
    Use second-order (VanLeer) scheme for outer operator and upwind for inner operator

    Args:
      c: 2d numpy array, density field
      u: 2d numpy array, wind field in x direction
      v: 2d numpy array, wind field in y direction
      dx: float, grid spacing (assume uniform)
      dy: float, grid spacing (assume uniform, but can be different from dx)
      dt: float, time step

    Returns:
      2d numpy array with same shape as `c`
    '''
    ny, nx = c.shape

    # operator splitting in x and y directions
    tendency = (_tend_x_outer(0.5*_tend_y_inner(c, v, dy, dt) + c, u, dx, dt) +
                _tend_y_outer(0.5*_tend_x_inner(c, u, dx, dt) + c, v, dy, dt)
                )

    return tendency

@jit(nopython=True)
def tendency_2d_upwind(c, u, v, dx, dy, dt):
    '''
    2D advection tendency with periodic boundary
    Use upwind scheme for both outer operator and inner operator

    Args:
      c: 2d numpy array, density field
      u: 2d numpy array, wind field in x direction
      v: 2d numpy array, wind field in y direction
      dx: float, grid spacing (assume uniform)
      dy: float, grid spacing (assume uniform, but can be different from dx)
      dt: float, time step

    Returns:
      2d numpy array with same shape as `c`
    '''
    ny, nx = c.shape

    # operator splitting in x and y directions
    tendency = (_tend_x_inner(0.5*_tend_y_inner(c, v, dy, dt) + c, u, dx, dt) +
                _tend_y_inner(0.5*_tend_x_inner(c, u, dx, dt) + c, v, dy, dt)
                )
    return tendency
