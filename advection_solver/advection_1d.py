"""1D advection operators.
2D/3D schemes can be built from 1D operators via dimension-splitting
"""

import numpy as np
from numba import jit

from . util import roll  # np.roll doesn't work with jit(nopython=True)
# roll is convenient for implementing periodic boundary condition
# An alternative is to use np.take(mode='wrap') instead of roll


@jit(nopython=True)
def upwind_tendency(c, u, dx, dt):
    '''
    Upwind tendency with periodic boundary

    Args:
      c: 1d numpy array, density field
      u: 1d numpy array, wind field
      dx: float, grid spacing (assume uniform)
      dt: float, time step

    Returns:
      1d numpy array with same shape as `c`
    '''

    nx = c.size

    flux = c*u
    flux_l = roll(flux, 1)  # so f_l[j] == f[j-1], with periodic boundary
    flux_r = roll(flux, -1)

    tendency = np.empty(nx)

    for i in range(nx):
        if u[i] > 0:
            tendency[i] = -flux[i] + flux_l[i]
        else:
            tendency[i] = -flux_r[i] + flux[i]

    return tendency*dt/dx


@jit(nopython=True)
def vanleer_tendency(c, u, dx, dt, limiter=True):
    """
    Second-order flux-limited (VanLeer) tendency with periodic boundary

    Args:
      c: 1d numpy array, density field
      u: 1d numpy array, wind field
      dx: float, grid spacing (assume uniform)
      dt: float, time step

    Returns:
      1d numpy array with same shape as `c`

    Reference: The "mono-5" limiter in
    Lin, S.-J., et al. (1994). "A class of the van Leer-type transport schemes and its application to the moisture transport in a general circulation model."
    """

    nx = c.size

    u = 0.5*(u+roll(u, 1))  # re-stagger to C-grid, pointing from box[i-1] to box[i]
    cfl = u*dt/dx

    c_l = roll(c, 1)  # so c_l[i] == c[i-1], with periodic boundary
    c_r = roll(c, -1)

    mismatch = np.empty(nx)  # left-right difference in piecewise-linear reconstruction
    flux = np.empty(nx)  # flux[i] is the flux from box[i-1] to box[i]

    delta = c - c_l
    delta_avg = (delta + roll(delta, -1))/2

    # compute slope (mismatch) in piecewise-linear reconstruction
    if limiter:
        for i in range(nx):
            # compute local limiter
            c_max = max(c[i], c_l[i], c_r[i])  # upper boundary
            c_min = min(c[i], c_l[i], c_r[i])  # lower boundary
            mismatch[i] = np.sign(delta_avg[i])*min(abs(delta_avg[i]),
                                                    2*(c[i]-c_min),
                                                    2*(c_max-c[i]))
    else:
        mismatch = delta_avg  # just take original slope

    # compute flux from slope
    mismatch_l = roll(mismatch, 1)
    for i in range(nx):
        if u[i] > 0:
            flux[i] = u[i]*(c_l[i]+mismatch_l[i]/2*(1-cfl[i]))
        else:
            flux[i] = u[i]*(c[i]-mismatch[i]/2*(1+cfl[i]))

    tendency = flux - roll(flux, -1)
    return tendency*dt/dx
