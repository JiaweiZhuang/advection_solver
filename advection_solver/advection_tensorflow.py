# Assume Eager mode.  Not tested with graph mode!
import tensorflow as tf


def upwind_tendency_1d(c, u, dx, dt, dim=0):
    '''
    Tensorflow vectorized version of advection tendency with periodic boundary.
    Use first-order, upwind approximation

    Works for both 1D and 2D data.
    In 2D case, only do one dimension specified by `dim`.
    Need to apply over both dimensions to get a true 2D solver (see `vanleer_tendency_2d()`).

    Args:
      c: 1d or 2d Tensor, density field
      u: 1d or 2d Tensor, wind field
      dx: 0d Tensor, grid spacing (assume uniform)
      dt: 0d Tensor, time step
      dim: int, dimension of advection. In 1D, always be 0; in 2D, use 0 or 1 for each dimension

    Returns:
      1d or 2d Tensor with same shape as `c`
    '''

    # re-stagger to C-grid, pointing from box[i-1] to box[i]
    # remove this step if input wind is already staggered
    u = 0.5*(u+tf.manip.roll(u, 1, dim))

    c_l = tf.manip.roll(c, 1, dim)  # so c_l[i] == c[i-1], with periodic boundary

    flux_right = tf.maximum(u, 0) * c_l
    flux_left = tf.minimum(u, 0) * c
    flux = flux_right + flux_left

    tendency = flux - tf.manip.roll(flux, -1, dim)

    return tendency*dt/dx


def vanleer_tendency_1d(c, u, dx, dt, dim=0):
    '''
    Tensorflow vectorized version of advection tendency with periodic boundary.
    Use second-order, piece-wise linear approximation with VanLeer flux-limiter.

    Works for both 1D and 2D data.
    In 2D case, only do one dimension specified by `dim`. 
    Need to apply over both dimensions to get a true 2D solver (see `vanleer_tendency_2d()`). 

    Args:
      c: 1d or 2d Tensor, density field
      u: 1d or 2d Tensor, wind field
      dx: 0d Tensor, grid spacing (assume uniform)
      dt: 0d Tensor, time step
      dim: int, dimension of advection. In 1D, always be 0; in 2D, use 0 or 1 for each dimension

    Returns:
      1d or 2d Tensor with same shape as `c`
    '''

    # re-stagger to C-grid, pointing from box[i-1] to box[i]
    # remove this step if input wind is already staggered
    u = 0.5*(u+tf.manip.roll(u, 1, dim))

    c_l = tf.manip.roll(c, 1, dim)
    c_r = tf.manip.roll(c, -1, dim)

    delta = c - c_l
    delta_avg = (delta + tf.manip.roll(delta, -1, dim))/2

    # can also use np.maximum.reduce()
    c_max = tf.maximum(c, tf.maximum(c_l, c_r))  # upper boundary
    c_min = tf.minimum(c, tf.minimum(c_l, c_r))  # lower boundary
    mismatch = tf.sign(delta_avg)*tf.minimum(tf.abs(delta_avg), tf.minimum(2*(c-c_min), 2*(c_max-c)))

    # compute flux from slope
    mismatch_l = tf.manip.roll(mismatch, 1, dim)

    cfl = u*dt/dx
    flux_right = tf.maximum(u, 0) * (c_l + mismatch_l*(1-cfl)/2)
    flux_left = tf.minimum(u, 0) * (c - mismatch*(1+cfl)/2)
    flux = flux_right + flux_left

    tendency = flux - tf.manip.roll(flux, -1, dim)

    return tendency*dt/dx


def upwind_tendency_2d(c, u, v, dx, dy, dt, flip_dim=True):
    '''
    2D advection tendency with periodic boundary
    Use upwind scheme for both outer operator and inner operator

    Args:
      c: 2d Tensor, density field
      u: 2d Tensor, wind field in x direction
      v: 2d Tensor, wind field in y direction
      dx: 0d Tensor, grid spacing (assume uniform)
      dy: 0d Tensor, grid spacing (assume uniform, but can be different from dx)
      dt: 0d Tensor, time step
      flip_dim: bool, assume (x, y) if True, assume (y, x) if False

    Returns:
      2d Tensor with same shape as `c`
    '''

    if flip_dim:  # (x, y), notation used in pde_superresolution_2d framework
        x_dim, y_dim = 0, 1
    else:  # (y, x), notation used in geoscience / my numpy advection solver
        x_dim, y_dim = 1, 0

    # operator splitting in x and y directions
    tendency = (upwind_tendency_1d(0.5*upwind_tendency_1d(c, v, dy, dt, dim=y_dim) + c, u, dx, dt, dim=x_dim) +
                upwind_tendency_1d(0.5*upwind_tendency_1d(c, u, dx, dt, dim=x_dim) + c, v, dy, dt, dim=y_dim)
                )

    return tendency


def vanleer_tendency_2d(c, u, v, dx, dy, dt, flip_dim=True):
    '''
    2D advection tendency with periodic boundary
    Use second-order (VanLeer) scheme for outer operator and upwind for inner operator

    Args:
      c: 2d Tensor, density field
      u: 2d Tensor, wind field in x direction
      v: 2d Tensor, wind field in y direction
      dx: 0d Tensor, grid spacing (assume uniform)
      dy: 0d Tensor, grid spacing (assume uniform, but can be different from dx)
      dt: 0d Tensor, time step
      flip_dim: bool, assume (x, y) if True, assume (y, x) if False

    Returns:
      2d Tensor with same shape as `c`
    '''

    if flip_dim:  # (x, y), notation used in pde_superresolution_2d framework
        x_dim, y_dim = 0, 1
    else:  # (y, x), notation used in geoscience / my numpy advection solver
        x_dim, y_dim = 1, 0

    # operator splitting in x and y directions
    tendency = (vanleer_tendency_1d(0.5*upwind_tendency_1d(c, v, dy, dt, dim=y_dim) + c, u, dx, dt, dim=x_dim) +
                vanleer_tendency_1d(0.5*upwind_tendency_1d(c, u, dx, dt, dim=x_dim) + c, v, dy, dt, dim=y_dim)
                )

    return tendency
