import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

import advection_solver.advection_1d as np_solver_1d
import advection_solver.advection_2d as np_solver_2d
import advection_solver.advection_tensorflow as tf_solver

tfe.enable_eager_execution()


def test_upwind_1d():
    nx = 100
    Lx = 1
    dx = Lx/nx
    dt = 0.01
    nt = 500

    u = np.ones(nx) * 0.5
    c0 = np.zeros(nx)
    c0[int(nx*0.2):int(nx*0.4)] = 1.0  # square wave

    # numpy reference
    c = c0.copy()
    for _ in range(nt):
        c += np_solver_1d.upwind_tendency(c, u, dx, dt)

    # TF solution
    u_tensor = tf.convert_to_tensor(u)
    c_tensor = tf.convert_to_tensor(c0)
    for _ in range(nt):
        c_tensor += tf_solver.upwind_tendency_1d(c_tensor, u_tensor, dx, dt)

    np.array_equal(c_tensor.numpy(), c)


def test_vanleer_1d():
    nx = 100
    Lx = 1
    dx = Lx/nx
    dt = 0.01
    nt = 500

    u = np.ones(nx) * 0.5
    c0 = np.zeros(nx)
    c0[int(nx*0.2):int(nx*0.4)] = 1.0  # square wave

    # numpy reference
    c = c0.copy()
    for _ in range(nt):
        c += np_solver_1d.vanleer_tendency(c, u, dx, dt)

    # TF solution
    u_tensor = tf.convert_to_tensor(u)
    c_tensor = tf.convert_to_tensor(c0)
    for _ in range(nt):
        c_tensor += tf_solver.vanleer_tendency_1d(c_tensor, u_tensor, dx, dt)

    np.array_equal(c_tensor.numpy(), c)


def test_vanleer_2d():
    nx = 100
    ny = 100
    Lx = 1
    Ly = 1
    dx = Lx/nx
    dy = Ly/ny
    dt = 0.01
    nt = 100

    u = np.ones([nx, ny]) * 0.5
    v = np.ones([nx, ny]) * 0.3

    c0 = np.zeros([nx, ny])
    c0[int(ny*0.2):int(ny*0.4), int(nx*0.1):int(nx*0.3)] = 1.0  # 2D square

    # numpy reference
    c = c0.copy()
    for _ in range(nt):
        c += np_solver_2d.tendency_2d_vanleer(c, u, v, dx, dy, dt)

    # TF solution
    c_tensor = tf.convert_to_tensor(c0)
    u_tensor = tf.convert_to_tensor(u)
    v_tensor = tf.convert_to_tensor(v)

    for _ in range(nt):
        c_tensor += tf_solver.vanleer_tendency_2d(c_tensor, u_tensor, v_tensor, dx, dy, dt)

    np.array_equal(c_tensor.numpy(), c)
