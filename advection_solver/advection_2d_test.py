"""Tests for 2D advection"""

import numpy as np

from advection_solver.advection_2d import tendency_2d_vanleer, tendency_2d_upwind

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


def test_2d_upwind():
    c = c0.copy()
    c = c0.copy()
    for _ in range(nt):
        c += tendency_2d_upwind(c, u, v, dx, dy, dt)

    np.testing.assert_almost_equal(c.mean(), c0.mean())  # mass conservation
    assert c.max() < 1.0  # no over-shoot
    np.testing.assert_almost_equal(c.min(), 0)  # positiivty


def test_2d_vanleer():
    c = c0.copy()
    for _ in range(nt):
        c += tendency_2d_vanleer(c, u, v, dx, dy, dt)

    np.testing.assert_almost_equal(c.mean(), c0.mean())  # mass conservation
    np.testing.assert_almost_equal(c.max(), 1, decimal=6)  # peak-preserving
    np.testing.assert_almost_equal(c.min(), 0, decimal=2)  # positiivty (slight violation)
