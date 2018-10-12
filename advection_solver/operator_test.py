"""Tests for 1D advection operators"""

import numpy as np

from advection_solver.operator import upwind_tendency, vanleer_tendency

nx = 100
Lx = 1
dx = Lx/nx
dt = 0.01
nt = 500

u = np.ones(nx) * 0.5
c0 = np.zeros(nx)
c0[int(nx*0.2):int(nx*0.4)] = 1.0  # square wave


def test_upwind():
    c = c0.copy()
    for _ in range(nt):
        c += upwind_tendency(c, u, dx, dt)

    assert c.argmax() == 79  # center location
    np.testing.assert_almost_equal(c.mean(), c0.mean())  # mass conservation
    assert c.max() < 1.0  # no over-shoot
    assert c.min() > 0.0  # positiivty


def test_vanleer():
    c = c0.copy()
    for _ in range(nt):
        c += vanleer_tendency(c, u, dx, dt)

    assert c.argmax() == 79  # center location
    np.testing.assert_almost_equal(c.mean(), c0.mean())  # mass conservation
    np.testing.assert_almost_equal(c.max(), 1, decimal=4)  # peak-preserving
    np.testing.assert_almost_equal(c.min(), 0)  # positiivty
