"""Ultility functions"""

# np.roll doesn't work with jit(nopython=True)
# Implement numba-version of roll here.
# See https://groups.google.com/a/continuum.io/forum/#!topic/numba-users/k64DZL9JE2I
# Code copied from https://gist.github.com/synapticarbors/a22e1834d7cfc46eee2a26cebc6f817b


import numpy as np
from numba import types, jit
from numba.extending import overload_method


@overload_method(types.Array, 'take')
def array_take(arr, indices):
    if isinstance(indices, types.Array):
        def take_impl(arr, indices):
            n = indices.shape[0]
            res = np.empty(n, arr.dtype)
            for i in range(n):
                res[i] = arr[indices[i]]
            return res
        return take_impl


@jit(nopython=True)
def roll(a, shift):
    n = a.size
    reshape = True

    if n == 0:
        return a
    shift %= n

    indexes = np.concatenate((np.arange(n - shift, n), np.arange(n - shift)))

    res = a.take(indexes)
    if reshape:
        res = res.reshape(a.shape)
    return res
