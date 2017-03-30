# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

import numpy as xp
cimport numpy as xp
cimport cython

ctypedef xp.float64_t DOUBLE_t
ctypedef xp.int_t INT_t

def resample(int num, xp.ndarray[DOUBLE_t, ndim=1] wcum):
    cdef int start, i, j, length
    cdef double n

    start = 0
    cdef xp.ndarray[INT_t, ndim=1] idxs = xp.zeros(num).astype(xp.int)
    cdef xp.ndarray[DOUBLE_t, ndim=1] rand = xp.sort(xp.random.rand(num))
    length = rand.size

    for i in xrange(length):
        for j in xrange(start, num):
            if rand[i] <= wcum[j]:
                idxs[i] = start = j
                break

    return idxs
