#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from qiskit import QuantumCircuit


def gray_code(b):
    '''Gray code of b.
    Args:
        b: int:
            binary integer
    Returns:
        Gray code of b.
    '''
    return b ^ (b >> 1)


def gray_permutation(a):
    '''Permute the vector a from binary to Gray code order.

    Args:
        a: vector
            1D NumPy array of size 2**n
    Returns:
        vector:
            Gray code permutation of a
    '''
    b = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        b[i] = a[gray_code(i)]
    return b

def gray_permutation_vectorized(a):
    """Fast Gray code permutation using NumPy vectorization."""
    indices = np.arange(a.shape[0])
    return a[indices ^ (indices >> 1)]


def sfwht(a):
    '''Scaled Fast Walsh-Hadamard transform of input vector a.

    Args:
        a: vector
            1D NumPy array of size 2**n.
    Returns:
        vector:
            Scaled Walsh-Hadamard transform of a.
    '''
    n = int(np.log2(a.shape[0]))
    for h in range(n):
        for i in range(0, a.shape[0], 2**(h+1)):
            for j in range(i, i+2**h):
                x = a[j]
                y = a[j + 2**h]
                a[j] = (x + y) / 2
                a[j + 2**h] = (x - y) / 2
    return a

from numba import njit

@njit
def sfwht_numba(a):
    """
    Numba Accelerated JIT SFWHT, can beat the vectorized version by almost 4x, and naive version by 400x
    
    Args:
        a (np.ndarray): Input vector of size 2^n.

    Returns:
        np.ndarray: Scaled Walsh-Hadamard transform of `a`.
    """
    n = int(np.log2(a.shape[0]))
    N = a.shape[0]
    for h in range(n):
        mh = 1 << (h + 1)
        m = mh // 2
        for i in range(0, N, mh):
            for j in range(i, i + m):
                x = a[j]
                y = a[j + m]
                a[j] = (x + y) / 2
                a[j + m] = (x - y) / 2
    return a

def sfwht_optimized_vectorized(a):
    """
    Fully vectorized SFWHT using 2D reshaping to eliminate Python loops. Almost 50-100x faster than naive version

    Args:
        a (np.ndarray): Input vector of size 2^n.

    Returns:
        np.ndarray: Scaled Walsh-Hadamard transform of `a`.
    """
    n = int(np.log2(a.shape[0]))
    N = a.shape[0]

    for h in range(n):
        mh = 1 << (h + 1)
        m = mh // 2
        a = a.reshape(-1, mh)
        a[:, :m] = a[:, :m] + a[:, m:]
        a[:, m:] = a[:, :m] - 2 * a[:, m:]
        a = a.flatten()

    a /= N
    return a
    
def compute_control(i, n):
    '''Compute the control qubit index based on the index i and size n.'''
    if i == 4**n:
        return 1
    return 2*n - int(np.log2(gray_code(i-1) ^ gray_code(i)))


def compressed_uniform_rotation(a, ry=True):
    '''Compute a compressed uniform rotation circuit based on the thresholded
    vector a.

    Args:
        a: vector:
            A thresholded vector a a of dimension 2**n
        ry: bool
            uniform ry rotation if true, else uniform rz rotation
    Returns:
        circuit
            A qiskit circuit representing the compressed uniform rotation.
    '''
    n = int(np.log2(a.shape[0])/2)
    circ = QuantumCircuit(2*n + 1)

    i = 0
    while i < a.shape[0]:
        parity_check = 0

        # add the rotation gate
        if a[i] != 0:
            if ry:
                circ.ry(a[i], 0)
            else:
                circ.rz(a[i], 0)

        # loop over sequence of consecutive zeros
        while True:
            ctrl = compute_control(i+1, n)
            # toggle control bit
            parity_check = (parity_check ^ (1 << (ctrl-1)))
            i += 1
            if i >= a.shape[0] or a[i] != 0:
                break

        # add CNOT gates
        for j in range(1, 2*n+1):
            if parity_check & (1 << (j-1)):
                circ.cx(j, 0)

    return circ


def count_trailing_ones_optimized(x):
    if x == 0:
        return 0
    return ((x ^ (x + 1)) >> 1).bit_length()

def compute_control_optimized(i, n, max_i):
    if i == max_i:
        return 1
    trailing_ones = count_trailing_ones_optimized(i - 1)
    return 2*n - trailing_ones

def precompute_control_lut(n):
    max_trailing_ones = 2*n
    return [2*n - t for t in range(max_trailing_ones + 1)]

from functools import lru_cache

# @lru_cache(maxsize=None)
def compressed_uniform_rotation_with_lut(a, ry=True):
    """
    Optimized compressed uniform rotation using a control LUT.

    Args:
        a (np.ndarray): Thresholded vector of dimension 2^(2n)
        ry (bool): Apply RY if True, RZ otherwise

    Returns:
        QuantumCircuit: Qiskit circuit representing the rotation.
    """
    from qiskit import QuantumCircuit

    n = int(np.log2(a.shape[0]) // 2)
    max_i = a.shape[0]
    circ = QuantumCircuit(2 * n + 1)

    # Precompute control LUT (size O(n))
    max_trailing_ones = 2 * n
    control_lut = precompute_control_lut(n)

    i = 0
    while i < max_i:
        parity_check = 0

        # Apply rotation
        if a[i] != 0:
            if ry:
                circ.ry(a[i], 0)
            else:
                circ.rz(a[i], 0)

        # Skip zero blocks
        while True:
            # Special case: i+1 == max_i
            if i + 1 == max_i:
                ctrl = 1
            else:
                # Compute trailing_ones(i) and use LUT
                trailing_ones = count_trailing_ones_optimized(i)
                ctrl = control_lut[trailing_ones]

            # Toggle control bit
            parity_check ^= (1 << (ctrl - 1))
            i += 1

            # Break if non-zero or end of array
            if i >= max_i or a[i] != 0:
                break

        # Add CNOT gates (only for set bits)
        bit = 1
        pos = 0
        while bit <= parity_check:
            if parity_check & bit:
                circ.cx(pos + 1, 0)
            bit <<= 1
            pos += 1

    return circ
