import ctypes
import numpy as np
import copy

#so_file = "/home/benni/Coding/5.PK/Code/utils.so"
utils = np.ctypeslib.load_library('utils', '.')

def convolve(inputMatrix, kernel, stride):
    a = np.array(inputMatrix, dtype="float64")
    b = np.array(kernel, dtype="float64")

    INT = ctypes.c_int64
    PINT = ctypes.POINTER(ctypes.c_int64)
    PDOUBLE = ctypes.POINTER(ctypes.c_double)
    ND_POINTER_4 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=4, flags="C_CONTIGUOUS")

    utils.convolve.argtypes = [ND_POINTER_4, INT, INT, INT, INT, ND_POINTER_4, INT, INT, INT, INT, INT, PINT, PINT, PINT, PINT, PINT]
    utils.convolve.restype = PDOUBLE

    d1_out, d2_out, d3_out, d4_out, d5_out = INT(), INT(), INT(), INT(), INT()
    p_d1_out = ctypes.pointer(d1_out)
    p_d2_out = ctypes.pointer(d2_out)
    p_d3_out = ctypes.pointer(d3_out)
    p_d4_out = ctypes.pointer(d4_out)
    p_d5_out = ctypes.pointer(d5_out)
    out = utils.convolve(a, a.shape[0], a.shape[1], a.shape[2], a.shape[3], b, b.shape[0], b.shape[1], b.shape[2], b.shape[3], 1, p_d1_out, p_d2_out, p_d3_out, p_d4_out, p_d5_out)

    d1_out = d1_out.value
    d2_out = d2_out.value
    d3_out = d3_out.value
    d4_out = d4_out.value
    d5_out = d5_out.value
    result = np.ctypeslib.as_array(out, shape=(d1_out, d2_out, d3_out, d4_out, d5_out))
    return result
