from ctypes import *
import numpy as np
import copy

so_file = "/home/benni/Coding/5.PK/Code/utils.so"
utils = CDLL(so_file)

INT = c_int64
ND_POINTER_4 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=4, flags="C")
ND_POINTER_5 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=5, flags="C")

utils.convolve.argtypes = [ND_POINTER_4, INT, INT, INT, INT, ND_POINTER_4, INT, INT, INT, INT, INT]
utils.convolve.restype = ND_POINTER_5

kernel_map = np.array([[[[1,2],
                       [3,4]]],
                      
                       [[[5,6],
                       [7,8]]],
                      
                        
                      [[[9,10],
                       [11,12]]]], dtype="float64")

gradient = np.array([ [[[1,1,1],
                     [1,1,1],
                     [1,1,1]]],
                    
                    [[[2,2,2],
                     [2,2,2],
                     [2,2,2]]],
                    
                    [[[3,3,3],
                     [3,3,3],
                     [3,3,3]]]], dtype="float64")

kernel_copy = copy.copy(kernel_map)
kernel_copy = np.flip(kernel_copy, axis=2)
kernel_copy = np.flip(kernel_copy, axis=1)
margin_height = (gradient.shape[2]*1 - 1)
margin_width = (gradient.shape[3]*1 - 1)
kernel_copy = np.pad(kernel_copy, ((0,0), (0,0),(margin_height, margin_height),(margin_width, margin_width)))

############################################

d1_inputMatrix = kernel_copy.shape[0]
d2_inputMatrix = kernel_copy.shape[1]
h_inputMatrix = kernel_copy.shape[2]
w_inputMatrix =kernel_copy.shape[3]

d1_kernel = gradient.shape[0]
d2_kernel = gradient.shape[1]
h_kernel = gradient.shape[2]
w_kernel = gradient.shape[3]

result = utils.convolve(kernel_copy, d1_inputMatrix, d2_inputMatrix, h_inputMatrix, w_inputMatrix, gradient, d1_kernel, d2_kernel, h_kernel, w_kernel, 1)
print(result)

