import ctypes
import numpy as np
import copy
from scipy.signal import convolve2d

#so_file = "/home/benni/Coding/5.PK/Code/utils.so"
'''
utils = np.ctypeslib.load_library('utils', '.')

def convolve_sharedlib(inputMatrix, kernel, stride):
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
    '''

def getinputStackedColumns(inputMatrix, kernel_map_shape, stride):
    s0, s1, s2, s3 = inputMatrix.strides

    d1_input, d2_input, h_input, w_input = inputMatrix.shape
    h_kernel, w_kernel = kernel_map_shape

    out_shape = ( d1_input, d2_input, (h_input-h_kernel+1)//stride, (w_input-w_kernel+1)//stride, h_kernel, w_kernel)
    inputStackedColumns = np.lib.stride_tricks.as_strided(inputMatrix,
                                                          shape=out_shape,
                                                          strides=(s0, s1, stride*s2,stride*s3,s2, s3))
    return inputStackedColumns

def sigmoid(x):
    return (1 / (1 + np.exp(-x)) )

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def Convolution_strided_img2col(inputMatrix, kernel_map, stride):

    inputStackedColumns = getinputStackedColumns(inputMatrix, kernel_map.shape[1:], stride)
    d1_input, d2_input, h_input, w_input = inputMatrix.shape
    d_kernel, h_kernel, w_kernel = kernel_map.shape

    out_shape = ( d1_input, d2_input, (h_input-h_kernel+1)//stride, (w_input-w_kernel+1)//stride, h_kernel, w_kernel)

    inputStackedColumns = inputStackedColumns.flatten()
    inputStackedColumns = np.reshape(inputStackedColumns, (d1_input, d2_input, (h_input-h_kernel+1)//stride * ((w_input-w_kernel+1)//stride), h_kernel*w_kernel ))
    kernel_map_edited = kernel_map.reshape(d_kernel, h_kernel*w_kernel).transpose()
    im2col_conv = np.einsum("ijkl,lm->jkm", inputStackedColumns, kernel_map_edited)
    im2col_conv = im2col_conv.swapaxes(0,2).swapaxes(1,2)
    im2col_conv = im2col_conv.reshape(im2col_conv.shape[0], im2col_conv.shape[1], out_shape[2], out_shape[3])
    return im2col_conv

def Pooling_Matrixoperation(inputMatrix, kernel_shape, stride):

    windows = getinputStackedColumns(inputMatrix, kernel_shape, stride)
    d_kernel_map, d_input, h_input, w_input = inputMatrix.shape
    h_kernel, w_kernel = kernel_shape
    out_shape = ( d_kernel_map, d_input, (h_input-h_kernel+1)//stride, (w_input-w_kernel+1)//stride, h_kernel, w_kernel)

    maxs = np.max(windows, axis=(4,5))
    maxs = maxs.reshape(d_kernel_map, d_input, (h_input-h_kernel+1)//stride, (w_input-w_kernel+1)//stride)
    return maxs

def RELU_Matrixoperation(inputMatrix):
    return np.maximum(inputMatrix, 0)

def getPartialDerivativeConvolutionWRTkernelmap(inputMatrix, kernel_map, stride):
    #kernelmap is a 4d array

    inputStackedColumns = getinputStackedColumns(inputMatrix, kernel_map.shape[2:], stride)
    d1_input, d2_input, h_input, w_input = inputMatrix.shape
    d1_kernel, d2_kernel,h_kernel, w_kernel = kernel_map.shape

    inputStackedColumns = inputStackedColumns.flatten()
    inputStackedColumns = np.reshape(inputStackedColumns, (d1_input, d2_input, (h_input-h_kernel+1)//stride * ((w_input-w_kernel+1)//stride), h_kernel*w_kernel ))
    kernel_map_edited = kernel_map.reshape(d1_kernel, d2_kernel, h_kernel*w_kernel)
    im2col_conv = np.einsum("ijkl,ijl->ijk", inputStackedColumns, kernel_map_edited).reshape(d1_input,d2_input,(h_input-h_kernel+1)//stride,(w_input-w_kernel+1)//stride)
    return im2col_conv

def getPartialDerivativeConvolutionWRTx(kernel_map, gradient, stride):
   
    d_kernel_map, h_kernel_map, w_kernel_map = kernel_map.shape
    #5, 5, 5
    d1_gradient, d2_gradient, w_gradient, h_gradient = gradient.shape
    #5, 27, 170, 170
    
    result = []
    for i in range(d_kernel_map):
        onelayer = []
        
        for j in range(d2_gradient):
            
            img_result = convolve2d(kernel_map[i], gradient[i][j], mode="full")
            img_result = img_result.flip(axis=1).flip(axis=0)
            
            onelayer.append(img_result)
            
        result.append(onelayer)
    return np.array(onelayer)
            

def getPartialDerivateMaxPool(inputMatrix, gradientPreviousLayer, kernel_shape, stride):
    windows = getinputStackedColumns(inputMatrix, kernel_shape, stride)

    max = np.max(windows, axis=(4,5)).flatten().repeat(4, axis=0).reshape(windows.shape)
    gradientPreviousLayer = gradientPreviousLayer.flatten().repeat(4, axis=0).reshape(windows.shape)
    mask = np.equal(max, windows)
    windows.fill(0)
    windows[mask]=gradientPreviousLayer[mask]
    windows = np.lib.stride_tricks.as_strided(windows,
                                              shape=inputMatrix.shape,
                                              strides=inputMatrix.strides)
    return windows

def getMask(inputMatrix):
    inputMatrix[inputMatrix<=0] = 0
    inputMatrix[inputMatrix!=0] = 1
    return inputMatrix
