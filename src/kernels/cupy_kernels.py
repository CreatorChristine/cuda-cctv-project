# src/kernels/cupy_kernels.py - CuPy-based GPU operations
import cupy as cp
import numpy as np
import time
from functools import wraps

def gpu_timing(func):
    """Decorator to time GPU operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
        end = time.time()
        print(f"{func.__name__} took {(end-start)*1000:.2f} ms")
        return result
    return wrapper

class CuPyKernels:
    """Collection of CuPy-based GPU operations"""
    
    @staticmethod
    @gpu_timing
    def vector_add(a, b):
        """GPU vector addition"""
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        result = a_gpu + b_gpu
        return cp.asnumpy(result)
    
    @staticmethod
    @gpu_timing
    def matrix_multiply(a, b):
        """GPU matrix multiplication"""
        a_gpu = cp.asarray(a)
        b_gpu = cp.asarray(b)
        result = cp.dot(a_gpu, b_gpu)
        return cp.asnumpy(result)
    
    @staticmethod
    @gpu_timing
    def element_wise_ops(arr):
        """Various element-wise operations"""
        arr_gpu = cp.asarray(arr)
        
        return {
            'squared': cp.asnumpy(arr_gpu ** 2),
            'sqrt': cp.asnumpy(cp.sqrt(cp.abs(arr_gpu))),
            'sin': cp.asnumpy(cp.sin(arr_gpu)),
            'exp': cp.asnumpy(cp.exp(arr_gpu / 10.0))
        }
    
    @staticmethod
    @gpu_timing
    def convolution_2d(image, kernel):
        """2D convolution (useful for image processing)"""
        from scipy import ndimage
        # Move to GPU
        image_gpu = cp.asarray(image)
        kernel_gpu = cp.asarray(kernel)
        
        # CuPy doesn't have built-in convolution, so we'll do it manually
        # This is a simple implementation - for production use cupyx.scipy
        result = cp.zeros_like(image_gpu)
        
        # Simple 2D convolution implementation
        kh, kw = kernel_gpu.shape
        ih, iw = image_gpu.shape
        
        for i in range(ih - kh + 1):
            for j in range(iw - kw + 1):
                result[i, j] = cp.sum(image_gpu[i:i+kh, j:j+kw] * kernel_gpu)
        
        return cp.asnumpy(result)
    
    @staticmethod
    @gpu_timing
    def parallel_reduction(arr, operation='sum'):
        """Parallel reduction operations"""
        arr_gpu = cp.asarray(arr)
        
        operations = {
            'sum': cp.sum,
            'mean': cp.mean,
            'max': cp.max,
            'min': cp.min,
            'std': cp.std
        }
        
        if operation not in operations:
            raise ValueError(f"Operation {operation} not supported")
        
        result = operations[operation](arr_gpu)
        return cp.asnumpy(result)
    
    @staticmethod
    def custom_kernel_elementwise(a, b, operation):
        """Custom element-wise operations using CuPy's RawKernel"""
        
        # Define CUDA kernel code as string
        cuda_code = '''
        extern "C" __global__
        void custom_operation(const float* a, const float* b, float* out, int size, int op) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                switch(op) {
                    case 0: out[idx] = a[idx] + b[idx]; break;  // add
                    case 1: out[idx] = a[idx] * b[idx]; break;  // multiply
                    case 2: out[idx] = fmaxf(a[idx], b[idx]); break;  // max
                    case 3: out[idx] = powf(a[idx], b[idx]); break;  // power
                }
            }
        }
        '''
        
        # Compile kernel
        kernel = cp.RawKernel(cuda_code, 'custom_operation')
        
        # Prepare data
        a_gpu = cp.asarray(a, dtype=cp.float32)
        b_gpu = cp.asarray(b, dtype=cp.float32)
        result_gpu = cp.zeros_like(a_gpu)
        
        # Operation mapping
        op_map = {'add': 0, 'multiply': 1, 'max': 2, 'power': 3}
        op_code = op_map.get(operation, 0)
        
        # Launch kernel
        threads_per_block = 256
        blocks_per_grid = (a_gpu.size + threads_per_block - 1) // threads_per_block
        
        kernel((blocks_per_grid,), (threads_per_block,), 
               (a_gpu, b_gpu, result_gpu, a_gpu.size, op_code))
        
        return cp.asnumpy(result_gpu)