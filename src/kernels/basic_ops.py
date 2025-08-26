from numba import cuda
import numpy as np
import cupy as cp

@cuda.jit
def add_arrays_kernel(a, b, result):
    "Basic CUDA kernel for array addition"
    idx = cuda.grid(1)
    if idx < a.size:
        result[idx] = a[idx] + b[idx]

def gpu_add_arrays(a, b):
    "Wrapper function for GPU array addition using Numba"
    # Allocate GPU memory
    a_gpu = cuda.to_device(a)
    b_gpu = cuda.to_device(b)
    result_gpu = cuda.device_array_like(a)
    
    # Configure kernel launch
    threads_per_block = 256
    blocks_per_grid = (a.size + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    add_arrays_kernel[blocks_per_grid, threads_per_block](a_gpu, b_gpu, result_gpu)
    
    # Copy result back to CPU
    return result_gpu.copy_to_host()

def cupy_add_arrays(a, b):
    "Simple array addition using CuPy"
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    result = a_gpu + b_gpu
    return cp.asnumpy(result)
