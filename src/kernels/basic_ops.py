import cupy as cp
import numpy as np
import time

def gpu_add_arrays(a, b):
    """GPU array addition using CuPy (no Numba needed)"""
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    result = a_gpu + b_gpu
    return cp.asnumpy(result)

def cupy_add_arrays(a, b):
    """Same as gpu_add_arrays - keeping for compatibility"""
    return gpu_add_arrays(a, b)

def gpu_matrix_multiply(a, b):
    """GPU matrix multiplication"""
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    result = cp.dot(a_gpu, b_gpu)
    return cp.asnumpy(result)

def gpu_timing_test():
    """Test GPU performance with timing"""
    print("GPU Performance Test:")
    
    # Large array operations
    size = 1000000
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    
    # GPU timing
    start = time.time()
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    result_gpu = a_gpu + b_gpu
    result = cp.asnumpy(result_gpu)
    gpu_time = time.time() - start
    
    # CPU timing for comparison
    start = time.time()
    cpu_result = a + b
    cpu_time = time.time() - start
    
    print(f"GPU time: {gpu_time*1000:.2f} ms")
    print(f"CPU time: {cpu_time*1000:.2f} ms")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    print(f"Results match: {np.allclose(result, cpu_result)}")
