import cupy as cp
import numpy as np

def check_gpu():
    "Check if GPU is available and print info"
    print(f"CUDA available: {cp.cuda.is_available()}")
    if cp.cuda.is_available():
        print(f"GPU count: {cp.cuda.runtime.getDeviceCount()}")
        print(f"Current device: {cp.cuda.device.get_device_id()}")
        
def simple_gpu_operation():
    "Simple test of GPU operations"
    # Using CuPy (NumPy-like)
    a_gpu = cp.array([1, 2, 3, 4, 5])
    b_gpu = cp.array([10, 20, 30, 40, 50])
    result = a_gpu + b_gpu
    return cp.asnumpy(result)  # Convert back to CPU

def get_gpu_memory_info():
    "Get GPU memory information"
    if cp.cuda.is_available():
        mempool = cp.get_default_memory_pool()
        print(f"Used memory: {mempool.used_bytes() / 1024**2:.2f} MB")
        print(f"Total memory: {mempool.total_bytes() / 1024**2:.2f} MB")
