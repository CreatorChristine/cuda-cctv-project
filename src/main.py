#!/usr/bin/env python3
"
Main application file for CUDA Python project
"

import numpy as np
from utils.gpu_utils import check_gpu, simple_gpu_operation
from kernels.basic_ops import gpu_add_arrays, cupy_add_arrays

def main():
    print("CUDA Python Project Starting...")
    print("=" * 40)
    
    # Check GPU availability
    check_gpu()
    print()
    
    # Test simple GPU operation
    print("Testing simple GPU operation:")
    result = simple_gpu_operation()
    print(f"Result: {result}")
    print()
    
    # Test custom CUDA kernel
    print("Testing custom CUDA kernel:")
    a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    b = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    
    kernel_result = gpu_add_arrays(a, b)
    cupy_result = cupy_add_arrays(a, b)
    
    print(f"Input A: {a}")
    print(f"Input B: {b}")
    print(f"Kernel result: {kernel_result}")
    print(f"CuPy result: {cupy_result}")
    print(f"Results match: {np.array_equal(kernel_result, cupy_result)}")

if __name__ == "__main__":
    main()
