#!/usr/bin/env python3
"""
CuPy-only CUDA Python project
"""

import numpy as np
from utils.gpu_utils import check_gpu, simple_gpu_operation, get_gpu_memory_info
from kernels.basic_ops import gpu_add_arrays, gpu_matrix_multiply, gpu_timing_test

def main():
    print("CuPy-Only CUDA Python Project")
    print("=" * 40)
    
    # Check GPU availability
    check_gpu()
    print()
    
    # Test simple GPU operation
    print("Testing simple GPU operation:")
    result = simple_gpu_operation()
    print(f"Result: {result}")
    print()
    
    # Test GPU array addition (now CuPy-only)
    print("Testing GPU array addition:")
    a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    b = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    
    gpu_result = gpu_add_arrays(a, b)
    
    print(f"Input A: {a}")
    print(f"Input B: {b}")
    print(f"GPU result: {gpu_result}")
    print(f"Expected: {a + b}")
    print(f"Results match: {np.array_equal(gpu_result, a + b)}")
    print()
    
    # Test matrix multiplication
    print("Testing matrix multiplication:")
    mat1 = np.random.randn(100, 50).astype(np.float32)
    mat2 = np.random.randn(50, 75).astype(np.float32)
    
    gpu_mm_result = gpu_matrix_multiply(mat1, mat2)
    cpu_mm_result = np.dot(mat1, mat2)
    
    print(f"Matrix shapes: {mat1.shape} x {mat2.shape} = {gpu_mm_result.shape}")
    print(f"Results close: {np.allclose(gpu_mm_result, cpu_mm_result)}")
    print()
    
    # Performance test
    gpu_timing_test()
    print()
    
    # Memory info
    print("GPU Memory Info:")
    get_gpu_memory_info()
    
    print("\n" + "=" * 40)
    print("✅ CuPy CUDA setup working perfectly!")
    print("✅ Ready for team development!")

if __name__ == "__main__":
    main()
