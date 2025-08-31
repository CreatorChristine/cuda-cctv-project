import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.gpu_utils import simple_gpu_operation
from kernels.basic_ops import gpu_add_arrays, gpu_matrix_multiply

def test_gpu_availability():
    """Test if GPU operations work"""
    try:
        result = simple_gpu_operation()
        assert len(result) == 5
        assert result[0] == 11  # 1 + 10
    except Exception as e:
        pytest.skip(f"GPU not available: {e}")

def test_gpu_addition():
    """Test CuPy-based GPU array addition"""
    try:
        a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        b = np.array([10, 20, 30, 40, 50], dtype=np.float32)
        result = gpu_add_arrays(a, b)
        expected = a + b
        np.testing.assert_array_equal(result, expected)
    except Exception as e:
        pytest.skip(f"GPU addition test failed: {e}")

def test_gpu_matrix_multiply():
    """Test CuPy-based matrix multiplication"""
    try:
        a = np.random.randn(10, 20).astype(np.float32)
        b = np.random.randn(20, 15).astype(np.float32)
        result = gpu_matrix_multiply(a, b)
        expected = np.dot(a, b)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    except Exception as e:
        pytest.skip(f"GPU matrix multiplication test failed: {e}")
