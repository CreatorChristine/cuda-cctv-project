# src/utils/performance.py - Performance comparison utilities
import time
import numpy as np
import cupy as cp

class PerformanceBenchmark:
    """Compare CPU vs GPU performance"""
    
    @staticmethod
    def benchmark_operation(cpu_func, gpu_func, data, iterations=5):
        """Benchmark CPU vs GPU operation"""
        
        # CPU timing
        cpu_times = []
        for _ in range(iterations):
            start = time.time()
            cpu_result = cpu_func(data)
            cpu_times.append(time.time() - start)
        
        # GPU timing  
        gpu_times = []
        for _ in range(iterations):
            start = time.time()
            gpu_result = gpu_func(data)
            cp.cuda.Stream.null.synchronize()
            gpu_times.append(time.time() - start)
        
        cpu_avg = np.mean(cpu_times) * 1000  # Convert to ms
        gpu_avg = np.mean(gpu_times) * 1000
        speedup = cpu_avg / gpu_avg
        
        # Check results are close
        results_match = np.allclose(cpu_result.flatten()[:100], 
                                  gpu_result.flatten()[:100], 
                                  rtol=1e-5)
        
        return {
            'cpu_time_ms': cpu_avg,
            'gpu_time_ms': gpu_avg,
            'speedup': speedup,
            'results_match': results_match
        }
