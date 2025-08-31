# src/examples/image_processing.py - Example image processing with CuPy
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    """GPU-based image processing examples"""
    
    @staticmethod
    def gaussian_blur(image, sigma=1.0):
        """Apply Gaussian blur using GPU"""
        from cupyx.scipy import ndimage
        
        image_gpu = cp.asarray(image)
        blurred = ndimage.gaussian_filter(image_gpu, sigma=sigma)
        return cp.asnumpy(blurred)
    
    @staticmethod
    def edge_detection(image):
        """Sobel edge detection on GPU"""
        from cupyx.scipy import ndimage
        
        image_gpu = cp.asarray(image, dtype=cp.float32)
        
        # Sobel filters
        sobel_x = ndimage.sobel(image_gpu, axis=1)
        sobel_y = ndimage.sobel(image_gpu, axis=0)
        
        # Combine
        edges = cp.sqrt(sobel_x**2 + sobel_y**2)
        return cp.asnumpy(edges)
    
    @staticmethod
    def histogram_equalization(image):
        """Histogram equalization on GPU"""
        image_gpu = cp.asarray(image)
        
        # Calculate histogram
        hist, bins = cp.histogram(image_gpu.flatten(), bins=256, range=(0, 255))
        
        # Calculate CDF
        cdf = cp.cumsum(hist)
        cdf_normalized = cdf * 255 / cdf[-1]
        
        # Apply equalization
        equalized = cp.interp(image_gpu.flatten(), bins[:-1], cdf_normalized)
        equalized = equalized.reshape(image_gpu.shape)
        
        return cp.asnumpy(equalized.astype(cp.uint8))