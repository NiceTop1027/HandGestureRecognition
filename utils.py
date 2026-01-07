"""
Utility functions for hand gesture AR application
"""
import numpy as np
import time


class FPSCounter:
    """Calculate and display FPS"""
    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0
        
    def update(self):
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
        self.prev_time = current_time
        return int(self.fps)


def normalize_to_pixel(landmark, width, height):
    """
    Convert MediaPipe normalized coordinates to pixel coordinates
    
    Args:
        landmark: MediaPipe landmark with x, y, z coordinates
        width: Frame width in pixels
        height: Frame height in pixels
    
    Returns:
        tuple: (x, y) pixel coordinates
    """
    x = int(landmark.x * width)
    y = int(landmark.y * height)
    return (x, y)


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1: tuple (x, y)
        point2: tuple (x, y)
    
    Returns:
        float: distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def smooth_value(current, target, smoothing=0.5):
    """
    Apply exponential smoothing to reduce jitter
    
    Args:
        current: Current value
        target: Target value
        smoothing: Smoothing factor (0-1, higher = smoother)
    
    Returns:
        float: Smoothed value
    """
    return current * smoothing + target * (1 - smoothing)


def get_hsv_color(hue):
    """
    Convert HSV hue to BGR color
    
    Args:
        hue: Hue value (0-179)
    
    Returns:
        tuple: BGR color
    """
    import cv2
    hsv = np.uint8([[[hue, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(map(int, bgr[0][0]))


def lerp(a, b, t):
    """
    Linear interpolation between a and b
    
    Args:
        a: Start value
        b: End value
        t: Interpolation factor (0-1)
    
    Returns:
        Interpolated value
    """
    return a + (b - a) * t
