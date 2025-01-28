# utils/calculations.py

import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def get_coordinates(point, frame_shape):
    """
    Extracts x and y coordinates from a point, handling both tuples and objects with `x` and `y` attributes.

    Args:
        point (tuple or object): 
            A tuple (x, y) with normalized coordinates or an object with `x` and `y` attributes.
        frame_shape (tuple): 
            A tuple (height, width) representing the shape of the frame in pixels.

    Returns:
        tuple: 
            A tuple (x_pixel, y_pixel) in pixel dimensions.

    Raises:
        TypeError: 
            If `frame_shape` is not a tuple/list of two integers.
        ValueError: 
            If `frame_shape` dimensions are not positive or if point coordinates are out of bounds.
        AttributeError: 
            If point objects lack `x` and `y` attributes.
    """
    if not isinstance(frame_shape, (tuple, list)) or len(frame_shape) != 2:
        raise TypeError("frame_shape must be a tuple or list with two elements (height, width).")
    
    height, width = frame_shape
    if not (isinstance(height, int) and isinstance(width, int)):
        raise TypeError("frame_shape must contain integers for height and width.")
    if height <= 0 or width <= 0:
        raise ValueError("frame_shape dimensions must be positive integers.")

    if isinstance(point, (tuple, list)):
        if len(point) != 2:
            raise ValueError("Point tuple must have exactly two elements.")
        x, y = point
    else:
        if not hasattr(point, 'x') or not hasattr(point, 'y'):
            raise AttributeError("Point object must have 'x' and 'y' attributes.")
        x, y = point.x, point.y

    if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
        raise ValueError("Point coordinates must be normalized between 0 and 1.")

    return x * width, y * height


def calculate_midpoint(point1, point2, frame_shape):
    """
    Calculates the midpoint between two 2D points in pixel coordinates.

    Args:
        point1 (tuple or object): 
            First point as a tuple (x, y) with normalized coordinates 
            or an object with `x` and `y` attributes.
        point2 (tuple or object): 
            Second point as a tuple (x, y) with normalized coordinates 
            or an object with `x` and `y` attributes.
        frame_shape (tuple): 
            A tuple (height, width) representing the shape of the frame in pixels.

    Returns:
        tuple: 
            A tuple (midpoint_x, midpoint_y) representing the midpoint in pixel dimensions.

    Raises:
        ValueError: 
            If `frame_shape` is not provided or is invalid.
        TypeError: 
            If `frame_shape` is not a tuple/list of two integers.
        AttributeError: 
            If point objects lack `x` and `y` attributes.
    """
    x1, y1 = get_coordinates(point1, frame_shape)
    x2, y2 = get_coordinates(point2, frame_shape)

    midpoint_x = int((x1 + x2) / 2)
    midpoint_y = int((y1 + y2) / 2)

    return (midpoint_x, midpoint_y)


def calculate_distance(point1, point2, frame_shape):
    """
    Calculates the Euclidean distance between two 2D points in pixel coordinates.

    Args:
        point1 (tuple or object): 
            First point as a tuple (x, y) with normalized coordinates 
            or an object with `x` and `y` attributes.
        point2 (tuple or object): 
            Second point as a tuple (x, y) with normalized coordinates 
            or an object with `x` and `y` attributes.
        frame_shape (tuple): 
            A tuple (height, width) representing the shape of the frame in pixels.

    Returns:
        float: 
            The distance in pixels.

    Raises:
        ValueError: 
            If `frame_shape` is not provided or is invalid.
        TypeError: 
            If `frame_shape` is not a tuple/list of two integers.
        AttributeError: 
            If point objects lack `x` and `y` attributes.
    """
    x1, y1 = get_coordinates(point1, frame_shape)
    x2, y2 = get_coordinates(point2, frame_shape)

    distance = math.hypot(x2 - x1, y2 - y1)
    if distance == 0:
        logging.warning("Both points are identical; distance is zero.")
    return distance