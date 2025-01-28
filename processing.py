import math

def calculate_midpoint(point1, point2, frame_shape):
    """
    Calculates the midpoint between two 2D points in pixel coordinates.

    Args:
        point1: First point (x1, y1) as a tuple or an object with `x` and `y` attributes.
        point2: Second point (x2, y2) as a tuple or an object with `x` and `y` attributes.
        frame_shape: A tuple (height, width) representing the shape of the frame.

    Returns:
        A tuple (midpoint_x, midpoint_y) in pixel dimensions.
    """
    if frame_shape:
        height, width = frame_shape
        x1, y1 = point1.x * width, point1.y * height
        x2, y2 = point2.x * width, point2.y * height
    else:
        raise ValueError("frame_shape must be provided for scaling normalized points.")

    midpoint_x = int((x1 + x2) / 2)
    midpoint_y = int((y1 + y2) / 2)

    return (midpoint_x, midpoint_y)

def calculate_distance(point1, point2, frame_shape):
    """
    Calculates the Euclidean distance between two 2D points in pixel coordinates.

    Args:
        point1: First point (x1, y1) as a tuple or an object with `x` and `y` attributes.
        point2: Second point (x2, y2) as a tuple or an object with `x` and `y` attributes.
        frame_shape: A tuple (height, width) representing the shape of the frame.

    Returns:
        The distance in pixels as a float.
    """
    if frame_shape:
        height, width = frame_shape
        x1, y1 = point1.x * width, point1.y * height
        x2, y2 = point2.x * width, point2.y * height
    else:
        raise ValueError("frame_shape must be provided for scaling normalized points.")

    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)