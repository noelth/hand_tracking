# tests/test_calculations.py

import unittest
import math
from utils.calculations import calculate_midpoint, calculate_distance

class TestCalculations(unittest.TestCase):
    def test_calculate_midpoint_with_tuples(self):
        point1 = (0.2, 0.3)
        point2 = (0.4, 0.5)
        frame_shape = (480, 640)
        midpoint = calculate_midpoint(point1, point2, frame_shape)
        expected = (int((0.2 * 640 + 0.4 * 640) / 2), int((0.3 * 480 + 0.5 * 480) / 2))
        self.assertEqual(midpoint, expected)

    def test_calculate_distance_with_objects(self):
        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        point1 = Point(0.0, 0.0)
        point2 = Point(1.0, 1.0)
        frame_shape = (480, 640)
        distance = calculate_distance(point1, point2, frame_shape)
        expected = math.hypot(1.0 * 640 - 0.0 * 640, 1.0 * 480 - 0.0 * 480)
        self.assertAlmostEqual(distance, expected)

    def test_invalid_frame_shape(self):
        point1 = (0.1, 0.1)
        point2 = (0.9, 0.9)
        frame_shape = None
        with self.assertRaises(TypeError):
            calculate_midpoint(point1, point2, frame_shape)

    def test_invalid_point_format(self):
        point1 = (0.1, 0.1, 0.1)  # Invalid tuple length
        point2 = (0.9, 0.9)
        frame_shape = (480, 640)
        with self.assertRaises(ValueError):
            calculate_midpoint(point1, point2, frame_shape)

    def test_invalid_point_object(self):
        class IncompletePoint:
            def __init__(self, x):
                self.x = x

        point1 = IncompletePoint(0.1)
        point2 = (0.9, 0.9)
        frame_shape = (480, 640)
        with self.assertRaises(AttributeError):
            calculate_distance(point1, point2, frame_shape)

    def test_out_of_bounds_coordinates(self):
        point1 = (1.1, 0.5)  # x is out of bounds
        point2 = (0.9, 0.9)
        frame_shape = (480, 640)
        with self.assertRaises(ValueError):
            calculate_midpoint(point1, point2, frame_shape)

    def test_zero_distance(self):
        point1 = (0.5, 0.5)
        point2 = (0.5, 0.5)
        frame_shape = (480, 640)
        distance = calculate_distance(point1, point2, frame_shape)
        self.assertEqual(distance, 0.0)

if __name__ == '__main__':
    unittest.main()