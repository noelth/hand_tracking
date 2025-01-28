# utils/visualization.py

import cv2
import numpy as np

def apply_vignette(frame, opacity=0.4, vignette_color=(128, 128, 128)):
    """
    Applies a vignette effect to the video frame.

    Args:
        frame (numpy.ndarray): The video frame.
        opacity (float): Vignette transparency level (0 = no effect, 1 = full effect).
        vignette_color (tuple): BGR color tuple for the vignette.

    Returns:
        numpy.ndarray: The frame with the vignette applied.
    """
    rows, cols = frame.shape[:2]

    # Create Gaussian kernels for the vignette
    kernel_x = cv2.getGaussianKernel(cols, cols / 2)
    kernel_y = cv2.getGaussianKernel(rows, rows / 2)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)

    # Create a colored vignette
    vignette = np.zeros_like(frame, dtype=np.uint8)
    for i in range(3):  # Apply the vignette color to each channel
        vignette[:, :, i] = mask * (vignette_color[i])

    # Blend the vignette with the frame
    blended = cv2.addWeighted(frame, 1 - opacity, vignette, opacity, 0)
    return blended