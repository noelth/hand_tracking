# webcam_module.py

import cv2
import logging

def start_webcam(camera_index=0, width=1280, height=720):
    """
    Initializes and starts the webcam.

    Args:
        camera_index (int): Index of the webcam (default is 0).
        width (int): Desired width of the video frame.
        height (int): Desired height of the video frame.

    Returns:
        cv2.VideoCapture: The video capture object.

    Raises:
        RuntimeError: If the webcam cannot be opened.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam with index {camera_index}.")

    # Set the desired frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    logging.info(f"Webcam started on camera index {camera_index} with resolution {width}x{height}.")
    return cap

def release_webcam(cap):
    """
    Releases the webcam and closes all OpenCV windows.

    Args:
        cap (cv2.VideoCapture): The video capture object.
    """
    if cap:
        cap.release()
        logging.info("Webcam resources released.")
    cv2.destroyAllWindows()
    logging.info("All OpenCV windows closed.")