import cv2

def start_webcam(camera_index=1):
    """
    Initializes the webcam with the specified camera index.
    Args:
        camera_index (int): The index of the camera to use (default is 1).
    Returns:
        cap: cv2.VideoCapture object for the webcam.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise Exception(f"Error: Could not access the webcam at index {camera_index}.")
    return cap

def release_webcam(cap):
    """
    Releases the webcam resources.
    Args:
        cap: cv2.VideoCapture object to release.
    """
    cap.release()
    cv2.destroyAllWindows()