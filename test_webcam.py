# test_webcam.py

import cv2
import logging

# Configure logging to display debug information in the console
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')

def test_webcam(camera_index=0):
    """
    Tests the webcam by capturing and displaying video frames.
    
    Args:
        camera_index (int): The index of the webcam to use (default is 0).
    """
    logging.info(f"Attempting to open webcam with index {camera_index}.")
    
    # Initialize the webcam
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logging.error(f"Failed to open webcam with index {camera_index}.")
        return
    logging.info(f"Webcam with index {camera_index} opened successfully.")
    
    # Optionally, set the desired frame width and height
    desired_width = 640
    desired_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    logging.debug(f"Set webcam resolution to {desired_width}x{desired_height}.")
    
    logging.info("Starting video stream. Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to read frame from webcam.")
            break
        
        # Display the frame in a window named "Webcam Test"
        cv2.imshow("Webcam Test", frame)
        
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exit signal received. Closing webcam test.")
            break
    
    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    logging.info("Webcam released and all windows closed.")

if __name__ == "__main__":
    test_webcam(camera_index=1)  # Change the index if you have multiple webcams