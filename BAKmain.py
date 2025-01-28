# main.py

import cv2
import mediapipe as mp
from webcam_module import start_webcam, release_webcam
import logging

# Configure logging to output to console with DEBUG level
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def main():
    logging.info("Starting Simplified Hand Tracking Application.")
    
    # Start webcam
    camera_index = 0  # Adjust if you have multiple webcams
    try:
        cap = start_webcam(camera_index=camera_index)
        logging.info(f"Webcam started successfully on camera index {camera_index}.")
    except Exception as e:
        logging.error(f"Failed to start webcam at index {camera_index}: {e}")
        return

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        logging.info("MediaPipe Hands initialized.")

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Warning: Could not read frame. Retrying...")
                continue  # Retry capturing the next frame

            logging.debug("Frame read successfully.")

            # Flip the frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            logging.debug("Frame flipped horizontally.")

            # Convert to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            logging.debug("Frame processed by MediaPipe Hands.")

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )
                    logging.debug("Hand landmarks drawn.")
            else:
                logging.debug("No hands detected in the frame.")

            # Display the frame
            cv2.imshow("Simplified Hand Tracking", frame)
            logging.debug("Frame displayed.")

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logging.info("Exit signal received. Closing application.")
                break

    # Release resources
    release_webcam(cap)
    logging.info("Webcam released and all windows closed.")

if __name__ == "__main__":
    main()