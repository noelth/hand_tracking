# main.py

import cv2
import mediapipe as mp
from webcam_module import start_webcam, release_webcam
from utils.calculations import calculate_midpoint, calculate_distance
from components.card import Card
from components.text_label import TextLabel
from utils.visualization import apply_vignette
import numpy as np
import logging

# Configure logging to output to console with DEBUG level
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define constants
FINGER_COLORS = {
    "Thumb": (255, 0, 0),            # Blue
    "Index Finger": (0, 255, 0),     # Green
    "Middle Finger": (0, 0, 255),    # Red
    "Ring Finger": (255, 255, 0),    # Cyan
    "Pinky": (255, 0, 255),          # Magenta
}

FINGER_LANDMARKS = {
    "Thumb": mp_hands.HandLandmark.THUMB_TIP,
    "Index Finger": mp_hands.HandLandmark.INDEX_FINGER_TIP,
    "Middle Finger": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    "Ring Finger": mp_hands.HandLandmark.RING_FINGER_TIP,
    "Pinky": mp_hands.HandLandmark.PINKY_TIP,
}

def main():
    logging.info("Starting the Hand Detection Application.")

    # Start webcam
    camera_index = 1  # Adjust if you have multiple webcams
    try:
        cap = start_webcam(camera_index=camera_index)
        logging.info(f"Webcam started successfully on camera index {camera_index}.")
    except Exception as e:
        logging.error(f"Failed to start webcam at index {camera_index}: {e}")
        # Attempt to start the default camera
        try:
            cap = start_webcam(camera_index=0)
            logging.info("Default webcam started successfully on camera index 0.")
        except Exception as e:
            logging.critical(f"Failed to start default webcam: {e}")
            return  # Exit the application if webcam cannot be started

    # Initialize the Details Card
    try:
        details_card = Card(position=(10, 10), width=300, box_color=(0, 0, 0), alpha=0.6)
        logging.debug("Details Card initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Details Card: {e}")
        details_card = None  # Proceed without the details card

    # Initialize MediaPipe Hands
    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            logging.info("MediaPipe Hands initialized.")
            start_time = cv2.getTickCount()  # For runtime calculation
            previous_time = start_time      # For FPS calculation

            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Warning: Could not read frame. Retrying...")
                    continue  # Retry capturing the next frame

                logging.debug("Frame read successfully.")

                # Flip the frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                logging.debug("Frame flipped horizontally.")

                # Apply the vignette effect
                try:
                    frame = apply_vignette(frame, opacity=0.6, vignette_color=(0, 0, 255))
                    logging.debug("Vignette effect applied.")
                except Exception as e:
                    logging.error(f"Failed to apply vignette effect: {e}")

                # Convert to RGB for MediaPipe processing
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)
                    logging.debug("Frame processed by MediaPipe Hands.")
                except Exception as e:
                    logging.error(f"Failed during MediaPipe processing: {e}")
                    results = None  # Proceed without processing

                # Clear the details card for new frame
                if details_card:
                    details_card.clear()

                # Calculate runtime
                current_time = cv2.getTickCount()
                runtime = (current_time - start_time) / cv2.getTickFrequency()
                runtime_text = f"Run Time: {runtime:.2f} s"

                # Calculate FPS
                fps = cv2.getTickFrequency() / (current_time - previous_time)
                previous_time = current_time
                fps_text = f"FPS: {fps:.2f}"

                # Add details to the card
                num_hands = len(results.multi_hand_landmarks) if results and results.multi_hand_landmarks else 0
                if details_card:
                    try:
                        details_card.add_line(f"Hands Detected: {num_hands}")
                        details_card.add_line(runtime_text)
                        details_card.add_line(fps_text)
                        logging.debug(f"Details Card updated: {num_hands} hands detected.")
                    except Exception as e:
                        logging.error(f"Failed to update Details Card: {e}")

                if results and results.multi_hand_landmarks:
                    right_hand_landmarks = None
                    left_hand_landmarks = None

                    # Separate hands by label
                    for hand_landmarks, hand_classification in zip(results.multi_hand_landmarks, results.multi_handedness):
                        try:
                            hand_label = hand_classification.classification[0].label
                            hand_score = hand_classification.classification[0].score
                            logging.debug(f"Detected {hand_label} hand with confidence {hand_score:.2f}.")
                        except Exception as e:
                            logging.error(f"Failed to extract hand classification: {e}")
                            continue  # Skip this hand

                        if hand_label == "Right":
                            right_hand_landmarks = hand_landmarks
                        elif hand_label == "Left":
                            left_hand_landmarks = hand_landmarks

                        # Add Hand Confidence to the details card
                        if details_card:
                            try:
                                details_card.add_line(f"{hand_label} Confidence: {hand_score:.2f}")
                            except Exception as e:
                                logging.error(f"Failed to add hand confidence to Details Card: {e}")

                    # Process right hand
                    if right_hand_landmarks:
                        try:
                            mp_drawing.draw_landmarks(
                                frame, right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                            )
                            logging.debug("Right hand landmarks drawn.")
                        except Exception as e:
                            logging.error(f"Failed to draw right hand landmarks: {e}")

                        # Draw the circle
                        try:
                            thumb_tip = right_hand_landmarks.landmark[FINGER_LANDMARKS["Thumb"]]
                            index_tip = right_hand_landmarks.landmark[FINGER_LANDMARKS["Index Finger"]]
                            circle_center = calculate_midpoint(thumb_tip, index_tip, frame.shape[:2])
                            circle_radius = int(calculate_distance(thumb_tip, index_tip, frame.shape[:2]) / 2)
                            logging.debug(f"Calculated circle center: {circle_center}, radius: {circle_radius}.")
                        except Exception as e:
                            logging.error(f"Failed to calculate circle parameters: {e}")
                            circle_center, circle_radius = None, 0

                        # Avoid drawing circles with zero radius
                        if circle_center and circle_radius > 0:
                            try:
                                cv2.circle(frame, circle_center, circle_radius, (0, 255, 0), 2)
                                logging.debug("Circle drawn on right hand.")
                            except Exception as e:
                                logging.error(f"Failed to draw circle on right hand: {e}")
                        else:
                            logging.info("Circle radius is zero or undefined; skipping drawing.")

                        # Add radius to the details card
                        if details_card and circle_radius > 0:
                            try:
                                details_card.add_line(f"Circle Radius: {circle_radius:.2f}px")
                            except Exception as e:
                                logging.error(f"Failed to add circle radius to Details Card: {e}")

                    # Process left hand
                    if left_hand_landmarks:
                        try:
                            mp_drawing.draw_landmarks(
                                frame, left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                            )
                            logging.debug("Left hand landmarks drawn.")
                        except Exception as e:
                            logging.error(f"Failed to draw left hand landmarks: {e}")

                        # Add Pointer 556 to left middle finger
                        try:
                            middle_finger_tip = left_hand_landmarks.landmark[FINGER_LANDMARKS["Middle Finger"]]
                            middle_x = int(middle_finger_tip.x * frame.shape[1])
                            middle_y = int(middle_finger_tip.y * frame.shape[0])
                            pointer_label = TextLabel(
                                label="Pointer 556",
                                anchor=(middle_x, middle_y),
                                offset=(-50, -10),
                                text_color=(255, 255, 0)  # Yellow
                            )
                            pointer_label.draw(frame)
                            logging.debug("Pointer label drawn on left middle finger.")
                        except Exception as e:
                            logging.error(f"Failed to draw Pointer Label: {e}")

                # Draw the details card
                if details_card:
                    try:
                        frame = details_card.draw(frame)
                        logging.debug("Details Card drawn on frame.")
                    except Exception as e:
                        logging.error(f"Failed to draw Details Card on frame: {e}")

                # Display the frame
                try:
                    cv2.imshow("Hand Tracking with Circle Radius", frame)
                    logging.debug("Frame displayed.")
                except Exception as e:
                    logging.error(f"Failed to display frame: {e}")

                # Exit when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logging.info("Exit signal received. Closing application.")
                    break
    finally:
        # Clean up resources
        if 'cap' in locals():
            release_webcam(cap)
            logging.info("Webcam released.")
        cv2.destroyAllWindows()
        logging.info("Windows destroyed. Application terminated.")

if __name__ == "__main__":
    main()
