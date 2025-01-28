import cv2
import mediapipe as mp
from webcam_module import start_webcam, release_webcam

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Colors for each finger
FINGER_COLORS = {
    0: (255, 0, 0),    # Thumb (Blue)
    1: (0, 255, 0),    # Index (Green)
    2: (0, 0, 255),    # Middle (Red)
    3: (255, 255, 0),  # Ring (Cyan)
    4: (255, 0, 255),  # Pinky (Magenta)
}

# Define the Card and TextLabel components
class Card:
    def __init__(self, position, box_color=(0, 0, 0), alpha=0.5, text_color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8):
        self.position = position
        self.box_color = box_color
        self.alpha = alpha
        self.text_color = text_color
        self.font = font
        self.font_scale = font_scale
        self.details = {}  # Dictionary to hold details

    def update_details(self, key, value):
        """
        Updates the details dictionary with a new key-value pair.
        Args:
            key: The property name.
            value: The property value.
        """
        self.details[key] = value

    def draw(self, frame):
        overlay = frame.copy()
        x, y = self.position
        line_height = 20  # Height of each row
        padding = 10  # Padding inside the card
        num_rows = len(self.details)

        # Calculate the card size dynamically
        width = 250
        height = padding * 2 + num_rows * line_height
        cv2.rectangle(overlay, (x, y), (x + width, y + height), self.box_color, -1)
        frame = cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0)

        # Draw each detail in a new row
        for i, (key, value) in enumerate(self.details.items()):
            text = f"{key}: {value}"
            text_position = (x + padding, y + padding + (i + 1) * line_height)
            cv2.putText(frame, text, text_position, self.font, self.font_scale, self.text_color, thickness=2)

        return frame


class StationaryTextLabel:
    def __init__(self, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, text_color=(255, 255, 255), thickness=2):
        self.position = position
        self.font = font
        self.font_scale = font_scale
        self.text_color = text_color
        self.thickness = thickness

    def draw(self, frame, text):
        """
        Draws a stationary text label on the frame.
        Args:
            frame: The video frame to draw on.
            text: The text to display.
        """
        cv2.putText(frame, text, self.position, self.font, self.font_scale, self.text_color, self.thickness)


class TrackedTextLabel:
    def __init__(self, landmark, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, text_color=(255, 255, 255), thickness=2, offset=(10, -10)):
        self.landmark = landmark
        self.font = font
        self.font_scale = font_scale
        self.text_color = text_color
        self.thickness = thickness
        self.offset = offset

    def draw(self, frame, text, hand_landmarks, frame_shape):
        """
        Draws a tracked text label anchored to a hand landmark with an offset.
        Args:
            frame: The video frame to draw on.
            text: The text to display.
            hand_landmarks: The landmarks of the detected hand.
            frame_shape: The shape of the frame (height, width, channels).
        """
        h, w, _ = frame_shape
        landmark_x = int(hand_landmarks.landmark[self.landmark].x * w)
        landmark_y = int(hand_landmarks.landmark[self.landmark].y * h)
        position = (landmark_x + self.offset[0], landmark_y + self.offset[1])

        cv2.putText(frame, text, position, self.font, self.font_scale, self.text_color, self.thickness)


def main():
    # Start webcam
    try:
        cap = start_webcam(camera_index=1)
    except Exception as e:
        print(e)
        return

    # Initialize DetailsCard
    details_card = Card(position=(10, 10), box_color=(0, 0, 0), alpha=0.6)

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        # Start the timer
        start_time = cv2.getTickCount() / cv2.getTickFrequency()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Flip the frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Update details dynamically
            run_time = int((cv2.getTickCount() / cv2.getTickFrequency()) - start_time)
            details_card.update_details("Run Time", f"{run_time}s")

            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                details_card.update_details("Hands Detected", num_hands)

                for hand_landmarks, hand_classification in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    hand_label = hand_classification.classification[0].label
                    color = (0, 0, 255) if hand_label == "Right" else (255, 0, 0)  # Red for right, blue for left

                    # Draw landmarks and connections
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    )

                    # Set offset dynamically based on hand label
                    offset = (100, 50) if hand_label == "Right" else (-50, -20)

                    # Draw the tracked text label
                    tracked_label = TrackedTextLabel(
                        landmark=mp_hands.HandLandmark.WRIST,
                        text_color=color,
                        offset=offset
                    )
                    tracked_label.draw(
                        frame,
                        text=f"{hand_label} Hand",
                        hand_landmarks=hand_landmarks,
                        frame_shape=frame.shape
                    )

            else:
                details_card.update_details("Hands Detected", 0)

            # Draw the details card
            frame = details_card.draw(frame)

            # Display the frame
            cv2.imshow("Hand Tracking with Customizable Details", frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release resources
    release_webcam(cap)


if __name__ == "__main__":
    main()