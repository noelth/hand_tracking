import cv2
import mediapipe as mp
import time
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

# Card class for details
class Card:
    def __init__(self, position, size=(250, 100), box_color=(0, 0, 0), alpha=0.6, text_color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8):
        self.position = position
        self.size = size
        self.box_color = box_color
        self.alpha = alpha
        self.text_color = text_color
        self.font = font
        self.font_scale = font_scale
        self.details = {}

    def update_details(self, key, value):
        """Update the details displayed on the card."""
        self.details[key] = value

    def draw(self, frame):
        """Draw the details card on the frame."""
        overlay = frame.copy()
        x, y = self.position
        w, h = self.size

        # Draw the semi-transparent box
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.box_color, -1)
        frame = cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0)

        # Draw each detail line
        line_height = 30  # Vertical spacing between lines
        for i, (key, value) in enumerate(self.details.items()):
            text = f"{key}: {value}"
            text_y = y + (i + 1) * line_height
            cv2.putText(frame, text, (x + 10, text_y), self.font, self.font_scale, self.text_color, thickness=2)

        return frame

# TextLabel class for hand labels
class TextLabel:
    def __init__(self, landmark, text="", text_color=(255, 255, 255), offset=(0, 0), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=2):
        self.landmark = landmark
        self.text = text
        self.text_color = text_color
        self.offset = offset
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness

    def set_text(self, text):
        self.text = text

    def set_offset(self, offset):
        self.offset = offset

    def draw(self, frame, hand_landmarks, frame_shape):
        if not self.text or not hand_landmarks:
            return

        # Get the landmark position in pixel coordinates
        lm = hand_landmarks.landmark[self.landmark]
        x, y = int(lm.x * frame_shape[1]), int(lm.y * frame_shape[0])

        # Apply the offset
        text_x = x + self.offset[0]
        text_y = y + self.offset[1]

        # Draw the text
        cv2.putText(frame, self.text, (text_x, text_y), self.font, self.font_scale, self.text_color, self.thickness)

def main():
    try:
        cap = start_webcam(camera_index=1)
    except Exception as e:
        print(e)
        return

    # Initialize Card and TextLabel instances
    details_card = Card(position=(10, 10), size=(300, 150), box_color=(0, 0, 0), alpha=0.6)
    hand_labels = {
        "Left": TextLabel(landmark=mp_hands.HandLandmark.WRIST, text_color=(255, 0, 0), offset=(-50, -20)),
        "Right": TextLabel(landmark=mp_hands.HandLandmark.WRIST, text_color=(0, 0, 255), offset=(50, 20)),
    }

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Update Run Time
            run_time = int(time.time() - start_time)
            details_card.update_details("Run Time", f"{run_time}s")

            # Process hand landmarks
            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                details_card.update_details("Hands Detected", num_hands)

                for hand_landmarks, hand_classification in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    hand_label = hand_classification.classification[0].label

                    # Draw finger connections and landmarks
                    for connection in mp_hands.HAND_CONNECTIONS:
                        start_idx, end_idx = tuple(connection)
                        start = hand_landmarks.landmark[start_idx]
                        end = hand_landmarks.landmark[end_idx]

                        start_point = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
                        end_point = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
                        finger_color = FINGER_COLORS[start_idx % 5]
                        cv2.line(frame, start_point, end_point, finger_color, thickness=2)

                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), radius=4, color=(255, 255, 255), thickness=-1)

                    # Update and draw the hand label
                    if hand_label == "Right":
                        hand_labels["Right"].set_text("Right Hand")
                        hand_labels["Right"].draw(frame, hand_landmarks, frame_shape=frame.shape)
                    elif hand_label == "Left":
                        hand_labels["Left"].set_text("Left Hand")
                        hand_labels["Left"].draw(frame, hand_landmarks, frame_shape=frame.shape)
            else:
                details_card.update_details("Hands Detected", 0)

            frame = details_card.draw(frame)
            cv2.imshow("Hand Tracking with Optimizations", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    release_webcam(cap)

if __name__ == "__main__":
    main()