import cv2
import mediapipe as mp
from webcam_module import start_webcam, release_webcam

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Colors for each finger
FINGER_COLORS = {
    "Thumb": (255, 0, 0),    # Blue
    "Index Finger": (0, 255, 0),  # Green
    "Middle Finger": (0, 0, 255),  # Red
    "Ring Finger": (255, 255, 0),  # Cyan
    "Pinky": (255, 0, 255),  # Magenta
}

# Finger landmark mapping
FINGER_LANDMARKS = {
    "Thumb": mp_hands.HandLandmark.THUMB_TIP,
    "Index Finger": mp_hands.HandLandmark.INDEX_FINGER_TIP,
    "Middle Finger": mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    "Ring Finger": mp_hands.HandLandmark.RING_FINGER_TIP,
    "Pinky": mp_hands.HandLandmark.PINKY_TIP,
}

# Define the reusable Card and TextLabel components
class Card:
    def __init__(self, position, width, box_color=(0, 0, 0), alpha=0.5, text_color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8):
        self.position = position
        self.width = width
        self.box_color = box_color
        self.alpha = alpha
        self.text_color = text_color
        self.font = font
        self.font_scale = font_scale
        self.lines = []

    def add_line(self, text):
        self.lines.append(text)

    def clear(self):
        self.lines = []

    def draw(self, frame):
        if not self.lines:
            return frame

        overlay = frame.copy()
        x, y = self.position

        # Calculate dynamic height based on the number of lines
        line_height = int(self.font_scale * 30)
        height = line_height * len(self.lines) + 20

        # Draw the semi-transparent box
        cv2.rectangle(overlay, (x, y), (x + self.width, y + height), self.box_color, -1)
        frame = cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0)

        # Draw each line of text
        for i, line in enumerate(self.lines):
            text_y = y + line_height * (i + 1)
            cv2.putText(frame, line, (x + 10, text_y), self.font, self.font_scale, self.text_color, thickness=2)

        return frame


class TextLabel:
    def __init__(self, label, anchor, offset=(0, 0), text_color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=2):
        self.label = label
        self.anchor = anchor
        self.offset = offset
        self.text_color = text_color
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness

    def draw(self, frame):
        x, y = self.anchor
        offset_x, offset_y = self.offset
        position = (x + offset_x, y + offset_y)
        cv2.putText(frame, self.label, position, self.font, self.font_scale, self.text_color, self.thickness)


def main():
    # Start webcam
    try:
        cap = start_webcam(camera_index=1)
    except Exception as e:
        print(e)
        return

    # Initialize the Details Card
    details_card = Card(position=(10, 10), width=300, box_color=(0, 0, 0), alpha=0.6)

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        start_time = cv2.getTickCount()  # For runtime calculation

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

            # Clear the details card for new frame
            details_card.clear()

            # Calculate runtime
            current_time = cv2.getTickCount()
            runtime = (current_time - start_time) / cv2.getTickFrequency()
            runtime_text = f"Run Time: {runtime:.2f} s"

            # Add details to the card
            num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            details_card.add_line(f"Hands Detected: {num_hands}")
            details_card.add_line(runtime_text)

            # Process each detected hand
            if results.multi_hand_landmarks:
                for hand_landmarks, hand_classification in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # Determine hand label: "Left" or "Right"
                    hand_label = hand_classification.classification[0].label
                    color = (0, 0, 255) if hand_label == "Right" else (255, 0, 0)  # Red for right, blue for left

                    # Draw landmarks and connections
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    )

                    # Add a label for the wrist
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    wrist_x = int(wrist.x * frame.shape[1])
                    wrist_y = int(wrist.y * frame.shape[0])

                    wrist_label = TextLabel(
                        label=f"{hand_label} Hand",
                        anchor=(wrist_x, wrist_y),
                        offset=(30, -20) if hand_label == "Right" else (-80, -20),
                        text_color=(0, 255, 0)
                    )
                    wrist_label.draw(frame)

                    # Add label for left pinky tip
                    if hand_label == "Left":
                        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                        pinky_x = int(pinky_tip.x * frame.shape[1])
                        pinky_y = int(pinky_tip.y * frame.shape[0])
                        pinky_label = TextLabel(
                            label="Left Pinky",
                            anchor=(pinky_x, pinky_y),
                            offset=(-100, -10),
                            text_color=(0, 155, 55)
                        )
                        pinky_label.draw(frame)

            # Draw the details card
            frame = details_card.draw(frame)

            # Display the frame
            cv2.imshow("Hand Tracking with Labels", frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Release resources
    release_webcam(cap)


if __name__ == "__main__":
    main()