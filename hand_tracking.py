import cv2
import mediapipe as mp
import time
from webcam_module import start_webcam, release_webcam

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Card Component
class Card:
    def __init__(self, position, box_color=(0, 0, 0), alpha=0.5, text_color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8):
        self.position = position
        self.box_color = box_color
        self.alpha = alpha
        self.text_color = text_color
        self.font = font
        self.font_scale = font_scale
        self.details = {}

    def update_details(self, new_details):
        self.details.update(new_details)

    def draw(self, frame):
        x, y = self.position
        line_height = 25  # Adjust line spacing
        total_height = line_height * len(self.details)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + 250, y + total_height + 20), self.box_color, -1)
        frame = cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0)

        for i, (key, value) in enumerate(self.details.items()):
            text = f"{key}: {value}"
            text_y = y + (i + 1) * line_height
            cv2.putText(frame, text, (x + 10, text_y), self.font, self.font_scale, self.text_color, 2)

        return frame

# TextLabel Component
class TextLabel:
    def __init__(self, label, anchor, offset=(0, 0), text_color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=2):
        self.label = label
        self.anchor = anchor
        self.offset = offset
        self.text_color = text_color
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness

    def draw(self, frame, landmarks=None):
        if isinstance(self.anchor, tuple):  # Fixed point (x, y)
            x, y = self.anchor
        elif landmarks and isinstance(self.anchor, mp_hands.HandLandmark):  # Dynamic point from landmarks
            x = int(landmarks[self.anchor].x * frame.shape[1])
            y = int(landmarks[self.anchor].y * frame.shape[0])
        else:
            raise ValueError("Invalid anchor point or missing landmarks for dynamic anchor.")

        # Apply offset
        x += self.offset[0]
        y += self.offset[1]

        # Draw the text
        cv2.putText(frame, self.label, (x, y), self.font, self.font_scale, self.text_color, self.thickness)

# Main Hand Tracking Function
def main():
    # Start webcam
    try:
        cap = start_webcam(camera_index=1)
    except Exception as e:
        print(e)
        return

    # Initialize DetailsCard
    details_card = Card(position=(10, 10), box_color=(0, 0, 0), alpha=0.6)

    # Start time for run time calculation
    start_time = time.time()

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
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

            # Gather details for the details card
            run_time = int(time.time() - start_time)
            details = {"Hands Detected": 0, "Run Time": f"{run_time}s"}

            # Store handedness information to avoid redundant checks
            handedness_info = {}

            # Draw landmarks, connections, and labels if detected
            if results.multi_hand_landmarks:
                details["Hands Detected"] = len(results.multi_hand_landmarks)
                for idx, (hand_landmarks, hand_classification) in enumerate(
                    zip(results.multi_hand_landmarks, results.multi_handedness)
                ):
                    # Determine hand label: "Left" or "Right"
                    hand_label = hand_classification.classification[0].label
                    handedness_info[idx] = hand_label
                    color = (0, 0, 255) if hand_label == "Right" else (255, 0, 0)

                    # Draw landmarks and connections
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    )

                    # Draw label for wrist
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    wrist_x = int(wrist.x * frame.shape[1])
                    wrist_y = int(wrist.y * frame.shape[0])
                    wrist_label = TextLabel(
                        label=f"{hand_label} Hand",
                        anchor=(wrist_x, wrist_y),
                        offset=(30, -30) if hand_label == "Right" else (-30, -30),
                        text_color=color
                    )
                    wrist_label.draw(frame)

                    # Draw label for left pinky if it's the left hand
                    if hand_label == "Left":
                        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                        pinky_x = int(pinky_tip.x * frame.shape[1])
                        pinky_y = int(pinky_tip.y * frame.shape[0])
                        pinky_label = TextLabel(
                            label="Left Pinky",
                            anchor=(pinky_x, pinky_y),
                            offset=(-50, -10),
                            text_color=(0, 255, 255)
                        )
                        pinky_label.draw(frame)

            # Update and draw the details card
            details_card.update_details(details)
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