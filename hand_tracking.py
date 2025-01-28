import cv2
import mediapipe as mp
from webcam_module import start_webcam, release_webcam
import numpy as np
from processing import calculate_distance

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


def apply_vignette(frame, opacity=0.4, vignette_color=(128, 128, 128)):
    """
    Applies a vignette effect to the video frame.
    Args:
        frame: The video frame.
        opacity: Opacity of the vignette (0 = no effect, 1 = full effect).
        vignette_color: BGR color of the vignette.
    Returns:
        The frame with the vignette applied.
    """
    rows, cols = frame.shape[:2]

    # Create a vignette mask using Gaussian kernels
    kernel_x = cv2.getGaussianKernel(cols, cols / 2)
    kernel_y = cv2.getGaussianKernel(rows, rows / 2)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)

    # Apply the mask to create a colored vignette
    vignette = np.zeros_like(frame, dtype=np.uint8)
    for i in range(3):  # Apply the vignette color to each channel
        vignette[:, :, i] = mask * (vignette_color[i]) # / 255.0) divide to normalize

    # Blend the vignette with the frame
    return cv2.addWeighted(frame, 1 - opacity, vignette, opacity, 0)


class DynamicCircle:
    def __init__(self, center, radius, color=(0, 255, 255), thickness=2):
        """
        Initializes the circle.
        Args:
            center: (x, y) tuple for the circle's center.
            radius: Initial radius of the circle.
            color: BGR color of the circle.
            thickness: Thickness of the circle edge.
        """
        self.center = center
        self.radius = radius
        self.color = color
        self.thickness = thickness

    def update(self, center, radius):
        """
        Updates the circle's center and radius.
        Args:
            center: (x, y) tuple for the circle's new center.
            radius: New radius of the circle.
        """
        self.center = center
        self.radius = radius

    def draw(self, frame):
        """
        Draws the circle on the frame.
        Args:
            frame: The video frame.
        """
        cv2.circle(frame, self.center, int(self.radius), self.color, self.thickness)


def main():
    # Start webcam
    try:
        cap = start_webcam(camera_index=1)
    except Exception as e:
        print(e)
        return

    # Initialize the Details Card and Dynamic Circle
    details_card = Card(position=(10, 10), width=300, box_color=(0, 0, 0), alpha=0.6)
    dynamic_circle = DynamicCircle(center=(0, 0), radius=0)

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

            frame = cv2.flip(frame, 1)  # Flip the frame horizontally
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Clear the details card for new frame
            details_card.clear()

            # Calculate runtime
            current_time = cv2.getTickCount()
            runtime = (current_time - start_time) / cv2.getTickFrequency()
            details_card.add_line(f"Run Time: {runtime:.2f} s")

            # Process landmarks and draw circle
            if results.multi_hand_landmarks:
                for hand_landmarks, hand_classification in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    if hand_classification.classification[0].label == "Right":
                        # Get right thumb and index finger landmarks
                        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                        # Calculate the distance
                        distance = calculate_distance(thumb_tip, index_tip, frame.shape[1], frame.shape[0])
                        details_card.add_line(f"Distance: {distance:.2f}px")

                        # Update and draw the dynamic circle
                        circle_center = (
                            int((thumb_tip.x + index_tip.x) / 2 * frame.shape[1]),
                            int((thumb_tip.y + index_tip.y) / 2 * frame.shape[0]),
                        )
                        dynamic_circle.update(center=circle_center, radius=distance / 2)
                        dynamic_circle.draw(frame)

            # Draw the details card
            frame = details_card.draw(frame)

            # Display the frame
            cv2.imshow("Hand Tracking with Circle", frame)

            # Exit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    release_webcam(cap)


if __name__ == "__main__":
    main()