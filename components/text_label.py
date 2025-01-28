# components/text_label.py

import cv2

class TextLabel:
    """
    Displays labels anchored to specific points (landmarks) in the video feed.
    """
    def __init__(
        self, 
        label, 
        anchor, 
        offset=(0, 0), 
        text_color=(255, 255, 255), 
        font=cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale=0.8, 
        thickness=2
    ):
        """
        Initializes the TextLabel component.

        Args:
            label (str): The text to display.
            anchor (tuple): (x, y) coordinates to anchor the label.
            offset (tuple): (dx, dy) offset from the anchor position.
            text_color (tuple): BGR color tuple for the text.
            font (int): OpenCV font type.
            font_scale (float): Scale of the font.
            thickness (int): Thickness of the text.
        """
        self.label = label
        self.anchor = anchor
        self.offset = offset
        self.text_color = text_color
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness

    def draw(self, frame):
        """
        Draws the label on the provided video frame at the specified anchor and offset.

        Args:
            frame (numpy.ndarray): The video frame.
        """
        x, y = self.anchor
        offset_x, offset_y = self.offset
        position = (x + offset_x, y + offset_y)
        cv2.putText(
            frame, 
            self.label, 
            position, 
            self.font, 
            self.font_scale, 
            self.text_color, 
            thickness=self.thickness
        )