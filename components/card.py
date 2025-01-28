# components/card.py

import cv2

class Card:
    """
    A dynamic, reusable text display component for overlaying information on the video feed.
    """
    def __init__(
        self, 
        position, 
        width, 
        box_color=(0, 0, 0), 
        alpha=0.5, 
        text_color=(255, 255, 255), 
        font=cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale=0.8,
        thickness=2
    ):
        """
        Initializes the Card component.

        Args:
            position (tuple): (x, y) coordinates for the top-left corner of the card.
            width (int): Width of the card in pixels.
            box_color (tuple): BGR color tuple for the background of the card.
            alpha (float): Transparency level of the card (0 = transparent, 1 = opaque).
            text_color (tuple): BGR color tuple for the text.
            font (int): OpenCV font type.
            font_scale (float): Scale of the font.
            thickness (int): Thickness of the text.
        """
        self.position = position
        self.width = width
        self.box_color = box_color
        self.alpha = alpha
        self.text_color = text_color
        self.font = font
        self.font_scale = font_scale
        self.thickness = thickness
        self.lines = []

    def add_line(self, text):
        """
        Adds a line of text to the card.

        Args:
            text (str): The text to add.
        """
        self.lines.append(text)

    def clear(self):
        """
        Clears all text from the card.
        """
        self.lines = []

    def draw(self, frame):
        """
        Draws the card on the provided video frame.

        Args:
            frame (numpy.ndarray): The video frame.

        Returns:
            numpy.ndarray: The frame with the card overlay.
        """
        if not self.lines:
            return frame

        overlay = frame.copy()
        x, y = self.position

        # Calculate dynamic height based on the number of lines
        line_height = int(self.font_scale * 30)
        height = line_height * len(self.lines) + 20

        # Draw the semi-transparent box
        cv2.rectangle(overlay, (x, y), (x + self.width, y + height), self.box_color, -1)
        cv2.addWeighted(overlay, self.alpha, frame, 1 - self.alpha, 0, frame)

        # Draw each line of text
        for i, line in enumerate(self.lines):
            text_y = y + line_height * (i + 1)
            cv2.putText(
                frame, 
                line, 
                (x + 10, text_y), 
                self.font, 
                self.font_scale, 
                self.text_color, 
                thickness=self.thickness
            )

        return frame