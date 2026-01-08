"""
Text Renderer for Korean Support
Uses Pillow (PIL) to draw TrueType fonts on OpenCV images
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

class TextRenderer:
    def __init__(self, font_path='/System/Library/Fonts/Supplemental/AppleGothic.ttf', font_size=32):
        self.font_path = font_path
        self.font_size = font_size
        try:
            self.font = ImageFont.truetype(font_path, font_size)
            print(f"✅ Loaded font: {font_path}")
        except IOError:
            print("⚠️ Font not found, falling back to default")
            self.font = ImageFont.load_default()

    def put_text(self, img, text, x, y, color=(255, 255, 255)):
        """
        Draw text with Korean support
        Args:
            img: OpenCV image (BGR)
            text: String to draw
            x, y: Position
            color: Tuple (B, G, R)
        """
        # Convert BGR to RGB (PIL uses RGB)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Color conversion (BGR -> RGB)
        rgb_color = (color[2], color[1], color[0])
        
        draw.text((x, y), text, font=self.font, fill=rgb_color)
        
        # Convert back to BGR and numpy array
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def get_text_size(self, text):
        """Get width and height of text"""
        bbox = self.font.getbbox(text)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width, height
