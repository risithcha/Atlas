from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPainter, QPen, QFont, QColor
from PyQt6.QtCore import Qt


class OverlayLabel(QLabel):
    """Custom QLabel widget that draws ML detection data as overlay on video frames."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.detection_data = None
        
    def update_data(self, new_data):
        """
        Update the detection data and trigger a repaint.
        """
        self.detection_data = new_data
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """
        Override paintEvent to draw video frame and detection overlays.
        Called automatically when update() is triggered.
        """
        # Draw the parent's content (the video frame)
        super().paintEvent(event)
        
        # Skip overlay if no detection data
        if not self.detection_data or not self.detection_data.get('objects'):
            return
        
        # Create painter for drawing overlays
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Set up pen for bounding boxes
        pen = QPen(QColor(0, 255, 0))  # Green color
        pen.setWidth(3)
        painter.setPen(pen)
        
        # Set up font for labels
        font = QFont('Arial', 12, QFont.Weight.Bold)
        painter.setFont(font)
        
        # Draw each detected object
        for obj in self.detection_data['objects']:
            # Extract bounding box coordinates
            box = obj.get('box', [])
            if len(box) != 4:
                continue
            
            x_min, y_min, x_max, y_max = box
            
            # Draw bounding box rectangle
            width = x_max - x_min
            height = y_max - y_min
            painter.drawRect(x_min, y_min, width, height)
            
            # Prepare label text with confidence
            label = obj.get('label', 'Unknown')
            confidence = obj.get('confidence', 0.0)
            text = f"{label} {confidence:.0%}"
            
            # Draw text background for readability
            text_rect = painter.fontMetrics().boundingRect(text)
            text_bg_rect = text_rect.adjusted(-4, -2, 4, 2)
            text_bg_rect.moveTopLeft(painter.fontMetrics().boundingRect(x_min, y_min, 0, 0, 0, text).topLeft())
            text_bg_rect.translate(x_min, y_min - text_bg_rect.height())
            
            painter.fillRect(text_bg_rect, QColor(0, 0, 0, 180))  # Semi-transparent black
            
            # Draw label text above the box
            painter.setPen(QColor(255, 255, 255))  # White text
            painter.drawText(x_min, y_min - 5, text)
            
            # Reset pen for next box
            painter.setPen(pen)
        
        painter.end()
