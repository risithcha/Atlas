import sys
import requests
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QStackedWidget
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QPalette, QColor, QImage, QPixmap


class StatusIndicator(QWidget):
    """A small circular status indicator widget."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.status = "disconnected"
        self.setFixedSize(20, 20)
        self.setToolTip("Checking backend connection...")
        
    def set_connected(self):
        """Set the indicator to connected state (green)."""
        self.status = "connected"
        self.setToolTip("Connected")
        self.update()
        
    def set_disconnected(self, error_message="Backend not running"):
        """Set the indicator to disconnected state (red)."""
        self.status = "disconnected"
        self.setToolTip(f"Error: {error_message}")
        self.update()
        
    def paintEvent(self, event):
        """Paint the status indicator circle."""
        from PyQt6.QtGui import QPainter
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if self.status == "connected":
            painter.setBrush(QColor(76, 175, 80))  # Green
        else:
            painter.setBrush(QColor(244, 67, 54))  # Red
            
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, 16, 16)


class ModeSelectionView(QWidget):
    """The initial view with mode selection buttons."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(30)
        
        # Title
        title = QLabel("Welcome to Atlas")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: #333;
            margin-bottom: 20px;
        """)
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Select an Assist Mode")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("""
            font-size: 18px;
            color: #666;
            margin-bottom: 30px;
        """)
        layout.addWidget(subtitle)
        
        # Vision Assist Mode Button
        self.vision_button = QPushButton("Vision Assist Mode")
        self.vision_button.setMinimumSize(QSize(300, 80))
        self.vision_button.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                font-weight: bold;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 20px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        layout.addWidget(self.vision_button)
        
        # Hearing Assist Mode Button
        self.hearing_button = QPushButton("Hearing Assist Mode")
        self.hearing_button.setMinimumSize(QSize(300, 80))
        self.hearing_button.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                font-weight: bold;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #2E7D32;
            }
        """)
        layout.addWidget(self.hearing_button)
        
        layout.addStretch()
        self.setLayout(layout)


class VisionModeWidget(QWidget):
    """Widget for Vision Assist Mode with live webcam feed."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = None
        self.timer = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))
        self.setPalette(palette)
        
        # Video display label
        self.video_label = QLabel()
        # Center both horizontally and vertically - not centering vertically YET
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setScaledContents(False)  # Disable to prevent distortion for object detection
        layout.addWidget(self.video_label)
        
        # Back button container
        button_container = QWidget()
        button_container.setStyleSheet("background-color: rgba(0, 0, 0, 180);")
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(20, 10, 20, 10)
        
        # Back button
        self.back_button = QPushButton("Back to Mode Selection")
        self.back_button.setMinimumSize(QSize(250, 50))
        self.back_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        button_layout.addStretch()
        button_layout.addWidget(self.back_button)
        button_layout.addStretch()
        
        layout.addWidget(button_container)
        self.setLayout(layout)
        
    def start_camera(self):
        """Start the webcam feed."""
        if self.camera is None:
            # Use DirectShow backend instead of MSMF to avoid frame grab errors
            self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            
            if not self.camera.isOpened():
                self.video_label.setText("Error: Could not access webcam")
                self.video_label.setStyleSheet("""
                    color: white;
                    font-size: 24px;
                    background-color: black;
                """)
                return
            
            # Set camera properties for better performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            # Set buffer size to 1 to minimize lag
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
        if self.timer is None:
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(33) 
            
    def stop_camera(self):
        """Stop the webcam feed."""
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
            
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            
        self.video_label.clear()
        
    def update_frame(self):
        """Capture and display a frame from the webcam."""
        if self.camera is None or not self.camera.isOpened():
            return
            
        ret, frame = self.camera.read()
        if ret:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get frame dimensions
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            # Convert to QImage
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Scale while maintaining aspect ratio
            scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.video_label.setPixmap(scaled_pixmap)
    
    def showEvent(self, event):
        """Called when widget is shown."""
        super().showEvent(event)
        self.start_camera()
        
    def hideEvent(self, event):
        """Called when widget is hidden."""
        super().hideEvent(event)
        self.stop_camera()


class HearingModePlaceholder(QWidget):
    """Placeholder widget for Hearing Assist Mode."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(232, 245, 233))
        self.setPalette(palette)
        
        # Label
        label = QLabel("Hearing Mode UI")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("""
            font-size: 36px;
            font-weight: bold;
            color: #2E7D32;
        """)
        layout.addWidget(label)
        
        # Back button
        self.back_button = QPushButton("Back to Mode Selection")
        self.back_button.setMinimumSize(QSize(250, 50))
        self.back_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px;
                margin-top: 30px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(self.back_button)
        
        self.setLayout(layout)


class BackendCommunicator:
    """Handles communication with the backend server."""
    
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
        
    def check_status(self):
        """
        Check the backend status by calling the /status endpoint.
        
        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            response = requests.get(
                f"{self.base_url}/status",
                timeout=2
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    return True, "Connected"
                else:
                    return False, "Unexpected response from backend"
            else:
                return False, f"Backend returned status {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "Backend not running"
        except requests.exceptions.Timeout:
            return False, "Connection timeout"
        except Exception as e:
            return False, f"Error: {str(e)}"


class AtlasMainWindow(QMainWindow):
    """Main application window for Atlas."""
    
    def __init__(self):
        super().__init__()
        self.backend = BackendCommunicator()
        self.init_ui()
        self.check_backend_status()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Atlas")
        self.setMinimumSize(800, 600)
        
        # Create central widget with stacked layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top bar with status indicator
        top_bar = QWidget()
        top_bar.setStyleSheet("background-color: #f5f5f5; border-bottom: 1px solid #ddd;")
        top_bar.setFixedHeight(40)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(10, 10, 10, 10)
        
        top_bar_layout.addStretch()
        
        # Status indicator
        self.status_indicator = StatusIndicator()
        top_bar_layout.addWidget(self.status_indicator)
        
        main_layout.addWidget(top_bar)
        
        # Stacked widget for different views
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # Create views
        self.mode_selection_view = ModeSelectionView()
        self.vision_mode_view = VisionModeWidget()
        self.hearing_mode_view = HearingModePlaceholder()
        
        # Add views to stacked widget
        self.stacked_widget.addWidget(self.mode_selection_view)
        self.stacked_widget.addWidget(self.vision_mode_view)
        self.stacked_widget.addWidget(self.hearing_mode_view)
        
        # Connect signals
        self.mode_selection_view.vision_button.clicked.connect(self.show_vision_mode)
        self.mode_selection_view.hearing_button.clicked.connect(self.show_hearing_mode)
        self.vision_mode_view.back_button.clicked.connect(self.show_mode_selection)
        self.hearing_mode_view.back_button.clicked.connect(self.show_mode_selection)
        
    def check_backend_status(self):
        """Check the backend status and update the indicator."""
        success, message = self.backend.check_status()
        
        if success:
            self.status_indicator.set_connected()
        else:
            self.status_indicator.set_disconnected(message)
    
    def show_vision_mode(self):
        """Switch to the Vision Assist Mode view."""
        self.stacked_widget.setCurrentWidget(self.vision_mode_view)
        
    def show_hearing_mode(self):
        """Switch to the Hearing Assist Mode view."""
        self.stacked_widget.setCurrentWidget(self.hearing_mode_view)
        
    def show_mode_selection(self):
        """Switch back to the Mode Selection view."""
        self.stacked_widget.setCurrentWidget(self.mode_selection_view)


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = AtlasMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
