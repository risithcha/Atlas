import sys
import requests
import cv2
import numpy as np
import base64
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, QStackedWidget,
    QTextEdit, QGroupBox, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QPalette, QColor, QImage, QPixmap
from data_overlay import OverlayLabel


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
        self.analysis_timer = None
        self.current_frame = None
        self.backend_url = "http://127.0.0.1:5000"
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Set background color
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))
        self.setPalette(palette)
        
        # Create a splitter for resizable layout between video and sidebar
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setStyleSheet("QSplitter::handle { background-color: #333; }")
        
        # Video container widget
        video_container = QWidget()
        video_container.setStyleSheet("background-color: black;")
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)
        
        # Video display label with overlay capability
        self.video_label = OverlayLabel(self)
        # Center both horizontally and vertically
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setScaledContents(False)  # Disable to prevent distortion for object detection
        self.video_label.setMinimumSize(480, 360)  # Ensure minimum visibility
        video_layout.addWidget(self.video_label, 1)  # Stretch factor 1 for maximum space
        
        # Add video container to splitter
        self.splitter.addWidget(video_container)
        
        # Create sidebar for text information
        sidebar = QWidget()
        sidebar.setMinimumWidth(250)
        sidebar.setMaximumWidth(400)
        sidebar.setStyleSheet("background-color: #1a1a1a;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(10)
        
        # OCR Text Group
        ocr_group = QGroupBox("Detected Text (OCR)")
        ocr_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #4CAF50;
                border: 1px solid #4CAF50;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        ocr_layout = QVBoxLayout(ocr_group)
        
        self.ocr_text_display = QTextEdit()
        self.ocr_text_display.setReadOnly(True)
        self.ocr_text_display.setPlaceholderText("No text detected yet...")
        self.ocr_text_display.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        self.ocr_text_display.setMinimumHeight(120)
        ocr_layout.addWidget(self.ocr_text_display)
        
        sidebar_layout.addWidget(ocr_group)
        
        # Scene Description Group
        scene_group = QGroupBox("Scene Description")
        scene_group.setStyleSheet("""
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #2196F3;
                border: 1px solid #2196F3;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        scene_layout = QVBoxLayout(scene_group)
        
        self.scene_description_display = QTextEdit()
        self.scene_description_display.setReadOnly(True)
        self.scene_description_display.setPlaceholderText("Analyzing scene...")
        self.scene_description_display.setStyleSheet("""
            QTextEdit {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 13px;
            }
        """)
        self.scene_description_display.setMinimumHeight(150)
        scene_layout.addWidget(self.scene_description_display)
        
        sidebar_layout.addWidget(scene_group)
        sidebar_layout.addStretch()  # Push content to top
        
        # Add sidebar to splitter
        self.splitter.addWidget(sidebar)
        
        # Set initial splitter sizes (70% video, 30% sidebar)
        self.splitter.setSizes([700, 300])
        self.splitter.setStretchFactor(0, 7)
        self.splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(self.splitter, 1)
        
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
        
        main_layout.addWidget(button_container)
        self.setLayout(main_layout)
        
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
        
        # Start analysis timer for backend communication
        if self.analysis_timer is None:
            self.analysis_timer = QTimer()
            self.analysis_timer.timeout.connect(self.send_frame_for_analysis)
            self.analysis_timer.start(1500)  # Send frame every 1.5 seconds 
            
    def stop_camera(self):
        """Stop the webcam feed."""
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
        
        if self.analysis_timer is not None:
            self.analysis_timer.stop()
            self.analysis_timer = None
            
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
            # Store current frame for backend analysis
            self.current_frame = frame
            
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
    
    def send_frame_for_analysis(self):
        """Send the current frame to backend for object detection analysis."""
        if self.current_frame is None:
            return
        
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', self.current_frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send POST request to backend
            response = requests.post(
                f"{self.backend_url}/vision",
                json={"image": frame_base64},
                timeout=2
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # Update overlay with detection data and frame size for coordinate mapping
                frame_height, frame_width = self.current_frame.shape[:2]
                self.video_label.update_data(response_data, frame_size=(frame_width, frame_height))
                
                # Update OCR text display
                ocr_text = response_data.get('ocr_text', '')
                if ocr_text and ocr_text.strip():
                    self.ocr_text_display.setPlainText(ocr_text)
                else:
                    self.ocr_text_display.setPlainText("No text detected in current view.")
                
                # Update scene description display
                scene_description = response_data.get('scene_description', '')
                if scene_description and scene_description.strip():
                    self.scene_description_display.setPlainText(scene_description)
                else:
                    # If no scene description, create a basic one from detected objects
                    detections = response_data.get('detections', [])
                    if detections:
                        object_names = [d.get('label', 'object') for d in detections]
                        unique_objects = list(set(object_names))
                        basic_description = f"Detected objects: {', '.join(unique_objects)}"
                        self.scene_description_display.setPlainText(basic_description)
                    else:
                        self.scene_description_display.setPlainText("Analyzing scene...")
        
        except requests.exceptions.RequestException as e:
            # Silently handle backend communication errors
            pass
    
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
        
        # Set up periodic backend status checking
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_backend_status)
        self.status_timer.start(3000)  # Check every 3 seconds
        
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
