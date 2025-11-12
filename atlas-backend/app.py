from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

@app.route('/status', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the server is running.
    Returns a simple status message.
    """
    return jsonify({"status": "ok"}), 200

@app.route('/vision', methods=['POST'])
def vision_mode():
    """
    Mock Vision Mode endpoint.
    Eventually will process camera frames for object detection, OCR, and scene description.
    For now, returns hard-coded mock data.
    """
    # Mock response data
    mock_response = {
        "objects": [
            {
                "label": "cup",
                "confidence": 0.92,
                "box": [150, 220, 250, 320]
            },
            {
                "label": "keyboard",
                "confidence": 0.88,
                "box": [100, 400, 600, 550]
            }
        ],
        "ocr_text": "Main Menu",
        "scene_description": "A desk with a cup and a keyboard."
    }
    
    return jsonify(mock_response), 200

@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 errors with a JSON response.
    """
    return jsonify({"error": "Endpoint not found"}), 404

if __name__ == '__main__':
    print("Atlas Backend Server Starting...")
    print("Server running at: http://127.0.0.1:5000")
    print("Health check: http://127.0.0.1:5000/status")
    print("Vision endpoint: http://127.0.0.1:5000/vision")
    app.run(host='127.0.0.1', port=5000, debug=True)
