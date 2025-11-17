from flask import Flask, jsonify, request
from flask_cors import CORS
from vision_processor import detect_objects

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
    Vision Mode endpoint.
    Processes image file and returns detected objects with bounding boxes.
    """
    try:
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        
        # Check if the file is empty
        if image_file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Read the image data
        image_data = image_file.read()
        
        # Perform object detection
        result = detect_objects(image_data)
        
        return jsonify(result), 200
    
    except Exception as e:
        # Log the error and return a 500 response
        print(f"Error in vision endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

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
