from flask import Flask, jsonify, request
from flask_cors import CORS
from vision_processor import detect_objects
from sound_classifier import classify_audio
import tempfile
import os
import base64
from faster_whisper import WhisperModel
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging for faster-whisper
logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.WARNING)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize Whisper model (using 'base' for good balance of speed/accuracy)
print("Loading Whisper model")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper model loaded")

# Initialize thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=2)

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
    Processes base64-encoded image and returns detected objects with bounding boxes.
    """
    try:
        # Check if JSON data was provided
        if not request.json or 'image' not in request.json:
            return jsonify({"error": "No image data provided"}), 400
        
        # Get base64-encoded image
        image_base64 = request.json['image']
        
        # Decode base64 to bytes
        image_data = base64.b64decode(image_base64)
        
        # Perform object detection
        result = detect_objects(image_data)
        
        return jsonify(result), 200
    
    except Exception as e:
        # Log the error and return a 500 response
        print(f"Error in vision endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.route('/hearing', methods=['POST'])
def hearing_mode():
    """
    Hearing endpoint. Accepts base64-encoded WAV audio and returns transcription and sound classification.
    Runs transcription and classification in parallel for responsiveness.
    """
    temp_path = None
    try:
        # Expect base64-encoded audio in JSON: { "audio": "...base64..." }
        if not request.json or 'audio' not in request.json:
            return jsonify({"error": "Missing audio data in request."}), 400

        audio_base64 = request.json['audio']
        try:
            audio_data = base64.b64decode(audio_base64)
        except Exception:
            return jsonify({"error": "Could not decode audio."}), 400

        if not audio_data or len(audio_data) == 0:
            return jsonify({"error": "Audio data is empty."}), 400

        print(f"[DEBUG] Received audio data: {len(audio_data)} bytes")
        
        # Save audio to temporary file (Whisper requires file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name
            temp_file.write(audio_data)
        
        print(f"[DEBUG] Saved to temp file: {temp_path}")
        
        # Run transcription and sound classification in parallel
        def do_transcription():
            """Transcribe audio with Whisper."""
            segments, info = whisper_model.transcribe(
                temp_path,
                beam_size=5,
                language=None,  # Auto-detect language
                vad_filter=True,  # Filter out non-speech
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            # Collect all transcribed text
            transcription_parts = []
            for segment in segments:
                transcription_parts.append(segment.text.strip())
            
            return " ".join(transcription_parts).strip(), info
        
        def do_sound_classification():
            """Classify environmental sounds."""
            return classify_audio(audio_data)
        
        # Submit both tasks to run in parallel
        transcription_future = executor.submit(do_transcription)
        classification_future = executor.submit(do_sound_classification)
        
        # Wait for both results
        full_transcription, info = transcription_future.result()
        classification_result = classification_future.result()
        
        print(f"[DEBUG] Transcription result: '{full_transcription}'")
        print(f"[DEBUG] Language: {info.language} (probability: {info.language_probability:.2f})")
        print(f"[DEBUG] Sound classification: {classification_result}")
        
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            temp_path = None
        
        # Build response with both transcription and alert info
        response = {
            "transcription": full_transcription,
            "alert": classification_result.get("alert"),
            "alert_type": classification_result.get("alert_type"),
            "alert_confidence": classification_result.get("confidence"),
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration
        }
        
        # Log if alert detected
        if classification_result.get("alert"):
            print(f"[ALERT] {classification_result.get('alert')} (confidence: {classification_result.get('confidence'):.2f})")
        
        return jsonify(response), 200
    
    except Exception as e:
        # Clean up temp file on error
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        
        print(f"Error in hearing endpoint: {str(e)}")
        return jsonify({"error": f"Transcription error: {str(e)}"}), 500

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
    print("Hearing endpoint: http://127.0.0.1:5000/hearing")
    # Use threaded=True for concurrent requests, debug=False to prevent model double-loading
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
