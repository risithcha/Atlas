import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from PIL import Image
import io
import numpy as np

# Global variables to cache the model
_model = None
_weights = None
_device = None

def _load_model():
    """
    Load the object detection model.
    Cached after first call for performance.
    """
    global _model, _weights, _device
    
    if _model is None:
        # Use GPU if available, otherwise CPU
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model and weights
        _weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        _model = ssdlite320_mobilenet_v3_large(weights=_weights)
        _model.to(_device)
        _model.eval()
        
        print(f"Model loaded successfully on device: {_device}")
    
    return _model, _weights, _device

def detect_objects(image_data):
    """
    Detect objects in an image using MobileNet-SSD.
    Returns detected objects with labels, confidence scores, and bounding boxes.
    Eventually will include OCR and scene description.
    """
    # Load model and prepare image
    model, weights, device = _load_model()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    # Preprocess image for model
    preprocess = weights.transforms()
    img_tensor = preprocess(image)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Run object detection
    with torch.no_grad():
        predictions = model(img_tensor)
    
    pred = predictions[0]
    categories = weights.meta['categories']
    detected_objects = []
    
    # Extract detection results
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    confidence_threshold = 0.5
    
    # Filter and format detected objects
    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence_threshold:
            x_min, y_min, x_max, y_max = box.astype(int).tolist()
            label_name = categories[label]
            
            detected_objects.append({
                'label': label_name,
                'confidence': float(score),
                'box': [x_min, y_min, x_max, y_max]
            })
    
    # Build response with detected objects
    # OCR and scene description to be added later
    response = {
        'objects': detected_objects,
        'ocr_text': '',
        'scene_description': ''
    }
    
    return response
