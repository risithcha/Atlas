// TensorFlow Lite Object Detection Output Decoder
// Converts raw tensor outputs from TFLite SSD models into structured detections.
// 
// Model outputs:
// - Output 0 (Boxes):   [1, num_detections, 4] - Bounding boxes [y1, x1, y2, x2] (0-1 normalized)
// - Output 1 (Classes): [1, num_detections] - Class IDs (0-90 for COCO)
// - Output 2 (Scores):  [1, num_detections] - Confidence scores (0.0-1.0)
// - Output 3 (Count):   [1] - Number of valid detections (optional)

// COCO class labels - 91 classes mapped from model output IDs (0-90)
// ID 0 is background/unknown
export const COCO_CLASSES: Record<number, string> = {
  0: '???',
  1: 'person',
  2: 'bicycle',
  3: 'car',
  4: 'motorcycle',
  5: 'airplane',
  6: 'bus',
  7: 'train',
  8: 'truck',
  9: 'boat',
  10: 'traffic light',
  11: 'fire hydrant',
  12: '???',
  13: 'stop sign',
  14: 'parking meter',
  15: 'bench',
  16: 'bird',
  17: 'cat',
  18: 'dog',
  19: 'horse',
  20: 'sheep',
  21: 'cow',
  22: 'elephant',
  23: 'bear',
  24: 'zebra',
  25: 'giraffe',
  26: '???',
  27: 'backpack',
  28: 'umbrella',
  29: '???',
  30: '???',
  31: 'handbag',
  32: 'tie',
  33: 'suitcase',
  34: 'frisbee',
  35: 'skis',
  36: 'snowboard',
  37: 'sports ball',
  38: 'kite',
  39: 'baseball bat',
  40: 'baseball glove',
  41: 'skateboard',
  42: 'surfboard',
  43: 'tennis racket',
  44: 'bottle',
  45: '???',
  46: 'wine glass',
  47: 'cup',
  48: 'fork',
  49: 'knife',
  50: 'spoon',
  51: 'bowl',
  52: 'banana',
  53: 'apple',
  54: 'sandwich',
  55: 'orange',
  56: 'broccoli',
  57: 'carrot',
  58: 'hot dog',
  59: 'pizza',
  60: 'donut',
  61: 'cake',
  62: 'chair',
  63: 'couch',
  64: 'potted plant',
  65: 'bed',
  66: '???',
  67: 'dining table',
  68: '???',
  69: '???',
  70: 'toilet',
  71: '???',
  72: 'tv',
  73: 'laptop',
  74: 'mouse',
  75: 'remote',
  76: 'keyboard',
  77: 'cell phone',
  78: 'microwave',
  79: 'oven',
  80: 'toaster',
  81: 'sink',
  82: 'refrigerator',
  83: '???',
  84: 'book',
  85: 'clock',
  86: 'vase',
  87: 'scissors',
  88: 'teddy bear',
  89: 'hair drier',
  90: 'toothbrush',
};

// Bounding box with normalized coordinates (0-1)
export interface BoundingBox {
  top: number;     // y1
  left: number;    // x1
  bottom: number;  // y2
  right: number;   // x2
}

// Single detection result from the model
export interface Detection {
  label: string;      // e.g., "person", "car"
  classId: number;    // 0-90 for COCO
  score: number;      // 0.0-1.0
  box: BoundingBox;   // normalized 0-1
}

// Raw tensor outputs from TFLite model
export interface TFLiteOutputs {
  boxes: Float32Array | number[];    // [1, num_detections, 4] flattened
  classes: Float32Array | number[];  // [1, num_detections]
  scores: Float32Array | number[];   // [1, num_detections]
  count?: number;                    // Optional: number of valid detections
}

// Decoder configuration options
export interface DecoderOptions {
  threshold?: number;                      // Minimum confidence (default: 0.5)
  maxDetections?: number;                  // Max results to return (default: 10)
  classLabels?: Record<number, string>;    // Custom class labels (default: COCO_CLASSES)
  imageWidth?: number;                     // For pixel coordinate conversion
  imageHeight?: number;                    // For pixel coordinate conversion
}

// Decode raw TFLite detection outputs into structured Detection objects.
// Filters by confidence threshold and maps class IDs to labels.
export function decodePredictions(
  outputs: TFLiteOutputs,
  options: DecoderOptions = {}
): Detection[] {
  const {
    threshold = 0.5,
    maxDetections = 10,
    classLabels = COCO_CLASSES,
  } = options;

  const { boxes, classes, scores } = outputs;
  const detections: Detection[] = [];

  // Limit to max detections or count provided by model
  const numDetections = Math.min(
    outputs.count ?? maxDetections,
    maxDetections,
    scores.length
  );

  for (let i = 0; i < numDetections; i++) {
    const score = scores[i];

    // Skip detections below threshold
    if (score < threshold) {
      continue;
    }

    // Get class ID and label
    const classId = Math.round(classes[i]);
    const label = classLabels[classId] ?? `class_${classId}`;

    // Skip unknown/background classes
    if (label === '???' || classId === 0) {
      continue;
    }

    // Extract bounding box: [y1, x1, y2, x2]
    const boxIndex = i * 4;
    const box: BoundingBox = {
      top: boxes[boxIndex],      // y1
      left: boxes[boxIndex + 1], // x1
      bottom: boxes[boxIndex + 2], // y2
      right: boxes[boxIndex + 3],  // x2
    };

    detections.push({
      label,
      classId,
      score,
      box,
    });
  }

  // Sort by confidence score (highest first)
  detections.sort((a, b) => b.score - a.score);

  return detections;
}

// Convert normalized box coordinates (0-1) to pixel coordinates
export function toPixelCoordinates(
  box: BoundingBox,
  imageWidth: number,
  imageHeight: number
): BoundingBox {
  return {
    top: Math.round(box.top * imageHeight),
    left: Math.round(box.left * imageWidth),
    bottom: Math.round(box.bottom * imageHeight),
    right: Math.round(box.right * imageWidth),
  };
}

// Get center point of a bounding box
export function getBoxCenter(box: BoundingBox): { x: number; y: number } {
  return {
    x: (box.left + box.right) / 2,
    y: (box.top + box.bottom) / 2,
  };
}

// Calculate bounding box area (useful for size filtering)
export function getBoxArea(box: BoundingBox): number {
  const width = Math.abs(box.right - box.left);
  const height = Math.abs(box.bottom - box.top);
  return width * height;
}

// Format detection as readable string: "Person detected at [x, y] (87% confidence)"
export function formatDetection(
  detection: Detection,
  imageWidth?: number,
  imageHeight?: number
): string {
  const { label, score, box } = detection;
  const confidencePercent = Math.round(score * 100);

  if (imageWidth && imageHeight) {
    const pixelBox = toPixelCoordinates(box, imageWidth, imageHeight);
    const center = getBoxCenter(pixelBox);
    return `${label} detected at [${Math.round(center.x)}, ${Math.round(center.y)}] (${confidencePercent}% confidence)`;
  } else {
    const center = getBoxCenter(box);
    return `${label} detected at [${center.x.toFixed(2)}, ${center.y.toFixed(2)}] (${confidencePercent}% confidence)`;
  }
}

// Decode from raw output array [boxes, classes, scores, count?]
// Convenience function for models that return flattened outputs
export function decodeFromRawArrays(
  rawOutputs: (Float32Array | number[])[],
  options: DecoderOptions = {}
): Detection[] {
  // Standard output order: [boxes, classes, scores, count?]
  
  if (rawOutputs.length < 3) {
    console.warn('Expected at least 3 output tensors (boxes, classes, scores)');
    return [];
  }

  const outputs: TFLiteOutputs = {
    boxes: rawOutputs[0],
    classes: rawOutputs[1],
    scores: rawOutputs[2],
    count: rawOutputs[3]?.[0],
  };

  return decodePredictions(outputs, options);
}

// Filter detections by class labels (e.g., ["person", "car"])
export function filterByClass(
  detections: Detection[],
  allowedLabels: string[]
): Detection[] {
  const labelSet = new Set(allowedLabels.map(l => l.toLowerCase()));
  return detections.filter(d => labelSet.has(d.label.toLowerCase()));
}

// Filter detections by minimum bounding box area (normalized 0-1)
// Use to remove noise from very small detections
export function filterByMinArea(
  detections: Detection[],
  minArea: number
): Detection[] {
  return detections.filter(d => getBoxArea(d.box) >= minArea);
}

// Default export for convenience
export default {
  decodePredictions,
  decodeFromRawArrays,
  toPixelCoordinates,
  getBoxCenter,
  getBoxArea,
  formatDetection,
  filterByClass,
  filterByMinArea,
  COCO_CLASSES,
};
