"""
Atlas MobileNet-SSD TFLite Conversion Script.
Downloads, converts, and optimizes object detection models for mobile deployment.
"""

import os
import sys
import tarfile
import tempfile
import urllib.request
import shutil
from pathlib import Path

import tensorflow as tf
import numpy as np

# Model URLs from TensorFlow 2 Detection Zoo (SavedModel format)
MODEL_CONFIGS = {
    "ssd_mobilenet_v2_320": {
        "url": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz",
        "input_size": 320,
        "description": "SSD MobileNet V2 320x320 - Fast, smaller model"
    },
    "ssd_mobilenet_v2_fpnlite_320": {
        "url": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz",
        "input_size": 320,
        "description": "SSD MobileNet V2 FPNLite 320x320 - Better accuracy"
    },
    "ssd_mobilenet_v2_fpnlite_640": {
        "url": "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz",
        "input_size": 640,
        "description": "SSD MobileNet V2 FPNLite 640x640 - Higher resolution"
    }
}

# Default model selection
DEFAULT_MODEL = "ssd_mobilenet_v2_320"

# Output configuration
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_FILENAME = "atlas_mobilenet_quant.tflite"
LABELS_FILENAME = "coco_labels.txt"

# COCO dataset class labels (91 classes - includes background and unused IDs)
# Will be bundled into the TFLite model as metadata
COCO_LABELS = [
    "???",  # 0: background/unknown
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "???",  # 12: not used in COCO
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "???",  # 26: not used
    "backpack",
    "umbrella",
    "???",  # 29: not used
    "???",  # 30: not used
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "???",  # 45: not used
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "???",  # 66: not used
    "dining table",
    "???",  # 68: not used
    "???",  # 69: not used
    "toilet",
    "???",  # 71: not used
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "???",  # 83: not used
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

def get_file_size_mb(filepath: Path) -> float:
    """Get file size in megabytes."""
    return filepath.stat().st_size / (1024 * 1024)


def get_dir_size_mb(dirpath: Path) -> float:
    """Get total directory size in megabytes."""
    total_size = 0
    for item in dirpath.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
    return total_size / (1024 * 1024)


def download_with_progress(url: str, dest: Path) -> None:
    """Download file with progress indication."""
    print(f"Downloading from: {url}")
    print(f"Destination: {dest}")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if count % 100 == 0:  # Simple progress indicator
            print(f"\rProgress: {min(percent, 100)}%", end="", flush=True)
    
    urllib.request.urlretrieve(url, dest, reporthook=progress_hook)
    print("\nDownload complete!")


def extract_tarfile(tar_path: Path, extract_to: Path) -> Path:
    """Extract tar.gz and return extracted directory path."""
    print(f"Extracting {tar_path.name}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_to)
    
    # Find the extracted directory
    extracted_dirs = [d for d in extract_to.iterdir() if d.is_dir()]
    if not extracted_dirs:
        raise RuntimeError("No directory found in extracted tar file")
    
    print(f"Extracted to: {extracted_dirs[0]}")
    return extracted_dirs[0]


def create_labels_file(labels: list, output_path: Path) -> None:
    """Create labels.txt file for TFLite metadata."""
    with open(output_path, "w") as f:
        for label in labels:
            f.write(f"{label}\n")
    print(f"Labels file created: {output_path}")

def download_model(model_key: str = DEFAULT_MODEL, cache_dir: Path = None) -> Path:
    """
    Download pre-trained MobileNet V2 SSD model.
    Cached after first download for performance.
    """
    if cache_dir is None:
        cache_dir = OUTPUT_DIR / "model_cache"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    config = MODEL_CONFIGS[model_key]
    url = config["url"]
    
    # Check if already cached
    model_name = url.split("/")[-1].replace(".tar.gz", "")
    model_dir = cache_dir / model_name
    saved_model_dir = model_dir / "saved_model"
    
    if saved_model_dir.exists():
        print(f"Model already cached: {saved_model_dir}")
        return saved_model_dir
    
    # Download model
    tar_path = cache_dir / f"{model_name}.tar.gz"
    
    if not tar_path.exists():
        download_with_progress(url, tar_path)
    
    # Extract model files
    extracted_dir = extract_tarfile(tar_path, cache_dir)
    
    # Locate saved_model directory
    saved_model_dir = extracted_dir / "saved_model"
    if not saved_model_dir.exists():
        # Some models store it directly in root
        if (extracted_dir / "saved_model.pb").exists():
            saved_model_dir = extracted_dir
    
    if not saved_model_dir.exists():
        raise RuntimeError(f"Could not find saved_model in {extracted_dir}")
    
    return saved_model_dir

def convert_to_tflite(
    saved_model_dir: Path,
    output_path: Path,
    enable_quantization: bool = True,
    representative_dataset_size: int = 100,
) -> bytes:
    """
    Convert SavedModel to TFLite with dynamic range quantization.
    Reduces model size by ~4x (float32 -> int8 weights).
    """
    print(f"\nLoading SavedModel from: {saved_model_dir}")
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    
    if enable_quantization:
        print("\nApplying dynamic range quantization (float32 -> int8)...")
        
        # Enable weight quantization (keeps activations as float for compatibility)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Allow TF ops fallback for object detection operations
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # Convert model (takes a few minutes)
    print("\nConverting model...")
    tflite_model = converter.convert()
    
    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"\nTFLite model saved to: {output_path}")
    print(f"Model size: {get_file_size_mb(output_path):.2f} MB")
    
    return tflite_model

def add_metadata(
    tflite_model_path: Path,
    labels_path: Path,
    output_path: Path,
    input_size: int = 320,
    model_description: str = "Atlas Object Detection Model"
) -> None:
    """
    Add metadata to TFLite model for TFLite Task Library compatibility.
    Bundles labels, input specs, and normalization parameters into the model.
    """
    from tflite_support.metadata_writers import object_detector
    from tflite_support.metadata_writers import writer_utils
    
    # Load model
    model_buffer = writer_utils.load_file(str(tflite_model_path))
    
    # MobileNet normalization: [0,255] -> [-1,1] using mean=127.5, std=127.5
    _INPUT_NORM_MEAN = 127.5
    _INPUT_NORM_STD = 127.5
    
    # Create metadata writer
    writer = object_detector.MetadataWriter.create_for_inference(
        model_buffer,
        input_norm_mean=[_INPUT_NORM_MEAN],
        input_norm_std=[_INPUT_NORM_STD],
        label_file_paths=[str(labels_path)]
    )
    
    # Display generated metadata (first 1000 chars)
    metadata_json = writer.get_metadata_json()
    print("\nGenerated Metadata:")
    print(metadata_json[:1000] + "..." if len(metadata_json) > 1000 else metadata_json)
    
    # Save model with embedded metadata
    writer_utils.save_file(writer.populate(), str(output_path))
    print(f"\nModel with metadata saved to: {output_path}")

def verify_model(tflite_path: Path) -> None:
    """
    Verify TFLite model by loading and running test inference.
    """
    # Load model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Get tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\nInput Tensors:")
    for i, detail in enumerate(input_details):
        print(f"  [{i}] {detail['name']}")
        print(f"      Shape: {detail['shape']}")
        print(f"      Type: {detail['dtype']}")
    
    print("\nOutput Tensors:")
    for i, detail in enumerate(output_details):
        print(f"  [{i}] {detail['name']}")
        print(f"      Shape: {detail['shape']}")
        print(f"      Type: {detail['dtype']}")
    
    # Run test inference with random data
    print("\nRunning test inference...")
    input_shape = input_details[0]['shape']
    test_input = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], test_input.astype(np.float32))
    interpreter.invoke()
    
    # Check output shapes
    print("Inference successful! Output shapes:")
    for detail in output_details:
        output = interpreter.get_tensor(detail['index'])
        print(f"  {detail['name']}: {output.shape}")
    
    print("\nModel verification passed.")

def generate_size_report(
    original_dir: Path,
    tflite_path: Path,
    model_key: str
) -> dict:
    """
    Generate size comparison report between original and optimized models.
    """
    original_size = get_dir_size_mb(original_dir)
    tflite_size = get_file_size_mb(tflite_path)
    reduction_ratio = original_size / tflite_size if tflite_size > 0 else 0
    reduction_percent = (1 - tflite_size / original_size) * 100 if original_size > 0 else 0
    
    report = {
        "model": model_key,
        "original_size_mb": round(original_size, 2),
        "tflite_size_mb": round(tflite_size, 2),
        "reduction_ratio": round(reduction_ratio, 2),
        "reduction_percent": round(reduction_percent, 1),
    }
    
    print(f"""
Model: {model_key}
{"-" * 40}
Original SavedModel size:  {original_size:>8.2f} MB
TFLite optimized size:     {tflite_size:>8.2f} MB
{"-" * 40}
Size reduction:            {reduction_ratio:>8.2f}x
Reduction percentage:      {reduction_percent:>8.1f}%

NUMBERS:
 - Original: {original_size:.1f} MB -> Optimized: {tflite_size:.1f} MB
 - Achieved {reduction_ratio:.1f}x compression ({reduction_percent:.0f}% smaller)
""")
    
    return report

def main():
    """Main conversion pipeline."""
    print("\nATLAS MODEL ENGINEERING - TFLite Conversion Pipeline")
    print("Converting MobileNet V2 SSD for mobile deployment")
    
    # Setup
    model_key = DEFAULT_MODEL
    config = MODEL_CONFIGS[model_key]
    
    print(f"Selected Model: {model_key}")
    print(f"Description: {config['description']}")
    print(f"Input Size: {config['input_size']}x{config['input_size']}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    labels_path = OUTPUT_DIR / LABELS_FILENAME
    tflite_path = OUTPUT_DIR / OUTPUT_FILENAME.replace(".tflite", "_nometadata.tflite")
    final_path = OUTPUT_DIR / OUTPUT_FILENAME
    
    # Step 1: Create labels file
    print("\n" + "=" * 60)
    print("STEP 1: CREATE LABELS")
    print("=" * 60)
    create_labels_file(COCO_LABELS, labels_path)
    
    # Step 2: Download model
    print("\n" + "=" * 60)
    print("STEP 2: DOWNLOAD MODEL")
    print("=" * 60)
    saved_model_dir = download_model(model_key)
    
    # Step 3: Convert to TFLite
    print("\n" + "=" * 60)
    print("STEP 3: CONVERT TO TFLITE")
    print("=" * 60)
    convert_to_tflite(
        saved_model_dir=saved_model_dir,
        output_path=tflite_path,
        enable_quantization=True
    )
    
    # Step 4: Add metadata
    print("\n" + "=" * 60)
    print("STEP 4: ADD METADATA")
    print("=" * 60)
    add_metadata(
        tflite_model_path=tflite_path,
        labels_path=labels_path,
        output_path=final_path,
        input_size=config['input_size'],
        model_description=f"Atlas Object Detection - {config['description']}"
    )
    
    # Step 5: Verify model
    print("\n" + "=" * 60)
    print("STEP 5: VERIFY MODEL")
    print("=" * 60)
    verify_model(final_path)
    
    # Step 6: Generate report
    print("\n" + "=" * 60)
    print("STEP 6: SIZE COMPARISON")
    print("=" * 60)
    report = generate_size_report(
        original_dir=saved_model_dir.parent,
        tflite_path=final_path,
        model_key=model_key
    )
    
    # Save report to disk
    import json
    report_path = OUTPUT_DIR / "conversion_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    
    # Final summary
    print("\nCONVERSION COMPLETE")
    print(f"Output files in: {OUTPUT_DIR}")
    print(f"  - {OUTPUT_FILENAME} - Main TFLite model with metadata")
    print(f"  - {LABELS_FILENAME} - COCO class labels")
    print(f"  - conversion_report.json - Size comparison data")
    
    return report


if __name__ == "__main__":
    try:
        report = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nConversion cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
