import os
import sys
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: Cannot import ultralytics. Install with: pip install ultralytics")
    sys.exit(1)

def validate_file_path(file_path):
    """Check if file exists and has appropriate extension"""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist!")
        return False
    
    # Allowed extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', 
                       '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    file_ext = Path(file_path).suffix.lower()
    if file_ext not in valid_extensions:
        print(f"Error: Unsupported file extension: {file_ext}")
        print(f"Supported extensions: {', '.join(valid_extensions)}")
        return False
    
    return True

def is_video_file(file_path):
    """Check if file is a video"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    return Path(file_path).suffix.lower() in video_extensions

def process_file(model, file_path, output_dir, confidence=0.5):
    print(f"Processing file: {file_path}")

    results = model.predict(
        source=file_path,
        conf=confidence,
        save=True,
        project=output_dir,
        name="detection",
        exist_ok=True
    )
    
    total_detections = 0
    for result in results:
        if result.boxes is not None:
            num_detections = len(result.boxes)
            total_detections += num_detections
            if total_detections <= num_detections:  
                print(f"Detected {num_detections} objects:")
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[class_id]
                    print(f"  - {class_name}: {conf:.2f}")
    
    if total_detections == 0:
        print("No objects detected")
    else:
        print(f"Total detections: {total_detections} objects")
    
    print(f"Results saved in directory: {output_dir}/detection/")

def main():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--model", "-m", default=None, help="Path to custom YOLO model (e.g., best.pt)")
    parser.add_argument("--confidence", "-c", type=float, default=0.25, help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--output", "-o", default="./results", help="Output directory")
    args = parser.parse_args()
    
    file_path = input("Enter file path (image or video): ").strip()
  
    if not file_path:
        print("Error: No file path provided!")
        return
    
    if not validate_file_path(file_path):
        return
    
    if args.model and os.path.exists(args.model):
        model_name = args.model
        print(f"Using custom model: {model_name}")
    elif args.model:
        print(f"Custom model '{args.model}' not found!")
        custom_model = input("Enter custom model path (or press Enter for default yolov8n.pt): ").strip()
        if custom_model and os.path.exists(custom_model):
            model_name = custom_model
        else:
            model_name = "yolov8n.pt"
    else:
        custom_model = input("Enter custom model path (or press Enter for default yolov8n.pt): ").strip()
        if custom_model and os.path.exists(custom_model):
            model_name = custom_model
            print(f"Using custom model: {model_name}")
        else:
            model_name = "yolov8n.pt"
            if custom_model:
                print(f"Custom model '{custom_model}' not found, using default: {model_name}")
    
    try:
        print(f"Loading model: {model_name}")
        model = YOLO(model_name) 
        print("Model loaded successfully!")
        if model_name != "yolov10n.pt":
            print(f"Model classes: {list(model.names.values())}")
        
        process_file(model, file_path, args.output, args.confidence)
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return

if __name__ == "__main__":
    main()