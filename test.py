import argparse
from ultralytics import YOLO
import os
import glob

def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLO object detection with custom weights')
    parser.add_argument('--test_image', type=str, required=True,
                        help='Path to the test image')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the model weights')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if not os.path.exists(args.weights):
        print(f"ERROR: Model weights not found at '{args.weights}'")
        print("Please verify the path. If on Kaggle, it's likely under '/kaggle/working/'.")
        return

    if not os.path.exists(args.test_image):
        print(f"ERROR: Test image not found at '{args.test_image}'")
        print("Please provide a valid path to an image file.")
        return

    try:
        print(f"Loading model from: {args.weights}")
        model = YOLO(args.weights)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        print(f"Predicting on image: {args.test_image}")
        results = model.predict(
            source=args.test_image,
            imgsz=416,       # Image size used during training
            conf=0.25,       # Confidence threshold for detections
            save=True,       # Save the image with detected bounding boxes
            project='personinwater_predictions', # Directory to save results
            name='simple_test_run'             # Sub-directory for this specific run
        )
        print(f"Prediction complete. Results saved in 'personinwater_predictions/simple_test_run'")

    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()