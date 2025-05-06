import cv2
import json
import os
import glob

def draw_predictions(image_path, json_path, output_image_path):
    # Read the JSON file
    with open(json_path, 'r') as f:
        try:
            # Attempt to load the JSON. It might be an empty file or malformed.
            loaded_json = json.load(f)
            if not loaded_json: # Check if the loaded JSON is empty (e.g., an empty list or dict)
                print(f"Warning: JSON file {json_path} is empty or contains no data. Skipping.")
                return False # Indicate failure or skip
            predictions_data = loaded_json[0]  # Get first prediction set
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {json_path}. Skipping.")
            return False # Indicate failure or skip

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        # ValueError is fine, but let's print a more user-friendly message here too
        print(f"Error: Could not read image at {image_path}. Skipping.")
        return False # Indicate failure or skip

    # Check if predictions key exists and is not empty
    if 'predictions' not in predictions_data or \
       'predictions' not in predictions_data['predictions'] or \
       not predictions_data['predictions']['predictions']:
        print(f"Warning: No 'predictions' found or 'predictions' list is empty in {json_path} for image {image_path}. Saving original image.")
        # Optionally, save the original image without annotations or skip saving
        # cv2.imwrite(output_image_path, img)
        # print(f"Saved original image (no predictions) to {output_image_path}")
        return True # Or False if you want to signify no predictions drawn


    # Draw each prediction
    for pred in predictions_data['predictions']['predictions']:
        # Extract coordinates
        x = int(pred['x'])
        y = int(pred['y'])
        w = int(pred['width'])
        h = int(pred['height'])
        conf = pred['confidence']
        
        # Calculate box coordinates
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add confidence score text
        label = f"{pred['class']}: {conf:.2f}"
        cv2.putText(img, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image
    try:
        cv2.imwrite(output_image_path, img)
        print(f"Saved predictions for {os.path.basename(image_path)} to {output_image_path}")
        return True # Indicate success
    except Exception as e:
        print(f"Error saving image {output_image_path}: {str(e)}")
        return False # Indicate failure

def process_all_images():
    base_dir = os.path.dirname(os.path.abspath(__file__)) # Use abspath for robustness
    output_dir = os.path.join(base_dir, "predictions_output")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files
    json_files = glob.glob(os.path.join(base_dir, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in {base_dir}")
        return

    processed_count = 0
    skipped_count = 0

    for json_path in json_files:
        # Skip non-prediction JSON files if any (e.g., empty files)
        if os.path.getsize(json_path) == 0:
            print(f"Skipping empty JSON file: {json_path}")
            skipped_count += 1
            continue
            
        # Construct corresponding image path
        base_name_json = os.path.splitext(os.path.basename(json_path))[0]
        
        # Try common image extensions if not just .jpg
        # This makes it more flexible
        image_found = False
        potential_image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_path = None
        original_image_filename = None

        for ext in potential_image_extensions:
            potential_image_path = os.path.join(base_dir, f"{base_name_json}{ext}")
            if os.path.exists(potential_image_path):
                image_path = potential_image_path
                original_image_filename = os.path.basename(image_path)
                image_found = True
                break
        
        if not image_found:
            print(f"Warning: No matching image (e.g., {base_name_json}.jpg/.png) for {json_path}")
            skipped_count += 1
            continue

        # Construct output image path
        output_image_path = os.path.join(output_dir, original_image_filename) # Save with original image name
        
        try:
            print(f"Processing {original_image_filename} with {os.path.basename(json_path)}...")
            success = draw_predictions(image_path, json_path, output_image_path)
            if success:
                processed_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"Error processing {original_image_filename}: {str(e)}")
            skipped_count += 1
            
    print(f"\nProcessing complete. {processed_count} images saved with predictions. {skipped_count} files skipped or had issues.")

if __name__ == "__main__":
    process_all_images()