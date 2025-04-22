import os
import json

IMG_WIDTH = 3840
IMG_HEIGHT = 2160

def convert_bbox(x, y, w, h):
    x_center = (x + w / 2) / IMG_WIDTH
    y_center = (y + h / 2) / IMG_HEIGHT
    w_norm = w / IMG_WIDTH
    h_norm = h / IMG_HEIGHT
    return x_center, y_center, w_norm, h_norm

def convert_set(json_path, annotations_dir, label_output_dir, image_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    os.makedirs(label_output_dir, exist_ok=True)

    for clip_id, frames in data.items():
        annotation_file = os.path.join(annotations_dir, f"{clip_id}.txt")
        if not os.path.exists(annotation_file):
            print(f"[!] No file: {annotation_file}")
            continue

        with open(annotation_file, 'r') as f:
            bboxes = [line.strip() for line in f.readlines() if line.strip()]

        if len(bboxes) != len(frames):
            print(f"[!] Mismatched number of frames and bboxes in clip {clip_id} ({len(frames)} frames, {len(bboxes)} bboxes)")
            continue

        for frame_name, bbox_line in zip(frames.keys(), bboxes):
            try:
                x, y, w, h = map(int, bbox_line.split(','))
            except ValueError:
                print(f"[!] Invalid bbox in clip {clip_id}: {bbox_line}")
                continue

            frame_num = int(frame_name.replace(".png", ""))
            img_path = os.path.join(image_dir, f"{frame_num}.jpg")
            if not os.path.exists(img_path):
                print(f"[!] Image not found: {img_path}")
                continue

            x_c, y_c, w_n, h_n = convert_bbox(x, y, w, h)
            label_file = os.path.join(label_output_dir, f"{frame_num}.txt")
            with open(label_file, 'w') as out:
                out.write(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")

convert_set("train.json", "labels/train", "labels/train_2", "images/train")
convert_set("val.json", "labels/val", "labels/val_2", "images/val")