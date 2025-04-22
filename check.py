import cv2
import os

img_path = 'datasets/images/train/123.jpg' 
label_path = 'datasets/labels/train/123.txt'
IMG_WIDTH = 3840
IMG_HEIGHT = 2160

img = cv2.imread(img_path)
if img is None:
    raise Exception(f"Picture not found: {img_path}")

with open(label_path, 'r') as f:
    for line in f.readlines():
        cls, x_c, y_c, w, h = map(float, line.strip().split())

        x = int((x_c - w / 2) * IMG_WIDTH)
        y = int((y_c - h / 2) * IMG_HEIGHT)
        w = int(w * IMG_WIDTH)
        h = int(h * IMG_HEIGHT)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

scale_percent = 30
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
resized = cv2.resize(img, (width, height))

cv2.imshow("BBox check", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()