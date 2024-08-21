import os
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('../weights/yolov8s.pt')

# Specify the input and output directories
input_dir = '../coco128/images/train2017'

# Process each image in the input directory
for img_file in os.listdir(input_dir):
    if img_file.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        results = model(img,save=True,save_frames=True)

print("Batch processing completed.")