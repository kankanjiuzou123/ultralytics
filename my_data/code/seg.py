import os
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('../weights/yolov8s-seg.pt')

# Specify the input and output directories
input_dir = '../coco128-seg/images/train2017'

# Process each image in the input directory
for img_file in os.listdir(input_dir):
    if img_file.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        results = model(img,save=True)
        
        # output_path = os.path.join(output_dir, img_file)
        # print(results)
        # results.save(filename=output_path)
        # print(f"Processed and saved: {img_file}")

print("Batch processing completed.")

