import os
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('../weights/yolov8s.pt')

# Specify the input and output directories
input_dir = '../coco128/images/train2017'

results = model(input_dir,save=True)

print("Batch processing completed.")