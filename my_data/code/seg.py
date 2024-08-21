import os
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('../weights/yolov8s-seg.pt')

# Specify the input and output directories
input_dir = '../coco128-seg/images/train2017'

results = model(input_dir,save=True)
        
        # output_path = os.path.join(output_dir, img_file)
        # print(results)
        # results.save(filename=output_path)
        # print(f"Processed and saved: {img_file}")

print("Batch processing completed.")

