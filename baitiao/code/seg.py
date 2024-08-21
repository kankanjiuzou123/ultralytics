import os
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('runs/segment/train2/weights/best.pt')

# Specify the input and output directories
input_dir = '/home/wkc/project/backup/baitiao/no_label_img/'

results = model(input_dir,save=True)
        
        # output_path = os.path.join(output_dir, img_file)
        # print(results)
        # results.save(filename=output_path)
        # print(f"Processed and saved: {img_file}")

print("Batch processing completed.")

