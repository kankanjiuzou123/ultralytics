from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="../cfg/coco128-seg.yaml", epochs=100, imgsz=640)