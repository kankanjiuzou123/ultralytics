from ultralytics import YOLO

# Load a model
model = YOLO("../weights/yolov8s.pt")
results = model.train(data="../cfg/coco128.yaml", epochs=30, imgsz=640,batch=2)