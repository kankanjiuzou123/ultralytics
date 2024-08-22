from ultralytics import YOLO

# Load a model
model = YOLO("../../my_data/weights/yolov8s.pt")
results = model.train(data="../cfg/baitiao_detect.yaml", epochs=100, imgsz=640,batch=8)