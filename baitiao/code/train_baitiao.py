from ultralytics import YOLO

# Load a model
model = YOLO("../../my_data/weights/yolov8s-seg.pt")
results = model.train(data="../cfg/baitiao.yaml", epochs=300, imgsz=640,batch=8)