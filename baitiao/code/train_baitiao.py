from ultralytics import YOLO

# Load a model
model = YOLO("../../my_data/weights/yolov8s-seg.pt")
results = model.train(data="../cfg/baitiao.yaml", epochs=100, imgsz=640,batch=8,lr0=0.001,optimizer='Adam',augment=False,auto_augment='',
                      hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, 
                      perspective=0.0, flipud=0.0,fliplr=0.0, mosaic=0.0, mixup=0.0)