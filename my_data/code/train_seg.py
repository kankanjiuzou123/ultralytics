from ultralytics import YOLO
import yaml

# Load a model
model = YOLO("../weights/yolov8s-seg.pt")
# print(model)# load a pretrained model (recommended for training)

# Train the model
yaml_path = '../cfg/coco128.yaml'  # 请将此路径替换为你实际的 YAML 文件路径

# 打开并读取 YAML 文件
# try:
#     with open(yaml_path, 'r', encoding='utf-8') as file:
#         config = yaml.safe_load(file)  # 使用 safe_load 安全地加载 YAML 文件内容

#     # 打印读取到的内容
#     print("YAML 文件内容:")
#     print(config)
# except FileNotFoundError:
#     print(f"错误: 无法找到文件 {yaml_path}")
# except yaml.YAMLError as exc:
#     print(f"错误: 读取 YAML 文件时出现错误 - {exc}")
results = model.train(data="../cfg/coco128-seg.yaml", epochs=100, imgsz=640,batch=2)