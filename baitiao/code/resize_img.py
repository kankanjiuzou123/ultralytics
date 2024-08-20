from PIL import Image
import os

# 设置源文件夹路径和目标文件夹路径
source_folder = '/home/wkc/project/baitiao/images/'
destination_folder = '/home/wkc/project/baitiao/new_images/'

# 确保目标文件夹存在，不存在则创建
os.makedirs(destination_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        # 打开图片
        img_path = os.path.join(source_folder, filename)
        with Image.open(img_path) as img:
            # 顺时针旋转90度
            rotated_img = img.rotate(-90, expand=True)
            # 调整图片大小
            resized_img = rotated_img.resize((640, 640))
            # 构建目标文件路径
            save_path = os.path.join(destination_folder, filename)
            # 保存修改后的图片到目标路径
            resized_img.save(save_path)
        print(f"Processed and saved: {filename}")

print("All images have been processed and saved to the destination folder.")
