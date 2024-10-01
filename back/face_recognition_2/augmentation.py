import os
from PIL import Image, ImageEnhance
import cv2
import numpy as np

def adjust_brightness(image, factor):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def adjust_saturation(image, factor):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2HSV)
    image[...,1] = image[...,1]*factor
    image = np.clip(image, 0, 255)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return Image.fromarray(image)

def process_images(directory):
    for filename in os.listdir(directory):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            original_path = os.path.join(directory, filename)
            image = Image.open(original_path)
            
            # 밝기 및 채도 조절 값
            adjustments = [
                (0.6, 0.8), # 밝기 낮추고, 채도 낮추기
                (1.2, 0.8), # 밝기 높이고, 채도 낮추기
                (0.8, 1.2), # 밝기 낮추고, 채도 높이기
                (1.2, 1.2), # 밝기 높이고, 채도 높이기
                (0.7, 1.0)  # 밝기만 낮추기
            ]
            
            for i, (brightness, saturation) in enumerate(adjustments, start=1):
                modified_image = adjust_brightness(image, brightness)
                modified_image = adjust_saturation(modified_image, saturation)
                
                # 변경된 이미지 저장
                new_filename = f"{filename.split('.')[0]}_modified_{i}.{filename.split('.')[-1]}"
                new_path = os.path.join(directory, new_filename)
                modified_image.save(new_path)

# 이미지가 있는 경로 지정
directory = 'C:/GRADU/Yolo_Project/face/Face-recognition-Using-Facenet-On-Tensorflow-2.X/Faces/Junghyun'
process_images(directory)