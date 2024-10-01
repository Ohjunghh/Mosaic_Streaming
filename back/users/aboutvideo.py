import cv2
import torch
import pickle
import numpy as np
from face_recognition_2.architecture import InceptionResNetV2
from face_recognition_2.train_v2_1 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
import os
from users.models import CustomUser  # CustomUser 모델 가져오기

# Constants
confidence_t = 0.5  # YOLO 탐지 신뢰도 임계값
recognition_t = 0.3  # 얼굴 인식 거리 임계값
required_size = (160, 160)

def load_yolo_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./face_recognition_2/926.pt', force_reload=True)
    model.conf = confidence_t  # 탐지 신뢰도 임계값
    model.classes = None
    model.agnostic_nms = False
    return model

def load_face_encoder():
    face_encoder = InceptionResNetV2()
    face_encoder.load_weights('./face_recognition_2/facenet_keras_weights.h5')
    return face_encoder

def load_encoding_dict(user_id):
    # 사용자 ID를 기반으로 인코딩 파일 경로 설정
    encodings_path = os.path.join(f'C:/GRADU/back/media/encodings/{user_id}/encoding_vector.pkl')
               
    if os.path.exists(encodings_path):
        with open(encodings_path, 'rb') as f:
            encoding_dict = pickle.load(f)
    else:
        encoding_dict = {}  # 파일이 없을 경우 빈 딕셔너리로 초기화

    return encoding_dict


# 얼굴 인코딩 계산
def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

# 모자이크 처리 함수
def apply_mosaic(image, pt_1, pt_2, kernel_size=15):
    x1, y1 = pt_1
    x2, y2 = pt_2
    face_height, face_width, _ = image[y1:y2, x1:x2].shape
    face = image[y1:y1 + face_height, x1:x1 + face_width]
    face = cv2.resize(face, (kernel_size, kernel_size), interpolation=cv2.INTER_LINEAR)
    face = cv2.resize(face, (face_width, face_height), interpolation=cv2.INTER_NEAREST)
    image[y1:y1 + face_height, x1:x1 + face_width] = face
    return image

# 얼굴 탐지 및 모자이크 적용
def detect_and_mosaic(img, model, face_encoder, encoding_dict,license_plate, invoice, id_card, license_card, knife,face):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)  # YOLO 탐지 수행
    
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        if conf < confidence_t:
            continue

        # 객체가 얼굴일 때
        object_img = img[int(y1):int(y2), int(x1):int(x2)]
        if cls == 5 and face:  # face 클래스가 선택된 경우 # 나중에 5로
            encode_face = get_encode(face_encoder, object_img, required_size)
            encode_face = l2_normalizer.transform(encode_face.reshape(1, -1))[0]

            name = 'unknown'
            distance = float("inf")
            for db_name, db_encode in encoding_dict.items():
                dist = cosine(db_encode, encode_face)
                if dist < recognition_t and dist < distance:
                    name = db_name
                    distance = dist

            # 모자이크 적용
            if name == 'unknown':
                img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            else:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f'{name} {distance:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
       
      
        if cls == 0 and license_plate:  # 차량 번호판 클래스가 선택된 경우
            img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            
        if cls == 1 and invoice:  # invoice(송장)클래스가 선택된 경우
            img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            
        if cls == 2 and id_card:  # id_card 클래스가 선택된 경우
            img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
        
        if cls == 3 and license_card:
            img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))

        if cls == 4 and knife:  # knife 클래스가 선택된 경우
            img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))

    return img


# # 얼굴 탐지 및 모자이크 적용 627.pt
# def detect_and_mosaic(img, model, face_encoder, encoding_dict,invoice,id_card,knife,face):
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = model(img_rgb)  # YOLO 탐지 수행
    
#     for det in results.xyxy[0]:
#         x1, y1, x2, y2, conf, cls = det
#         cls = int(cls)
#         if conf < confidence_t:
#             continue

#         # 객체가 얼굴일 때
#         object_img = img[int(y1):int(y2), int(x1):int(x2)]
#         if cls == 4 and face:  # face 클래스가 선택된 경우
#             encode_face = get_encode(face_encoder, object_img, required_size)
#             encode_face = l2_normalizer.transform(encode_face.reshape(1, -1))[0]

#             name = 'unknown'
#             distance = float("inf")
#             for db_name, db_encode in encoding_dict.items():
#                 dist = cosine(db_encode, encode_face)
#                 if dist < recognition_t and dist < distance:
#                     name = db_name
#                     distance = dist

#             # 모자이크 적용
#             if name == 'unknown':
#                 img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
#             else:
#                 cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#                 cv2.putText(img, f'{name} {distance:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         # else:
#         #     img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
#         #     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#         #     cv2.putText(img, str(cls), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         # 일단 박아둠
#         license_plate = True
#         if cls == 0 and license_plate:  # 차량 번호판 클래스가 선택된 경우
#             img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            
#         if cls == 1 and invoice:  # invoice(송장)클래스가 선택된 경우
#             img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            
#         if cls == 2 and id_card:  # id_card 클래스가 선택된 경우
#             img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
  
#         if cls == 3 and knife:  # knife 클래스가 선택된 경우
#             img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))

#     return img
