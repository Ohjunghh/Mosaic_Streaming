import os 
import cv2
from .architecture import * 
import os 
import cv2
import mtcnn
import numpy as np 
from sklearn.preprocessing import Normalizer
from django.conf import settings

###### Detect Environment and Set Paths #########
if os.path.exists('/workspace'):
    # Docker 환경
    base_path = '/workspace/face_recognition_2/'
else:
    # 로컬 환경
    base_path = './face_recognition_2/'

path = os.path.join(base_path, 'facenet_keras_weights.h5')

###### paths and variables #########
required_shape = (160, 160)
face_encoder = InceptionResNetV2()
face_encoder.load_weights(path)
face_detector = mtcnn.MTCNN()
l2_normalizer = Normalizer('l2')
###############################


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def encode_faces(user_id):
    # 사용자별 폴더 경로 설정
    person_dir = os.path.join(settings.MEDIA_ROOT, 'faces', str(user_id))
    
    # 인코딩 딕셔너리 초기화
    encoding_dict = dict()

    # 사용자 폴더에 이미지가 없을 경우
    if not os.path.exists(person_dir):
        return encoding_dict  # 빈 딕셔너리 반환
    
    encodes = []  # 각 얼굴에 대한 인코딩을 새로 계산하기 전에 초기화
    
    # 폴더 내의 모든 이미지 파일을 가져옴
    image_files = os.listdir(person_dir)
       
    for image_name in image_files:
        image_path = os.path.join(person_dir, image_name)

        img_BGR = cv2.imread(image_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

        # 얼굴 탐지
        faces = face_detector.detect_faces(img_RGB) 
        if faces:  # 얼굴이 감지되었을 때만 진행
            x1, y1, width, height = faces[0]['box'] 
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = img_RGB[y1:y2 , x1:x2] 

            # 얼굴 인코딩 처리
            face = normalize(face)
            face = cv2.resize(face, required_shape)
            face_d = np.expand_dims(face, axis=0)
            encode = face_encoder.predict(face_d)[0]
            encodes.append(encode)

    if encodes:
        # 얼굴 인코딩 벡터 평균화 및 정규화
        encode = np.sum(encodes, axis=0)
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[str(user_id)] = encode

    return encoding_dict  # 인코딩 딕셔너리 반환
