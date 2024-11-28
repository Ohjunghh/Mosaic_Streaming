import os 
import cv2
from .architecture import * 
import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model

###### Detect Environment and Set Paths #########

if os.path.exists('/workspace'):
    # Docker 환경
    base_path = '/workspace/face_recognition_2/'
else:
    # 로컬 환경
    base_path = './face_recognition_2/'

face_data = os.path.join(base_path, 'Faces/')
required_shape = (160, 160)
face_encoder = InceptionResNetV2()
path = os.path.join(base_path, 'facenet_keras_weights.h5')
face_encoder.load_weights(path)
face_detector = mtcnn.MTCNN()
encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')
encodings_path = os.path.join(base_path, 'encodings/encodings.pkl')

###############################


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def main():
    # 기존 인코딩 딕셔너리 로드
    if os.path.exists(encodings_path):
        with open(encodings_path, 'rb') as file:
            encoding_dict.update(pickle.load(file))

    for face_names in os.listdir(face_data):
        if face_names in encoding_dict:
            print(f"{face_names} already exists. Skipping...")
            continue

        person_dir = os.path.join(face_data, face_names)
        encodes = []  # 각 얼굴에 대한 인코딩을 새로 계산하기 전에 초기화

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)

            img_BGR = cv2.imread(image_path)
            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

            x = face_detector.detect_faces(img_RGB) 
            if x:  # 얼굴이 감지되었을 때만 진행
                x1, y1, width, height = x[0]['box'] 
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face = img_RGB[y1:y2 , x1:x2] 

                face = normalize(face)
                face = cv2.resize(face, required_shape)
                face_d = np.expand_dims(face, axis=0)
                encode = face_encoder.predict(face_d)[0]
                encodes.append(encode)

        if encodes:
            encode = np.sum(encodes, axis=0)
            encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
            encoding_dict[face_names] = encode

    # 인코딩 딕셔너리 저장
    with open(encodings_path, 'wb') as file:
        pickle.dump(encoding_dict, file)

if __name__ == '__main__':
    main()
