import cv2 
import numpy as np
import mtcnn
from architecture import *
from train_v2_1 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle
import time


confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)

#'get_face: 얼굴 이미지 및 좌표를 반환
def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

# get_encode: 얼굴을 인코딩하여 특징 벡터를 반환합니다.
def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

#load_pickle: 저장된 얼굴 인코딩 데이터를 불러옴
def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

# 'apply_mosaic: 모자이크 처리를 수행
def apply_mosaic(image, pt_1, pt_2, kernel_size=15):
    x1, y1 = pt_1
    x2, y2 = pt_2
    face_height, face_width, _ = image[y1:y2, x1:x2].shape
    
    # 모자이크 처리할 영역의 크기를 기준으로 잘라내기
    face = image[y1:y1+face_height, x1:x1+face_width]
    
    # 모자이크 처리
    face = cv2.resize(face, (kernel_size, kernel_size), interpolation=cv2.INTER_LINEAR)
    face = cv2.resize(face, (face_width, face_height), interpolation=cv2.INTER_NEAREST)
    
    # 모자이크 처리된 영역을 이미지에 적용
    image[y1:y1+face_height, x1:x1+face_width] = face
    
    return image

# 'detect: 얼굴을 감지하고 인식하며, 모자이크 처리를 수행
def detect(img, detector, encoder, encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist

        if name == 'unknown':
            img = apply_mosaic(img, pt_1, pt_2) #모자이크
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
             # 이름만 표시하도록 확률 값 제거
            cv2.putText(img, name, (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 200, 200), 2)
    return img 

if __name__ == "__main__":
    required_shape = (160,160) # 얼굴 인코딩을 위한 이미지 크기 설정
    face_encoder = InceptionResNetV2() # 모델 생성
    path_m = "facenet_keras_weights.h5" # 모델 가중치를 로드
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings/encodings.pkl' #얼굴 인코딩 데이터 로드
    face_detector = mtcnn.MTCNN() #MTCNN 얼굴 검출기 생성
    encoding_dict = load_pickle(encodings_path)
    
    cap = cv2.VideoCapture(0) #비디오 캡처 객체 생성

    # 비디오 녹화 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')#비디오 코덱 설정
    out = cv2.VideoWriter('output4.avi', fourcc, 20.0, (640, 480))#비디오 녹화 객체 생성

    #비디오 스트림 처리
    while cap.isOpened(): # 카메라가 열려 있는 동안 반복
        start_time = time.time()  # 현재 시간 기록
        ret, frame = cap.read() #프레임을 캡처

        if not ret:
            print("CAM NOT OPENED") 
            break
        
        frame = detect(frame, face_detector, face_encoder, encoding_dict)
        
        #처리된 프레임을 녹화
        out.write(frame)

        cv2.imshow('camera', frame) #처리된 프레임을 화면에 표시

         # FPS 계산 및 표시
        end_time = time.time()  # 현재 시간 기록
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}")

        # 'q' 키를 누르면 루프를 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished 리소스 해제
    cap.release() #비디오 캡처 객체 해제
    out.release() #비디오 녹화 객체 해제
    cv2.destroyAllWindows() #모든 OpenCV 창 닫기
