import cv2
import torch
import pickle
import numpy as np
from face_recognition_2.architecture import InceptionResNetV2
from face_recognition_2.train_v2_1 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
import os
from users.models import CustomUser  # CustomUser 모델 가져오기
from webcam.sort import Sort 
import logging

# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class VideoProcessor:
    def __init__(self, user_id):
        self.face_encoder = self.load_face_encoder()
        self.encoding_dict = self.load_encoding_dict(user_id)
        self.tracker = Sort()  # SORT 트래커 초기화
        self.frame_count = 0
        
        self.broadcaster_vote_count = {}
        self.broadcaster_id = None
        self.broadcaster_box = None
        self.vote_threshold = 5
        self.max_frames = 30
        self.confidence_t = 0.5  # YOLO 탐지 신뢰도 임계값
        self.recognition_t = 0.26  # 얼굴 인식 거리 임계값
        self.required_size = (160, 160)
        self.face_frame_count=0
        self.broadcaster_frame_count = 0  # 방송인 유지 프레임 카운트 (추가)
        self.recognized_faces = {}  # 인식된 얼굴들 저장 (추가)

    def load_face_encoder(self):
        face_encoder = InceptionResNetV2()
        face_encoder.load_weights('./face_recognition_2/facenet_keras_weights.h5')
        return face_encoder

    def load_encoding_dict(self,user_id):
        # 사용자 ID를 기반으로 인코딩 파일 경로 설정
        encodings_path = os.path.join(f'C:/GRADU/back/media/encodings/{user_id}/encoding_vector.pkl')
                
        if os.path.exists(encodings_path):
            with open(encodings_path, 'rb') as f:
                encoding_dict = pickle.load(f)
        else:
            encoding_dict = {}  # 파일이 없을 경우 빈 딕셔너리로 초기화

        return encoding_dict


    # 얼굴 인코딩 계산
    def get_encode(self, face, size ):
        if face is None or face.size == 0:
            logging.warning("Face image is None or empty")
            return None
    
        logging.debug(f"Processing face image with shape: {face.shape}")
        face = normalize(face)
        face = cv2.resize(face, size)
        encode = self.face_encoder.predict(np.expand_dims(face, axis=0))[0]
        return encode


    # 모자이크 처리 함수
    def apply_mosaic(self, image, pt_1, pt_2, kernel_size=15):
        x1, y1 = pt_1
        x2, y2 = pt_2

        # 좌표 유효성 검사 (이미지 경계를 넘는지 확인)
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
            logging.warning(f"Skipping invalid region: {(x1, y1, x2, y2)}")
            return image

        # 얼굴 영역 추출
        face = image[y1:y2, x1:x2]
        
        # 빈 이미지인지 확인
        if face.size == 0:
            logging.warning(f"Empty region at: {(x1, y1, x2, y2)}")
            return image
        
        face_height, face_width = face.shape[:2]
        logging.debug(f"Applying mosaic to region: {(x1, y1, x2, y2)} with size: {(face_width, face_height)}")
        
        # 모자이크 처리
        face = cv2.resize(face, (kernel_size, kernel_size), interpolation=cv2.INTER_LINEAR)
        face = cv2.resize(face, (face_width, face_height), interpolation=cv2.INTER_NEAREST)
        
        # 원본 이미지에 모자이크된 영역 복사
        image[y1:y2, x1:x2] = face
        return image


    def detect_and_mosaic(self, img, model, license_plate, invoice, id_card, license_card, knife, face):
        global face_frame_count, broadcaster_id, broadcaster_frame_count, recognized_faces

        # 유효하지 않은 이미지일 경우 None 반환
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        detect_face = False

        img_height, img_width = img.shape[:2]
        detections = []

            # 탐지된 객체를 순회하며 각 클래스에 따라 처리
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            if conf < self.confidence_t:
                continue

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 이미지 크기 범위 내에 있는지 확인
            if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height or x1 >= x2 or y1 >= y2:
                continue

            # 번호판, 송장, ID 카드 등 모자이크 처리
            if cls == 0 and license_plate:
                img = self.apply_mosaic(img, (x1, y1), (x2, y2))
            if cls == 1 and invoice:
                img = self.apply_mosaic(img, (x1, y1), (x2, y2))
            if cls == 2 and id_card:
                img = self.apply_mosaic(img, (x1, y1), (x2, y2))
            if cls == 3 and license_card:
                img = self.apply_mosaic(img, (x1, y1), (x2, y2))
            if cls == 4 and knife:
                img = self.apply_mosaic(img, (x1, y1), (x2, y2))

                # 얼굴 클래스일 경우
            if cls == 5 and face:
                detections.append([x1, y1, x2, y2])
                detect_face = True

            # 얼굴 추적
        if detections:
            tracked_objects = self.tracker.update(np.array(detections))
            logging.debug(f"Tracked objects: {tracked_objects}")
        else:
            tracked_objects = self.tracker.update(np.empty((0, 4)))
            logging.debug(f"No faces detected")

        # 방송인 얼굴 추적 및 선택 (투표 로직 추가)
        if self.face_frame_count % 30 < 10:
            min_distance = float('inf')
            selected_track_id = None

            # 트래커로 추적된 얼굴 중에서 방송인 얼굴 선택
            for track in tracked_objects:
                if len(track) >= 5:
                    x1, y1, x2, y2, track_id = map(int, track[:5])

                    face_img = img[y1:y2, x1:x2]

                    # 얼굴 이미지가 유효한지 확인
                    if face_img.size == 0:
                        logging.warning(f"Invalid face image size at: {(x1, y1, x2, y2)}")
                        continue

                        # 얼굴 인코딩
                    encode_face = self.get_encode(face_img, self.required_size)
                    if encode_face is None:
                        logging.warning(f"Failed to encode face at: {(x1, y1, x2, y2)}")
                        continue

                    encode_face = l2_normalizer.transform(encode_face.reshape(1, -1))[0]

                        # 저장된 얼굴 인코딩과 비교하여 가장 가까운 얼굴 선택
                    for db_name, db_encode in self.encoding_dict.items():
                        dist = cosine(db_encode, encode_face)
                        logging.debug(f"Distance between db_name {db_name} and current face: {dist}")
                        if dist < self.recognition_t and dist < min_distance:
                            min_distance = dist
                            selected_track_id = track_id

                # 투표 기반으로 방송인 얼굴 선택
            if selected_track_id is not None:
                logging.debug(f"Selected track ID for broadcaster: {selected_track_id}")
                if selected_track_id not in self.broadcaster_vote_count:
                    self.broadcaster_vote_count[selected_track_id] = 0
                self.broadcaster_vote_count[selected_track_id] += 1

            # 투표 결과를 통해 방송인 얼굴 확정
        if self.face_frame_count % 30 == 0:
            max_votes = 0
            for track_id, votes in self.broadcaster_vote_count.items():
                if votes > max_votes:
                    max_votes = votes
                    self.broadcaster_id = track_id

            if max_votes >= self.vote_threshold:
                # 방송인 얼굴의 위치를 업데이트
                self.broadcaster_box = next((track[:4] for track in tracked_objects if len(track) >= 5 and int(track[-1]) == self.broadcaster_id), None)
                logging.debug(f"인식된 방송자: {self.broadcaster_box}, 투표수: {max_votes}")
            else:
                self.broadcaster_box = None

            # 투표 기록 초기화
            self.broadcaster_vote_count.clear()

            # 방송인 얼굴 외 나머지 얼굴에 모자이크 적용
        for track in tracked_objects:
            if len(track) >= 5:
                x1, y1, x2, y2, track_id = map(int, track[:5])

                    # 방송인 얼굴 외에 모자이크 적용
                if track_id != self.broadcaster_id:
                    img = self.apply_mosaic(img, (x1, y1), (x2, y2))
                else:
                    logging.debug(f"Broadcaster ID {track_id} is not blurred. Coordinates: {(x1, y1, x2, y2)}")
        if detect_face:
            self.face_frame_count += 1

        return img