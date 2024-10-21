import cv2
import torch
import pickle
import numpy as np
import subprocess
import threading
import os
import asyncio 
import traceback

from multiprocessing import Process
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators import gzip
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from face_recognition_2.architecture import InceptionResNetV2
from face_recognition_2.train_v2_1 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from users.models import CustomUser
from yolo_loader import yolo_model
import logging
from webcam.sort import Sort 
import torch.utils.cpp_extension
torch.utils.cpp_extension.CppExtension.load = lambda *args, **kwargs: None
# Constants and configurations
confidence_t = 0.5
recognition_t = 0.35
required_size = (160, 160)
CONN_LIMIT = 10
possible_list = [True] * CONN_LIMIT
stream_instances = {}
logger = logging.getLogger(__name__)


def setup_logger():
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler('streaming.log')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

setup_logger()

class WebcamStream:
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.cap = None
        self.process = None
        self.model = yolo_model  # 글로벌 모델 사용
        self.tracker = Sort()

        # 얼굴 인식 관련 변수
        self.face_frame_count = 0  # 각 스트림별로 독립적인 프레임 카운트
        self.recognized_faces = {}  # 각 스트림별로 인식된 얼굴 상태를 저장하는 딕셔너리


    def load_face_encoder(self):
        self.face_encoder = InceptionResNetV2()
        self.face_encoder.load_weights('./face_recognition_2/facenet_keras_weights.h5')


    def load_encoding_dict(self, email):
        user = CustomUser.objects.get(email=email)
        user_id = user.id  # 사용자 id 가져오기

        # 사용자별로 경로를 동적으로 생성
        pkl_path = os.path.join('C:/GRADU/back/media/encodings', str(user_id), 'encoding_vector.pkl')

        # 해당 경로의 pkl 파일을 불러오기
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                self.encoding_dict = pickle.load(f)
            logger.info(f"Loaded encoding dictionary for user {email} from {pkl_path}")
        else:
            logger.error(f"Encoding file not found for user {email} at {pkl_path}")
            self.encoding_dict = {}  # 파일이 없을 경우 빈 딕셔너리로 초기화

    def start(self, email, stream_key, license_plate, invoice, id_card, license_card, knife, face):
        stream_opened = False  # 스트림이 정상적으로 열렸는지 여부를 추적하는 플래그

        try:
            #self.load_model()
            self.load_face_encoder()
            self.load_encoding_dict(email)
            #self.cap = cv2.VideoCapture(f"rtmp://192.168.58.194/live/{stream_key}")
            self.cap = cv2.VideoCapture(f"rtmp://172.20.79.30/live/{stream_key}")
            
            if not self.cap.isOpened():
                logger.error("Unable to open RTMP stream.")
                self.stop()  # 스트림을 종료
                return
            
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.command = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f"{width}x{height}",  # 해상도 추가 (예: 640x480)
                '-pix_fmt', 'bgr24',
                '-r', str(fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast',
                '-f', 'flv',
                f'rtmp://172.20.79.30/live-out/{self.stream_id}'  # 스트리밍 대상 RTMP URL
            ]


            stream_opened = True  # 스트림이 정상적으로 열렸음을 기록
            self.start_ffmpeg()

           

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Unable to read frame.")
                    break
                
                #모자이크 멈춰
                frame = self.detect(frame,license_plate, invoice, id_card, license_card, knife, face)
                #frame = cv2.resize(frame, (1280, 720))

                #  더 이상 화면에 표시하지 않기 때문에 cv2.imshow()와 cv2.waitKey() 제거
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # FFmpeg로 프레임 전송
                if self.process:
                    try:
                        self.process.stdin.write(frame.tobytes())  # 프레임 전송
                    except Exception as e:
                        logger.error(f"Error writing frame to FFmpeg process: {e}")
                        logger.error(traceback.format_exc())  # 전체 스택 추적을 출력
                        break  # 에러 발생 시 루프 종료

            
        except Exception as e:
            logger.error(f"Error during streaming process: {e}")
        finally:
            if stream_opened:
              logger.info("Stopping the stream and releasing resources.")
            # 리소스 해제와 관련된 로그 추가
            if self.cap:
                self.cap.release()
                logger.info("Video capture released.")

            if self.process:
                self.stop()
                logger.info("FFmpeg process terminated.")

            logger.info("Stream stopped and resources cleaned up.")
                

    def start_ffmpeg(self):
        try:
            # FFmpeg 프로세스 실행
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stderr=None)#stderr=subprocess.PIPE)
            if self.process.poll() is None:
                logger.debug("FFmpeg process is running.")
            else:
                logger.error("FFmpeg process did not start correctly.")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg process: {e}")
            self.stop()  # FFmpeg 시작 실패 시 스트림을 종료

    #직전꺼
    # def detect(self, img, license_plate, invoice, id_card, license_card, knife, face):
    #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     results = self.model(img_rgb)

    #     track_results = []
    #     for det in results.xyxy[0]:
    #         x1, y1, x2, y2, conf, cls = det
    #         cls = int(cls)
    #         if conf < confidence_t:
    #             continue

    #         if cls == 5 and face:  # 얼굴 클래스일 경우
    #             track_results.append([x1.item(), y1.item(), x2.item(), y2.item()])  # 좌표만 추가

    #     # SORT 알고리즘 적용 (얼굴 추적)
    #     tracked_objects = self.tracker.update(np.array(track_results)) if len(track_results) > 0 else self.tracker.update(np.empty((0, 4)))

    #     closest_broadcaster = {'name': 'unknown', 'distance': float("inf"), 'box': None}

    #     # 얼굴 인식 플래그 초기화
    #     face_recognized = False

    #     for obj in tracked_objects:
    #         x1, y1, x2, y2, track_id = obj[:5]
    #         object_img = img[int(y1):int(y2), int(x1):int(x2)]

    #         # 얼굴 인식을 30프레임마다 수행
    #         if self.face_frame_count % 30 == 0:
    #             encode_face = self.get_encode(object_img, required_size)
    #             if encode_face is None:
    #                 continue  # 얼굴 인식 실패 시 건너뜀

    #             encode_face = l2_normalizer.transform(encode_face.reshape(1, -1))[0]
    #             name = 'unknown'
    #             distance = float("inf")

    #             for db_name, db_encode in self.encoding_dict.items():
    #                 dist = cosine(db_encode, encode_face)
    #                 if dist < recognition_t and dist < distance:
    #                     name = db_name
    #                     distance = dist

    #             # 인식된 얼굴 정보 저장
    #             self.recognized_faces[track_id] = {'name': name, 'distance': distance}
    #             face_recognized = True  # 얼굴이 인식되었음을 표시

    #             # 가장 가까운 방송인 정보 업데이트
    #             if distance < closest_broadcaster['distance']:
    #                 closest_broadcaster = {'name': name, 'distance': distance, 'box': (int(x1), int(y1), int(x2), int(y2))}

    #     # 방송인 이름이 'unknown'이 아닐 경우, 다른 얼굴을 모자이크 처리
    #     if closest_broadcaster['name'] != 'unknown':
    #         for track_id, face_info in self.recognized_faces.items():
    #             if face_info['name'] != closest_broadcaster['name']:
    #                 # 모자이크 처리
    #                 if 'box' in face_info:  # 박스 정보가 있을 때만
    #                     face_box = face_info['box']
    #                     img = self.apply_mosaic(img, (face_box[0], face_box[1]), (face_box[2], face_box[3]))

    #     # 얼굴에 대한 처리
    #     for obj in tracked_objects:
    #         x1, y1, x2, y2, track_id = obj[:5]
    #         recognized_face = self.recognized_faces.get(track_id, {'name': 'unknown', 'distance': float("inf")})
    #         name = recognized_face['name']
    #         distance = recognized_face['distance']

    #         if name == 'unknown':
    #             img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
    #             cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    #             cv2.putText(img, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #         else:
    #             cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #             cv2.putText(img, f'{name} {distance:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #     # 프레임 카운트 증가 (얼굴이 인식된 경우에만)
    #     if face_recognized:
    #         self.face_frame_count += 1

    #     # 클래스별로 모자이크 처리 적용
    #     for det in results.xyxy[0]:
    #         x1, y1, x2, y2, conf, cls = det
    #         cls = int(cls)
    #         if conf < confidence_t:
    #             continue

    #         if cls == 0 and license_plate:  # 차량 번호판
    #             img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
    #         if cls == 1 and invoice:  # 송장
    #             img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
    #         if cls == 2 and id_card:  # ID 카드
    #             img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
    #         if cls == 3 and license_card:  # 라이센스 카드
    #             img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
    #         if cls == 4 and knife:  # 칼
    #             img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))

    #     return img

    def detect(self, img, license_plate, invoice, id_card, license_card, knife, face):
        global face_frame_count, broadcaster_id, broadcaster_frame_count, recognized_faces

        # 유효하지 않은 이미지일 경우 None 반환
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb)
        track_results = []
        face_detect = False

       

        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            if conf < confidence_t:
                continue

            if cls == 0 and license_plate:  # 차량 번호판
                img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            
            if cls == 1 and invoice:  # 송장
                img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            
            if cls == 2 and id_card:  # ID 카드
                img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            
            if cls == 3 and license_card:  # 라이센스 카드
                img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            
            if cls == 4 and knife:  # 칼
                img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
        
            if cls == 5 and face:  # 얼굴 클래스일 경우
                track_results.append([x1.item(), y1.item(), x2.item(), y2.item()])
                face_detect = True

        # SORT 알고리즘 적용 (얼굴 추적)
        if len(track_results) > 0:
            tracked_objects = self.tracker.update(np.array(track_results))
        else:
            tracked_objects = self.tracker.update(np.empty((0, 4)))


        # 방송인으로 설정할 얼굴을 위한 변수 초기화
        min_distance = float('inf')
        best_broadcaster_id = None

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj[:5]
            object_img = img[int(y1):int(y2), int(x1):int(x2)]

            if object_img.size == 0:
                logging.debug(f"Invalid face region: x1={x1}, y1={y1}, x2={x2}, y2={y2}. Skipping face.")
                continue

            # 30프레임마다 얼굴 인식 수행
            if self.face_frame_count % 30 == 0:
                logging.debug(f"Performing face recognition on frame {self.face_frame_count}.")

                encode_face = self.get_encode(object_img, required_size)
                if encode_face is None:
                    logging.debug("Face encoding failed, skipping.")
                    continue

                encode_face = l2_normalizer.transform(encode_face.reshape(1, -1))[0]
                name = 'unknown'
                distance = float('inf')

                # 사전에 저장된 인코딩과 비교
                for db_name, db_encode in self.encoding_dict.items():
                    dist = cosine(db_encode, encode_face)
                    if dist < recognition_t and dist < distance:
                        name = db_name
                        distance = dist

                # 인식된 얼굴을 저장 (트랙 ID로 관리)
                self.recognized_faces[track_id] = {'name': name, 'distance': distance}
                logging.debug(f"Track ID {track_id}, Name: {name}, Distance: {distance}")

                # Best broadcaster 업데이트 로직 추가
                if name != 'unknown' and (best_broadcaster_id is None or distance < self.recognized_faces[best_broadcaster_id]['distance']):
                    # 현재 프레임에서 방송인을 식별하면 갱신
                    self.recognized_faces[track_id] = {'name': name, 'distance': distance}
                    best_broadcaster_id = track_id
                    logging.debug(f"Best broadcaster updated: Track ID {track_id}, Name: {name}, Distance: {distance}")
                else:
                    if best_broadcaster_id is not None and best_broadcaster_id in self.recognized_faces:
                        best_broadcaster = self.recognized_faces[best_broadcaster_id]
                        logging.debug(f"Best broadcaster maintained: Track ID {best_broadcaster_id}, Name: {best_broadcaster['name']}, Distance: {best_broadcaster['distance']}")
                    else:
                        logging.debug("No previous broadcaster available to maintain.")

            else:
                if track_id in self.recognized_faces:
                    recognized_face = self.recognized_faces[track_id]
                    name = recognized_face['name']
                    distance = recognized_face['distance']
                else:
                    name = 'unknown'
                    distance = float('inf')

            # 가장 거리가 작은 트랙 ID를 방송인으로 설정
            if name != 'unknown' and distance < min_distance:
                min_distance = distance
                best_broadcaster_id = track_id

        # 방송인으로 설정된 트랙 ID가 있을 경우 표시
        if best_broadcaster_id is not None:
            broadcaster_info = self.recognized_faces[best_broadcaster_id]
            broadcaster_name = broadcaster_info['name']
            broadcaster_distance = broadcaster_info['distance']
            logging.debug(f"Best broadcaster: Track ID {best_broadcaster_id}, Name: {broadcaster_name}, Distance: {broadcaster_distance}")

            # 방송인 얼굴만 초록색 박스로 표시
            for obj in tracked_objects:
                x1, y1, x2, y2, track_id = obj[:5]
                if track_id == best_broadcaster_id:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f'{broadcaster_name} {broadcaster_distance:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(img, 'unknown', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if face_detect:
            self.face_frame_count += 1

        return img


    # def detect(self, img, license_plate, invoice, id_card, license_card, knife, face):
    #     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     results = self.model(img_rgb)

    #     track_results = []
    #     for det in results.xyxy[0]:
    #         x1, y1, x2, y2, conf, cls = det
    #         cls = int(cls)
    #         if conf < confidence_t:
    #             continue

    #         if cls == 5 and face:  # 얼굴 클래스일 경우
    #             track_results.append([x1.item(), y1.item(), x2.item(), y2.item()])  # 좌표만 추가

    #     # SORT 알고리즘 적용 (얼굴 추적)
    #     tracked_objects = self.tracker.update(np.array(track_results)) if len(track_results) > 0 else self.tracker.update(np.empty((0, 4)))

    #     closest_broadcaster = {'name': 'unknown', 'distance': float("inf"), 'box': None}

    #     for obj in tracked_objects:
    #         x1, y1, x2, y2, track_id = obj[:5]
    #         object_img = img[int(y1):int(y2), int(x1):int(x2)]

    #         # 얼굴 인식을 30프레임마다 수행
    #         if self.face_frame_count % 30 == 0:
    #             encode_face = self.get_encode(object_img, required_size)
    #             if encode_face is None:
    #                 continue  # 얼굴 인식 실패 시 건너뜀

    #             encode_face = l2_normalizer.transform(encode_face.reshape(1, -1))[0]
    #             name = 'unknown'
    #             distance = float("inf")

    #             for db_name, db_encode in self.encoding_dict.items():
    #                 dist = cosine(db_encode, encode_face)
    #                 if dist < recognition_t and dist < distance:
    #                     name = db_name
    #                     distance = dist

    #             # 인식된 얼굴 정보 저장
    #             self.recognized_faces[track_id] = {'name': name, 'distance': distance}

    #             # 가장 가까운 방송인 정보 업데이트
    #             if distance < closest_broadcaster['distance']:
    #                 closest_broadcaster = {'name': name, 'distance': distance, 'box': (int(x1), int(y1), int(x2), int(y2))}

    #     # 방송인 이름이 'unknown'이 아닐 경우, 다른 얼굴을 모자이크 처리
    #     if closest_broadcaster['name'] != 'unknown':
    #         for track_id, face_info in self.recognized_faces.items():
    #             if face_info['name'] != closest_broadcaster['name']:
    #                 # 모자이크 처리
    #                 if 'box' in face_info:  # 박스 정보가 있을 때만
    #                     face_box = face_info['box']
    #                     img = self.apply_mosaic(img, (face_box[0], face_box[1]), (face_box[2], face_box[3]))

    #     # 얼굴에 대한 처리
    #     for obj in tracked_objects:
    #         x1, y1, x2, y2, track_id = obj[:5]
    #         recognized_face = self.recognized_faces.get(track_id, {'name': 'unknown', 'distance': float("inf")})
    #         name = recognized_face['name']
    #         distance = recognized_face['distance']

    #         if name == 'unknown':
    #             img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
    #             cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    #             cv2.putText(img, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    #         else:
    #             cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    #             cv2.putText(img, f'{name} {distance:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #     # 프레임 카운트 증가
    #     self.face_frame_count += 1

    #     # 클래스별로 모자이크 처리 적용
    #     for det in results.xyxy[0]:
    #         x1, y1, x2, y2, conf, cls = det
    #         cls = int(cls)
    #         if conf < confidence_t:
    #             continue

    #         if cls == 0 and license_plate:  # 차량 번호판
    #             img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
    #         if cls == 1 and invoice:  # 송장
    #             img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
    #         if cls == 2 and id_card:  # ID 카드
    #             img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
    #         if cls == 3 and license_card:  # 라이센스 카드
    #             img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
    #         if cls == 4 and knife:  # 칼
    #             img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))

    #     return img

    

    def apply_mosaic(self, image, pt_1, pt_2, kernel_size=15):
        x1, y1 = pt_1
        x2, y2 = pt_2

        # 좌표를 이미지 경계 내로 클램핑
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)

         # 이미지 크기 및 좌표가 유효한지 확인
        if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0] or x2 <= x1 or y2 <= y1:
            logger.error(f"Invalid coordinates for mosaic: {pt_1}, {pt_2}")
            return image  # 잘못된 좌표일 경우 원본 이미지 반환
    
        #face_height, face_width, _ = image[y1:y2, x1:x2].shape
        

        face_region = image[y1:y2, x1:x2]

        # 얼굴 영역이 비어 있는지 확인
        if face_region.size == 0:
            logger.error("Empty face region encountered during mosaic.")
            return image  # 빈 영역일 경우 원본 이미지 반환

        face_height, face_width, _ = face_region.shape

        # 프레임 축소
        scale_factor = 0.5  # 프레임 크기를 줄이는 배율 (값을 조정 가능)
        small_face = cv2.resize(image[y1:y2, x1:x2], (int(face_width * scale_factor), int(face_height * scale_factor)))
        # 모자이크 처리
        small_face = cv2.resize(small_face, (kernel_size, kernel_size), interpolation=cv2.INTER_LINEAR)
        small_face = cv2.resize(small_face, (int(face_width * scale_factor), int(face_height * scale_factor)), interpolation=cv2.INTER_NEAREST)

        # 다시 원래 크기로 복원
        image[y1:y2, x1:x2] = cv2.resize(small_face, (face_width, face_height), interpolation=cv2.INTER_LINEAR)
        
        return image

    def get_encode(self, face, size):
         # 얼굴 이미지가 비어 있거나 None인 경우 처리
        if face is None or face.size == 0:
            logger.error("Empty or invalid face image passed to get_encode.")
            return None  # 빈 이미지일 경우 None 반환
    
        face = normalize(face)
        try:
            face = cv2.resize(face, size)
        except cv2.error as e:
            logger.error(f"Error resizing face image: {e}")
            return None  # 리사이즈 실패 시 None 반환

        encode = self.face_encoder.predict(np.expand_dims(face, axis=0))[0]
        return encode

    def stop(self):
        if self.process:
            try:
                logger.debug("Terminating FFmpeg process...")
                self.process.terminate()
                self.process.wait(timeout=5)
                logger.debug("FFmpeg process terminated.")
            except subprocess.TimeoutExpired:
                logger.error("FFmpeg process did not terminate within the timeout period.")
                self.process.kill()
            except Exception as e:
                logger.error(f"Error stopping FFmpeg process: {e}")
            finally:
                self.process = None

        if self.cap:
            try:
                logger.debug("Releasing VideoCapture...")
                self.cap.release()
                logger.debug("Destroying all OpenCV windows...")
            except Exception as e:
                logger.error(f"Error releasing VideoCapture: {e}")
       

def index(request):
    return render(request, 'webcam/index.html')

def find_possible_id():
    for i in range(CONN_LIMIT):
        if possible_list[i]:
            possible_list[i] = False
            return i
    return -1

@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def start_flutter_stream(request):
    print(f"Request body: {request.body}")
    available_id = find_possible_id()
    if available_id == -1:
        return JsonResponse({"error": "Connection limit reached"}, status=429)

    email = request.data.get("email")
    stream_key=request.data.get("streamKey")

    # 실시간 중 모자이크 처리 옵션 설정
    license_plate=request.data.get('mosaic_license_plate') == 'true'
    invoice = request.data.get('mosaic_invoice') == 'true'
    id_card = request.data.get('mosaic_id_card') == 'true'
    license_card = request.data.get('mosaic_license_card') == 'true'
    knife = request.data.get('mosaic_knife') == 'true'
    face = request.data.get('mosaic_face') == 'true'
    
    license_plate = True
    invoice = True
    id_card=True
    license_card=True
    knife=True
    face=True
    try:
        # # YOLO 모델이 로드되어 있지 않다면 로드
        # if yolo_model is None:
        #     load_global_model()
            
        if available_id in stream_instances:
            return JsonResponse({"error": f"Stream with ID {available_id} is already running"}, status=400)
        
        stream_instance = WebcamStream(available_id)

        # 사용자의 인코딩 파일을 로드
        #stream_instance.load_encoding_dict(email)
        
        # 스트림을 실행하기 위해 새로운 스레드 시작, 추가한 옵션들도 전달
        stream_instances[available_id] = stream_instance
        threading.Thread(target=start_stream_process, args=(available_id, email,stream_key, license_plate, invoice, id_card, license_card, knife, face)).start()
        return JsonResponse({'message': '웹캠 스트림을 시작합니다.', 'stream_id': str(available_id)})
    except Exception as e:
        logger.error(f"Error starting stream: {e}")
        return JsonResponse({"error": str(e)}, status=500)


def start_stream_process(stream_id, email, stream_key, license_plate, invoice, id_card, license_card, knife, face):
    try:
        stream_instance = stream_instances.get(stream_id)
        if stream_instance:
            # 옵션들을 넘겨서 시작
            stream_instance.start(email, stream_key, license_plate, invoice, id_card, license_card, knife, face)
    except Exception as e:
        logger.error(f"Error starting stream process: {e}")


@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def stop_flutter_stream(request):
    print(f"Request body: {request.body}")
    stream_id = int(request.data.get("stream_id"))
    
    if stream_id not in stream_instances:
        return JsonResponse({"error": "Invalid stream_id"}, status=400)

    try:
        # 해당 stream_id에 대한 스트림만 종료
        stream_instances[stream_id].stop()
        del stream_instances[stream_id]
        possible_list[stream_id] = True  # 해당 ID를 다시 사용할 수 있게 함
        
        return JsonResponse({"status": "Stream stopped successfully"})
    except Exception as e:
        logger.error(f"Error stopping stream: {e}")
        return JsonResponse({"error": str(e)}, status=500)

@csrf_exempt
@api_view(['GET'])
@permission_classes([AllowAny])
def get_id(request):
    active_streams = [i for i, available in enumerate(possible_list) if not available]
    
    # 활성 스트림이 없는 경우
    if not active_streams:
        return JsonResponse({"active_streams": [], "message": "No active streams"}, status=200)
    
    return JsonResponse({"active_streams": active_streams}, status=200)

import atexit

def cleanup():
    for stream_id in list(stream_instances.keys()):
        stream_instances[stream_id].stop()

atexit.register(cleanup)
#http://<nginx_server_ip>/live-out/streamkey.m3u8
