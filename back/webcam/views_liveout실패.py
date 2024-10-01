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

# Constants and configurations
confidence_t = 0.5
recognition_t = 0.35
required_size = (160, 160)
CONN_LIMIT = 10
possible_list = [True] * CONN_LIMIT
stream_instances = {}
logger = logging.getLogger(__name__)

# Logger setup
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
        self.command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', "1280x720", #"854x480",  # 480p 해상도
            '-r', '15',
            '-i', '-',
            '-c:v', 'h264_nvenc',  # GPU 인코딩으로 변경
            #'-c:v', 'libx264',  # CPU 인코딩 #'-c:v', 'h264_nvenc',  # GPU 인코딩으로 변경
            '-pix_fmt', 'yuv420p',
            #'-preset', 'ultrafast', #GPU시 # h264_nvenc는 `ultrafast`가 없으므로 'fast' 혹은 'p7'을 사용
            '-preset', 'fast',
            '-f', 'flv',
            '-c:a', 'aac',  # 오디오 코덱?
            #'-b:a', '128k',  # 오디오 비트레이트
            f'rtmp://172.20.71.121/live-out/{self.stream_id}'#'rtmp://192.168.58.194/live-out/{self.stream_id}'
        ]

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

    def start(self, email, stream_key, invoice, id_card, knife, face):
        stream_opened = False  # 스트림이 정상적으로 열렸는지 여부를 추적하는 플래그

        #self.cap = cv2.VideoCapture("rtmp://172.24.240.191/live/live")
        #self.cap = cv2.VideoCapture("rtmp://broadcast.api.video/s/23afe2eb-e5e9-4158-818b-395782ff2c34")
        #self.cap = cv2.VideoCapture("rtmp://192.168.248.241/stream/live")
        #self.cap = cv2.VideoCapture("rtmp://172.30.1.26/stream/live")
        try:
            #self.load_model()
            self.load_face_encoder()
            self.load_encoding_dict(email)
            #self.cap = cv2.VideoCapture(f"rtmp://192.168.174.14/live/{stream_key}")
            self.cap = cv2.VideoCapture(f"rtmp://172.20.71.121/live/{stream_key}")
            
            if not self.cap.isOpened():
                logger.error("Unable to open RTMP stream.")
                self.stop()  # 스트림을 종료
                return
            
            stream_opened = True  # 스트림이 정상적으로 열렸음을 기록
            self.start_ffmpeg()

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Unable to read frame.")
                    break

                frame = self.detect(frame,invoice, id_card, knife, face)
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

    def detect(self, img, invoice, id_card, knife, face):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb)
        
        
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            if conf < confidence_t:
                continue

            object_img = img[int(y1):int(y2), int(x1):int(x2)]
            if cls == 4 and face:
                encode_face = self.get_encode(object_img, required_size)
                encode_face = l2_normalizer.transform(encode_face.reshape(1, -1))[0]

                name = 'unknown'
                distance = float("inf")
                for db_name, db_encode in self.encoding_dict.items():
                    dist = cosine(db_encode, encode_face)
                    if dist < recognition_t and dist < distance:
                        name = db_name
                        distance = dist

                if name == 'unknown':
                    img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(img, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f'{name} {distance:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 클래스별로 모자이크 처리 적용
            license_plate = True
            invoice = True
            id_card=True
            if cls == 0 and license_plate: # 차량 번호판 
                img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            if cls == 1 and invoice:  # 송장
                img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            if cls == 2 and id_card:  # ID 카드
                img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            if cls == 3 and knife:  # 칼
                img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))

        return img

    def apply_mosaic(self, image, pt_1, pt_2, kernel_size=15):
        x1, y1 = pt_1
        x2, y2 = pt_2
        face_height, face_width, _ = image[y1:y2, x1:x2].shape
        
        # face = image[y1:y1 + face_height, x1:x1 + face_width]
        # face = cv2.resize(face, (kernel_size, kernel_size), interpolation=cv2.INTER_LINEAR)
        # face = cv2.resize(face, (face_width, face_height), interpolation=cv2.INTER_NEAREST)
        #image[y1:y1 + face_height, x1:x1 + face_width] = face
        
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
        face = normalize(face)
        face = cv2.resize(face, size)
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
    invoice = request.data.get('mosaic_delivery_slip') == 'true'
    id_card = request.data.get('mosaic_id_card') == 'true'
    knife = request.data.get('mosaic_knife') == 'true'
    face = request.data.get('mosaic_face') == 'true'

    #일단
    face=True

    try:
        if available_id in stream_instances:
            return JsonResponse({"error": f"Stream with ID {available_id} is already running"}, status=400)
        
        stream_instance = WebcamStream(available_id)
        
        # 스트림을 실행하기 위해 새로운 스레드 시작, 추가한 옵션들도 전달
        stream_instances[available_id] = stream_instance
        threading.Thread(target=start_stream_process, args=(available_id, email,stream_key, invoice, id_card, knife, face)).start()
        return JsonResponse({'message': '웹캠 스트림을 시작합니다.', 'stream_id': str(available_id)})
    except Exception as e:
        logger.error(f"Error starting stream: {e}")
        return JsonResponse({"error": str(e)}, status=500)


def start_stream_process(stream_id, email, stream_key, invoice, id_card, knife, face):
    try:
        stream_instance = stream_instances.get(stream_id)
        if stream_instance:
            # 옵션들을 넘겨서 시작
            stream_instance.start(email, stream_key, invoice, id_card, knife, face)
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


import atexit

def cleanup():
    for stream_id in list(stream_instances.keys()):
        stream_instances[stream_id].stop()

atexit.register(cleanup)
#http://<nginx_server_ip>/live-out/streamkey.m3u8
