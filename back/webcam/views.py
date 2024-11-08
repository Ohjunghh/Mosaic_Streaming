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
recognition_t = 0.32
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
        self.output_filename = f"output_{self.stream_id}.mp4"

       
        self.track_id_distances = {}  # 각 track_id에 대한 거리값을 저장
        self.previous_broadcaster_id = None  # 이전 방송인의 track_id
        self.broadcaster_frame_count = 0  # 방송인 결과를 유지할 프레임 카운트

        self.broadcaster_persistence_count = 0  # 방송인 결과를 유지할 프레임 수를 카운팅
        self.last_broadcaster_id = None  # 마지막으로 인식된 방송인 ID 저장
        self.last_broadcaster_distance = None  # 마지막으로 인식된 방송인 거리 저장
        

        # 10/30 새로운 함수 변수 추가
        self.broadcaster_id = None  # 방송자 ID를 저장하는 변수
        self.broadcaster_persistence_frames = 30  # 방송자로 인식된 후 유지할 프레임 수
        self.broadcaster_track_id = None  # 현재 방송자의 트랙 ID
        self.broadcaster_frame_count = 0  # 방송자로 인식된 후 유지할 프레임 카운트
        self.face_frame_count = 0  
        self.recognized_faces={}
        self.best_broadcaster_id = None  # best_broadcaster_id를 전역으로 초기화
        self.SOME_THRESHOLD = 50#30
        self.face_recognition_needed = False  # 얼굴 인식이 필요한지 여부를 결정하는 플래그
        self.tracked_objects = []  # tracked_objects 초기화
        self.no_detection_frames = 0 

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
            self.cap = cv2.VideoCapture(f"rtmp://192.168.0.110/live/{stream_key}")
            
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
                f'rtmp://192.168.0.110/live-out/{self.stream_id}'  # 스트리밍 대상 RTMP URL
            ]

            # 비디오 파일 저장을 위한 VideoWriter 객체 생성
            self.video_writer = cv2.VideoWriter(self.output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            

            stream_opened = True  # 스트림이 정상적으로 열렸음을 기록
            self.start_ffmpeg()

           

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Unable to read frame.")
                    break
                
                #모자이크 
                frame = self.detect(frame,license_plate, invoice, id_card, license_card, knife, face)
        
                # 모자이크된 프레임 저장
                if self.video_writer:
                    self.video_writer.write(frame)

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


    #최종
    def detect(self, img, license_plate, invoice, id_card, license_card, knife, face):
        # Check for valid image
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb)
        track_results = []
        face_detect = False

        # Detect objects and faces
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            if conf < confidence_t:
                continue

            if cls == 0 and license_plate:
                img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            
            if cls == 1 and invoice:
                img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            
            if cls == 2 and id_card:
                img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            
            if cls == 3 and license_card:
                img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            
            if cls == 4 and knife:
                img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            
            if cls == 5 and face:
                track_results.append([x1.item(), y1.item(), x2.item(), y2.item()])
                face_detect = True
                
        # Apply SORT algorithm for tracking faces
        if len(track_results) > 0:
            tracked_objects = self.tracker.update(np.array(track_results))
        else:
            tracked_objects = self.tracker.update(np.empty((0, 4)))

        # Variables to track best broadcaster
        min_distance = float('inf')
        best_broadcaster_id = None

        # Process each tracked face
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj[:5]
            object_img = img[int(y1):int(y2), int(x1):int(x2)]
            
             # 잘못된 얼굴 영역 처리
            if object_img.size == 0 or x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
                logging.debug(f"잘못된 얼굴 영역: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                if track_id != self.previous_broadcaster_id:  # 이전 방송인이 아닐 경우 모자이크 적용
                    img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
                    logging.debug(f"잘못된 얼굴 영역이지만 이전방송인이 아니므로 모자이크를 적용합니다.")
                continue  # 잘못된 얼굴 영역일 경우 건너뜀


            # Perform face recognition every 30 frames
            if self.face_frame_count % 50 == 0 or  self.face_frame_count == 3:
                logging.debug(f"Performing face recognition on frame {self.face_frame_count}.")

                encode_face = self.get_encode(object_img, required_size)
               
                if encode_face is None:
                    logging.debug(f"Track ID {track_id}의 얼굴 인식이 실패했습니다.")
                    if track_id != self.previous_broadcaster_id:  # 이전 방송인이 아닌 경우에만 모자이크 적용
                        logging.debug(f"Track ID {track_id}는 이전 방송인이 아니므로 모자이크를 적용합니다.")
                        img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
                    continue

                encode_face = l2_normalizer.transform(encode_face.reshape(1, -1))[0]
                name = 'unknown'
                distance = float('inf')

                # Compare the face encoding with the stored encoding dictionary
                for db_name, db_encode in self.encoding_dict.items():
                    dist = cosine(db_encode, encode_face)
                    if dist < recognition_t and dist < distance:
                        name = db_name
                        distance = dist

                # Store recognized faces with track ID
                self.recognized_faces[track_id] = {'name': name, 'distance': distance}
                logging.debug(f"Track ID {track_id}, Name: {name}, Distance: {distance}")

                # Update the best broadcaster based on recognition
                if name != 'unknown' and (self.best_broadcaster_id is None or distance < self.recognized_faces[self.best_broadcaster_id]['distance']+0.09):
                    self.best_broadcaster_id = track_id
                    self.previous_broadcaster_id = self.best_broadcaster_id  # Remember the broadcaster
                    #logging.debug(f"Best broadcaster updated: Track ID {track_id}, Name: {name}, Distance: {distance}")
                else:
                    # If no new broadcaster, maintain the previous one
                    if self.previous_broadcaster_id is not None and self.previous_broadcaster_id in self.recognized_faces:
                        self.best_broadcaster_id = self.previous_broadcaster_id
                        #logging.debug(f"Best broadcaster maintained: Track ID {self.previous_broadcaster_id}")

            else:
                # Continue using the previous broadcaster if no new face is recognized
                if self.previous_broadcaster_id is not None and self.previous_broadcaster_id in self.recognized_faces:
                    self.best_broadcaster_id = self.previous_broadcaster_id
                    #logging.debug(f"Continuing to use previous broadcaster ID: {self.previous_broadcaster_id}")

        # Highlight the broadcaster face if detected
        if self.best_broadcaster_id is not None:
            broadcaster_info = self.recognized_faces[self.best_broadcaster_id]
            broadcaster_name = broadcaster_info['name']
            broadcaster_distance = broadcaster_info['distance']
            #logging.debug(f"Best broadcaster: Track ID {best_broadcaster_id}, Name: {broadcaster_name}, Distance: {broadcaster_distance}")

            # Highlight broadcaster with a green box, and apply mosaic to others
            for obj in tracked_objects:
                x1, y1, x2, y2, track_id = obj[:5]
                if track_id == self.best_broadcaster_id:
                    pass
                else:
                    img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
                    logging.debug(f"베스트 방송인아니어서 모자이크 베스트 방송인 : {self.best_broadcaster_id}")
                   
        else:
            if len(tracked_objects) > 0:
                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id = obj[:5]
                    img = self.apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
                    logging.debug(f"베스트 방송인 없어서 모자이크")
                
                
        if face_detect:
            self.face_frame_count += 1

        return img
  
    
    def apply_mosaic(self,image, pt_1, pt_2, kernel_size=15):
        x1, y1 = pt_1
        x2, y2 = pt_2

        # 얼굴 영역을 계산
        face = image[y1:y2, x1:x2]

        # 얼굴 영역이 비어있지 않도록 확인
        if face.size == 0:
            return image

        # 실제로 적용하려는 영역의 크기 계산
        face_height, face_width = face.shape[:2]

        # 모자이크 처리
        face = cv2.resize(face, (kernel_size, kernel_size), interpolation=cv2.INTER_LINEAR)  # 얼굴을 작게 축소
        face = cv2.resize(face, (face_width, face_height), interpolation=cv2.INTER_NEAREST)  # 다시 원래 크기로 확대

        # 원본 이미지에 모자이크 적용 (사이즈 맞지 않으면 맞춤)
        if face.shape == image[y1:y2, x1:x2].shape:
            image[y1:y2, x1:x2] = face
        else:
            # 크기가 맞지 않으면 재조정하여 맞춤
            h, w = image[y1:y2, x1:x2].shape[:2]
            face = cv2.resize(face, (w, h))
            image[y1:y2, x1:x2] = face

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
            
        if available_id in stream_instances:
            return JsonResponse({"error": f"Stream with ID {available_id} is already running"}, status=400)
        
        stream_instance = WebcamStream(available_id)

        
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
