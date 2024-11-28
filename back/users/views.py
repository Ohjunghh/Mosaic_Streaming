from django.contrib.auth import authenticate, login
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .face_image_save import save_encoding_vector#,save_face_images
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework import status
from rest_framework.permissions import AllowAny
from .serializers import UserRegistrationSerializer
from .models import CustomUser  # 수정한 사용자 모델
from .models import UserImage
from .aboutvideo import VideoProcessor
from yolo_loader import yolo_model  # 전역 YOLO 모델 사용
from django.conf import settings
from django.http import FileResponse
from .models import ProcessedVideo, ProfileImage  # 모델 임포트
import os
import json
import cv2
import jwt
import torch
from django.conf import settings


@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def signup(request): 
    print("Received data from frontend:", request.data)
    serializer = UserRegistrationSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response({'status': 'success'}, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def signin(request): 
    if request.method == 'POST':
        # 요청에서 로그인 정보 추출
        #Json으로 올 때
        data = json.loads(request.body)
        print("Received data from frontend:",data)
        email = data.get('email')
        password = data.get('password')

        # 사용자 인증
        user = authenticate(request, email=email, password=password)
        if user is not None:
            login(request, user)
            # JWT 토큰 생성
            refresh = RefreshToken.for_user(user)
            access_token = str(refresh.access_token)
            refresh_token = str(refresh) 
            
            return JsonResponse({'status': 'success', 'user_id': user.id,'access': access_token,'refresh': refresh_token,'email':email })
        else:
            return JsonResponse({'status': 'error', 'message': 'Invalid credentials'}, status=401)

@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def register_face(request):
    if request.method == 'POST':
        #user_id = request.username
        email = request.data.get('email')  # POST 요청에서 'email' 키의 값을 가져옴

        if not email:
            return JsonResponse({'status': 'error', 'message': 'Username not provided'}, status=400)
        
        # 전달받은 'username'으로 사용자 검색
        try:
            user = CustomUser.objects.get(email=email)#username을 사용하여 사용자를 검색하면 CustomUser 모델의 인스턴스(즉, 사용자 객체)를 반환
        except CustomUser.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'User not found'}, status=404)


        # 여러 이미지 파일을 받음
        image_files = request.FILES.getlist('faces')
        if not image_files:
            return JsonResponse({'status': 'error', 'message': 'No images provided'}, status=400)
        
        # 얼굴 사진 저장
        #save_face_images(image_files, user_id)
        for image in image_files:
            #user_image = UserImage(user=request.user, image=image)
            user_image = UserImage(user=user, image=image)
            user_image.save()

        # 얼굴 인코딩 벡터 저장
        encoding_path = save_encoding_vector(user.id)
        
        return JsonResponse({'status': 'success', 'encoding_file': encoding_path})

import subprocess

# 중복된 파일 이름이 있을 때 숫자를 붙이는 함수
def get_unique_video_name(directory, name):
    base_name = name
    counter = 1
    video_name = f"{base_name}.mp4"
    
    while os.path.exists(os.path.join(directory, video_name)):
        video_name = f"{base_name}({counter}).mp4"
        counter += 1
    
    return video_name


#from VToonify.style_transfer import detect_and_stylize


@csrf_exempt
@api_view(['POST'])
@permission_classes([AllowAny])
def video_upload(request):
    print("Request data:", request.data)

    # 비디오 처리 옵션 설정
    license_plate=request.data.get('mosaic_license_plate') == 'true'
    invoice = request.data.get('mosaic_invoice') == 'true'
    id_card = request.data.get('mosaic_id_card') == 'true'
    license_card = request.data.get('mosaic_license_card') == 'true'
    knife = request.data.get('mosaic_knife') == 'true'
    face = request.data.get('mosaic_face') == 'true'
    avatar = request.data.get('avatar_processing') == 'true'

    email = request.data.get("email")
    try:
        user = CustomUser.objects.get(email=email)
    except CustomUser.DoesNotExist:
        return JsonResponse({'error': 'User not found'}, status=404)

    video_file = request.FILES.get('video')
    if not video_file:
        return JsonResponse({'error': 'No video file provided'}, status=400)

    name = request.data.get('name')
    user_video_dir = os.path.join(settings.MEDIA_ROOT, 'video', str(user.id))
    os.makedirs(user_video_dir, exist_ok=True)

    # 중복된 이름 체크 후 고유한 이름 생성
    original_video_name = get_unique_video_name(user_video_dir, name)
    original_video_path = os.path.join(user_video_dir, original_video_name)

    # 원본 비디오 파일 저장
    with open(original_video_path, 'wb+') as destination:
        for chunk in video_file.chunks():
            destination.write(chunk)

    processor=VideoProcessor(user.id)

    # YOLO 및 FaceNet 모델 로드
    model = yolo_model
    face_encoder=processor.face_encoder
    encoding_dict=processor.encoding_dict

    if not encoding_dict:
        return JsonResponse({'error': 'Encoding file not found'}, status=400)

    # 처리 후 저장될 비디오 파일 경로
    processed_video_path = f"{os.path.splitext(original_video_name)[0]}_processed.mp4"
    processed_video_path_full = os.path.join(user_video_dir, processed_video_path)

    
    cap = cv2.VideoCapture(original_video_path)
    if not cap.isOpened():
        return JsonResponse({'error': 'Failed to open video file'}, status=500)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    ret, frame = cap.read()
    if not ret:
        return JsonResponse({'error': 'Failed to read video frame'}, status=500)

    height, width, _ = frame.shape
    out = cv2.VideoWriter(processed_video_path_full, fourcc, 20.0, (width, height))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device : ",device)
    style_degree = 0.5  # 또는 다른 값으로 설정
  
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if avatar:
            from VToonify.style_transfer import detect_and_stylize
            frame = detect_and_stylize(frame, model, face_encoder, encoding_dict, device, style_degree, license_plate, invoice, id_card, license_card, knife, face)
        else:
            frame = processor.detect_and_mosaic(frame, model, license_plate, invoice, id_card, license_card, knife, face)
            
        out.write(frame)

    cap.release()
    out.release()

    # FFmpeg로 비디오 변환
    base_ffmpeg_name = f"ffmpeg_{os.path.splitext(original_video_name)[0]}"
    ffmpeg_video_name = get_unique_video_name(user_video_dir, base_ffmpeg_name)
    ffmpeg_video_path = os.path.join(user_video_dir, ffmpeg_video_name)

    convert_video_with_ffmpeg(processed_video_path_full, ffmpeg_video_path)

    # URL 생성
    original_video_url = f"http://{request.get_host()}{settings.MEDIA_URL}video/{user.id}/{original_video_name}"
    processed_video_url = f"http://{request.get_host()}{settings.MEDIA_URL}video/{user.id}/{ffmpeg_video_name}"

    # 처리된 비디오 정보를 모델에 저장
    ProcessedVideo.objects.create(
        user=user,
        name=ffmpeg_video_name,
        url=processed_video_url,
    )

    return JsonResponse({
        'status': 'success',
        'video_url': original_video_url,
        'processed_video_url': processed_video_url,
    })

# FFmpeg로 비디오 변환
def convert_video_with_ffmpeg(input_path, output_path):
    command = [
        'ffmpeg', 
        '-i', input_path, 
        '-filter:v', 'setpts=0.6897*PTS',  # 비디오 배속 설정 (1.45배속)
        '-filter:a', 'atempo=1.45',          # 오디오 배속 설정
        '-vcodec', 'libx264', 
        '-acodec', 'aac', 
        '-strict', '-2', 
        '-stats', 
        output_path
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Conversion successful: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg conversion: {str(e)}")
        raise RuntimeError("FFmpeg conversion failed")

def generate_thumbnail(video_path, thumbnail_path):
    # FFmpeg 명령어로 1초 지점에서 한 프레임 추출
    command = ['ffmpeg', '-i', video_path, '-ss', '00:00:01.000', '-vframes', '1', thumbnail_path]
    try:
        subprocess.run(command, check=True)
        print(f"Thumbnail generated: {thumbnail_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating thumbnail: {str(e)}")
        raise RuntimeError("Thumbnail generation failed") 
    
@csrf_exempt
@api_view(['GET'])
@permission_classes([AllowAny])
def video_download(request):
    email = request.GET.get("email")
    try:
        user = CustomUser.objects.get(email=email)
    except CustomUser.DoesNotExist:
        return JsonResponse({'error': 'User not found'}, status=404)

    # 사용자별 처리된 비디오 목록 가져오기
    processed_videos = ProcessedVideo.objects.filter(user=user)
    # 디버깅을 위해 비디오 목록을 출력
    print(f"Found {processed_videos.count()} videos for user {email}")
    
    # 비디오가 없을 때 처리
    if not processed_videos.exists():
        return JsonResponse({'processed_videos': []}, status=200)
    
    # ProcessedVideo 모델 직렬화
    video_urls = [{'video_url': video.url, 'name': video.name} for video in processed_videos]
    print(f"Returning {len(video_urls)} videos")
    return JsonResponse({'processed_videos': video_urls}, status=200)



@csrf_exempt
@api_view(['GET', 'POST'])
@permission_classes([AllowAny])
def profile(request):
    if request.method == 'POST':
        # POST 요청 처리: 프로필 이미지 업로드 로직
        token = request.META.get('HTTP_AUTHORIZATION')
        if token is None or not token.startswith('Bearer '):
            return JsonResponse({'status': 'error', 'message': 'Authorization header missing or malformed'}, status=401)

        access_token = token.split(' ')[1]

        try:
            payload = jwt.decode(access_token, settings.SECRET_KEY, algorithms=["HS256"])
            user = CustomUser.objects.get(id=payload['user_id'])
        except jwt.ExpiredSignatureError:
            return JsonResponse({'status': 'error', 'message': 'Token has expired'}, status=401)
        except (jwt.InvalidTokenError, CustomUser.DoesNotExist):
            return JsonResponse({'status': 'error', 'message': 'Invalid token'}, status=401)

        profile_image = request.FILES.get('profile_image')
        if profile_image:
            profile_image_instance = ProfileImage(user=user, image=profile_image, is_primary=True)
            profile_image_instance.save()

            ProfileImage.objects.filter(user=user).exclude(id=profile_image_instance.id).update(is_primary=False)

            profile_image_url = f"http://{request.get_host()}{profile_image_instance.image.url}"
            return JsonResponse({
                'status': 'success',
                'profile_image': profile_image_url,
                'message': 'Profile image uploaded successfully.'
            }, status=200)

        return JsonResponse({'status': 'error', 'message': 'No image provided'}, status=400)

    elif request.method == 'GET':
        # GET 요청 처리: 사용자 프로필 정보 로드
        token = request.META.get('HTTP_AUTHORIZATION')
        if token is None or not token.startswith('Bearer '):
            return JsonResponse({'status': 'error', 'message': 'Authorization header missing or malformed'}, status=401)

        access_token = token.split(' ')[1]

        try:
            payload = jwt.decode(access_token, settings.SECRET_KEY, algorithms=["HS256"])
            user = CustomUser.objects.get(id=payload['user_id'])
        except jwt.ExpiredSignatureError:
            return JsonResponse({'status': 'error', 'message': 'Token has expired'}, status=401)
        except (jwt.InvalidTokenError, CustomUser.DoesNotExist):
            return JsonResponse({'status': 'error', 'message': 'Invalid token'}, status=401)

        profile_image_instance = ProfileImage.objects.filter(user=user, is_primary=True).first()
        profile_image_url = profile_image_instance.image.url if profile_image_instance else None

        return JsonResponse({
            'status': 'success',
            'profile_image': f"http://{request.get_host()}{profile_image_url}" if profile_image_url else None
        }, status=200)

    return JsonResponse({'status': 'error', 'message': 'Invalid request method.'}, status=405)
