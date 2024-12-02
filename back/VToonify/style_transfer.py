import cv2
import numpy as np
import torch
from scipy.spatial.distance import cosine
import pickle
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from .model.vtoonify import VToonify
from .util import tensor2cv2, load_psp_standalone
import dlib
from VToonify.model.bisenet.model import BiSeNet
import sys
from face_recognition_2.architecture import InceptionResNetV2
from face_recognition_2.train_v2_1 import normalize, l2_normalizer
import argparse

confidence_t = 0.5  # YOLOv5 탐지 신뢰도 임계값
recognition_t = 0.35  # 얼굴 인식 거리 임계값
required_size = (160, 160)



# 얼굴 인코딩 계산
def get_encode(face_encoder, face, size):
    try:
        face = normalize(face)
        face = cv2.resize(face, size)
        encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
        return encode
    except Exception as e:
        print(f"얼굴 인코딩 계산 중 에러 발생: {e}")
        return None  # 에러가 발생할 경우 None 반환
    
def transfer_color(original_image, stylized_image):
    """원본 이미지의 색상 정보를 스타일화된 이미지에 적용"""
    # 원본과 스타일화된 이미지를 LAB 색 공간으로 변환
    original_lab = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
    stylized_lab = cv2.cvtColor(stylized_image, cv2.COLOR_BGR2LAB)

    # 원본의 색상(A, B) 정보를 스타일화된 이미지에 적용
    l, a, b = cv2.split(stylized_lab)
    _, original_a, original_b = cv2.split(original_lab)

    # 스타일화된 이미지의 L 채널(밝기 정보)을 유지하고, A, B 채널(색상 정보)을 원본으로 교체
    combined_lab = cv2.merge([l, original_a, original_b])

    # LAB 색 공간을 다시 BGR로 변환
    result_image = cv2.cvtColor(combined_lab, cv2.COLOR_LAB2BGR)
    return result_image

def apply_masked_face(original_frame, stylized_face, face_center, face_size, mask_blur=15):
    """
    스타일화된 얼굴을 원본 이미지에 부드럽게 합성하는 함수 (원본 색상 적용)
    """
    x1, y1 = max(0, int(face_center[0] - face_size[0] // 2)), max(0, int(face_center[1] - face_size[1] // 2))
    x2, y2 = min(original_frame.shape[1], x1 + face_size[0]), min(original_frame.shape[0], y1 + face_size[1])

    # 얼굴 영역의 크기를 맞추기 위해 얼굴 영역을 원본 프레임에서 잘라냄
    original_face_region = original_frame[y1:y2, x1:x2]

    # 원본 얼굴 영역 크기에 맞게 스타일화된 얼굴 크기를 조정
    stylized_face_resized = cv2.resize(stylized_face, (original_face_region.shape[1], original_face_region.shape[0]))

    # 원본 얼굴의 색상 정보를 스타일화된 얼굴에 적용
    stylized_face_resized = transfer_color(original_face_region, stylized_face_resized)

    # 얼굴 영역의 마스크 생성 (원형 마스크)
    mask = np.zeros_like(stylized_face_resized, dtype=np.float32)
    cv2.circle(mask, (mask.shape[1] // 2, mask.shape[0] // 2), min(mask.shape[0], mask.shape[1]) // 2, (1.0, 1.0, 1.0), -1)

    # 마스크 블러링 (경계를 부드럽게)
    mask = cv2.GaussianBlur(mask, (mask_blur, mask_blur), 0)

    # 마스크를 적용해 원본 배경과 부드럽게 합성
    blended_face = (stylized_face_resized * mask + original_face_region * (1 - mask)).astype(np.uint8)
    original_frame[y1:y2, x1:x2] = blended_face

    return original_frame

# YOLO로 얼굴 탐지 함수
def detect_faces_yolo(model, img, confidence_t=0.5, face_class=4, upscale_factor=2):
    img_upscaled = cv2.resize(img, (img.shape[1] * upscale_factor, img.shape[0] * upscale_factor))
    img_rgb = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)

    faces = []
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        if conf >= confidence_t and cls == face_class:
            faces.append((
                int(x1 / upscale_factor),
                int(y1 / upscale_factor),
                int(x2 / upscale_factor),
                int(y2 / upscale_factor)
            ))
    return faces

# Dlib으로 랜드마크 탐지 함수
def detect_landmarks_dlib(image, predictor, x1, y1, x2, y2, upscale_factor=2):
    # 좌표 값을 int로 변환
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    face_roi = image[y1:y2, x1:x2]
    face_roi_upscaled = cv2.resize(face_roi, (face_roi.shape[1] * upscale_factor, face_roi.shape[0] * upscale_factor))
    gray = cv2.cvtColor(face_roi_upscaled, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1)

    if len(rects) == 0:
        print("No landmarks detected in the face region")
        return None

    for rect in rects:
        shape = predictor(gray, rect)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])

    landmarks[:, 0] = landmarks[:, 0] / upscale_factor + x1
    landmarks[:, 1] = landmarks[:, 1] / upscale_factor + y1

    return landmarks

# 얼굴 정렬 함수
def align_face(image, landmarks):
    lm_eye_left = landmarks[36:42]
    lm_eye_right = landmarks[42:48]

    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (landmarks[48] + landmarks[54]) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = Image.fromarray(image)
    img = img.transform((256, 256), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)

    return img, quad, c

# 얼굴 탐지 및 인식에서 unknown 얼굴에 VToonify 스타일 적용
# 모자이크 적용 함수
def apply_mosaic(img, top_left, bottom_right, mosaic_factor=15):
    try:
        """모자이크를 적용하는 함수"""
        (x1, y1) = top_left
        (x2, y2) = bottom_right
        face_region = img[y1:y2, x1:x2]

        # 모자이크 처리 (크기를 줄인 후 다시 확대)
        small = cv2.resize(face_region, (face_region.shape[1] // mosaic_factor, face_region.shape[0] // mosaic_factor))
        mosaic_face = cv2.resize(small, (face_region.shape[1], face_region.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 원본 이미지에 모자이크 얼굴을 적용
        img[y1:y2, x1:x2] = mosaic_face
        return img
    except Exception as e:
        print(f"Error in apply_mosaic: {e}")
        return img  # 에러 발생 시 원본 이미지를 그대로 반환


def detect_and_stylize(img, model, face_encoder, encoding_dict,  device, style_degree, license_plate, invoice, id_card, license_card, knife, face):
    print("Starting detect_and_stylize...")
    try:
        # VToonify 및 관련 모델 초기화
        print("Initializing VToonify...")
        vtoonify = VToonify(backbone='dualstylegan')
        vtoonify.load_state_dict(torch.load('./VToonify/checkpoint/vtoonify_d_conan/vtoonify_s_d_c.pt', map_location=device)['g_ema'])
        vtoonify.to(device)
        print("VToonify model loaded.")
        
        pspencoder = load_psp_standalone('./VToonify/checkpoint/encoder.pt', device)
        exstyles = np.load('./VToonify/checkpoint/vtoonify_d_conan/exstyle_code.npy', allow_pickle=True).item()
        style_id = 26
        exstyle = torch.tensor(exstyles[list(exstyles.keys())[style_id]]).to(device)
        exstyle = vtoonify.zplus2wplus(exstyle)

        # Dlib 랜드마크와 얼굴 파싱 모델 로드
        print("Loading Dlib predictor...")
        predictor = dlib.shape_predictor('./VToonify/checkpoint/shape_predictor_68_face_landmarks.dat')
        parsingpredictor = BiSeNet(n_classes=19)
        parsingpredictor.load_state_dict(torch.load('./VToonify/checkpoint/faceparsing.pth', map_location=device))
        parsingpredictor.to(device).eval()
        print("Dlib predictor loaded.")
        
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("Running YOLO detection...")
        results = model(img_rgb)  # YOLO 탐지 수행

        print("Processing detection results...")
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            if conf < confidence_t:
                continue

            object_img = img[int(y1):int(y2), int(x1):int(x2)]

            if cls == 5 and face:  # 얼굴 클래스일 경우
                encode_face = get_encode(face_encoder, object_img, required_size)
                encode_face = l2_normalizer.transform(encode_face.reshape(1, -1))[0]

                name = 'unknown'
                distance = float("inf")
                for db_name, db_encode in encoding_dict.items():
                    dist = cosine(db_encode, encode_face)
                    if dist < recognition_t and dist < distance:
                        name = db_name
                        distance = dist

                if name == 'unknown':
                    print(f"Applying VToonify to unknown face at ({x1}, {y1})")
                    landmarks = detect_landmarks_dlib(img, predictor, x1, y1, x2, y2)
                    if landmarks is None:
                        continue

                    aligned_face, quad, face_center = align_face(img, landmarks)
                    face_tensor = transform(aligned_face).unsqueeze(dim=0).to(device)

                    with torch.no_grad():
                        x_p = F.interpolate(parsingpredictor(2 * (F.interpolate(face_tensor, scale_factor=2, mode='bilinear', align_corners=False)))[0],
                                            scale_factor=0.5, recompute_scale_factor=False).detach()
                        inputs = torch.cat((face_tensor, x_p / 16.), dim=1)
                        s_w = pspencoder(face_tensor)
                        s_w = vtoonify.zplus2wplus(s_w)
                        s_w[:, :7] = exstyle[:, :7]
                        y_tilde = vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s=style_degree)
                        y_tilde = torch.clamp(y_tilde, -1, 1)

                    stylized_face_np = tensor2cv2(y_tilde[0].cpu())
                    stylized_face_np_bgr = cv2.cvtColor(stylized_face_np, cv2.COLOR_RGB2BGR)

                    face_width = int(quad[2][0] - quad[0][0])
                    face_height = int(quad[2][1] - quad[0][1])
                    img = apply_masked_face(img, stylized_face_np_bgr, face_center, (face_width, face_height))

                else:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f'{name} {distance:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


            if cls == 0 and license_plate:  # 차량 번호판 클래스가 선택된 경우
                img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
                
            if cls == 1 and invoice:  # invoice(송장)클래스가 선택된 경우
                img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
                
            if cls == 2 and id_card:  # id_card 클래스가 선택된 경우
                img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
    
            if cls == 3 and license_card:  # knife 클래스가 선택된 경우
                img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            
            if cls == 4 and knife:  # knife 클래스가 선택된 경우
                img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))

        return img
    
    except Exception as e:
        print(f"Error during detect_and_stylize: {e}")
