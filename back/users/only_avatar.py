import os
import argparse
import numpy as np
import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from model.vtoonify import VToonify
from util import save_image, tensor2cv2, load_psp_standalone
from PIL import Image
import dlib
from model.bisenet.model import BiSeNet


class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Style Transfer")
        self.parser.add_argument("--content", type=str, default='./data/077436.jpg', help="path of the content image/video")
        self.parser.add_argument("--style_id", type=int, default=26, help="the id of the style image")
        self.parser.add_argument("--style_degree", type=float, default=0.5, help="style degree for VToonify-D")
        self.parser.add_argument("--color_transfer", action="store_true", help="transfer the color of the style")
        self.parser.add_argument("--ckpt", type=str, default='./checkpoint/vtoonify_d_cartoon/vtoonify_s_d_c.pt', help="path of the saved model")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--style_encoder_path", type=str, default='./checkpoint/encoder.pt', help="path of the style encoder")
        self.parser.add_argument("--exstyle_path", type=str, default=None, help="path of the extrinsic style code")
        self.parser.add_argument("--faceparsing_path", type=str, default='./checkpoint/faceparsing.pth', help="path of the face parsing model")
        self.parser.add_argument("--video", action="store_true", help="if true, video stylization; if false, image stylization")
        self.parser.add_argument("--cpu", action="store_true", help="if true, only use cpu")
        self.parser.add_argument("--backbone", type=str, default='dualstylegan', help="dualstylegan | toonify")
        self.parser.add_argument("--batch_size", type=int, default=4, help="batch size of frames when processing video")
        self.parser.add_argument("--yolo_model_path", type=str, default='../face_recognition_2/627.pt', help="path to the YOLO model")
       
    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('Load options')
        for name, value in sorted(args.items()):
            print(f'{name}: {value}')
        return self.opt


def detect_faces_yolo(model, img, confidence_t=0.5, face_class=4, upscale_factor=2):
    """YOLO로 얼굴 탐지 (저해상도를 고려해 입력 영상 확대)"""
    # 입력 이미지를 확대하여 탐지 정확도를 높임
    img_upscaled = cv2.resize(img, (img.shape[1] * upscale_factor, img.shape[0] * upscale_factor))

    # YOLO 모델에 확대된 이미지 입력
    img_rgb = cv2.cvtColor(img_upscaled, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)

    faces = []
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        if conf >= confidence_t and cls == face_class:
            # 좌표를 다시 원래 이미지 크기에 맞게 조정
            faces.append((
                int(x1 / upscale_factor),
                int(y1 / upscale_factor),
                int(x2 / upscale_factor),
                int(y2 / upscale_factor)
            ))
    return faces


def detect_landmarks_dlib(image, predictor, x1, y1, x2, y2, upscale_factor=2):
    """YOLO로 탐지한 얼굴 영역에 대해 Dlib으로 랜드마크 탐지 (입력 이미지 확대)"""
    face_roi = image[y1:y2, x1:x2]
    
    # 얼굴 영역 확대
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

    # 원본 좌표로 변환
    landmarks[:, 0] = landmarks[:, 0] / upscale_factor + x1
    landmarks[:, 1] = landmarks[:, 1] / upscale_factor + y1

    return landmarks


def align_face(image, landmarks):
    """얼굴을 랜드마크 기반으로 정렬하는 함수"""
    lm_eye_left = landmarks[36:42]  # 왼쪽 눈
    lm_eye_right = landmarks[42:48]  # 오른쪽 눈

    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (landmarks[48] + landmarks[54]) * 0.5  # 입 좌우 평균
    eye_to_mouth = mouth_avg - eye_avg

    # 얼굴의 회전 및 크기를 맞추기 위한 벡터 계산
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = Image.fromarray(image)
    img = img.transform((256, 256), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)

    return img, quad, c  # 얼굴의 중심 좌표 'c'를 반환


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


if __name__ == "__main__":
    parser = TestOptions()
    args = parser.parse()
    print('*' * 98)
   
    device = "cpu" if args.cpu else "cuda"
   
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    # YOLO 모델 로드
    confidence_t = 0.5  # YOLO 모델 신뢰도 설정
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.yolo_model_path, force_reload=True).to(device)
    yolo_model.conf = confidence_t  # 신뢰도 설정
    yolo_model.classes = None  # 모든 클래스를 탐지
    yolo_model.agnostic_nms = False  # 클래스 무관 NMS 비활성화

    # VToonify 모델 로드
    vtoonify = VToonify(backbone=args.backbone)
    vtoonify.load_state_dict(torch.load(args.ckpt, map_location=lambda storage, loc: storage)['g_ema'])
    vtoonify.to(device)

    # 스타일 인코더 로드
    pspencoder = load_psp_standalone(args.style_encoder_path, device)

    # 스타일 코드 로드
    if args.backbone == 'dualstylegan':
        exstyles = np.load(args.exstyle_path, allow_pickle='TRUE').item()
        stylename = list(exstyles.keys())[args.style_id]
        exstyle = torch.tensor(exstyles[stylename]).to(device)
        with torch.no_grad():
            exstyle = vtoonify.zplus2wplus(exstyle)

    print('Load models successfully!')

    filename = args.content
    basename = os.path.basename(filename).split('.')[0]
    print(f'Processing {filename} with vtoonify_{args.backbone[0]}')

    # dlib 랜드마크 모델 로드
    predictor = dlib.shape_predictor('./checkpoint/shape_predictor_68_face_landmarks.dat')

    if args.video:
        video_cap = cv2.VideoCapture(filename)
        num = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_video_path = os.path.join(args.output_path, f"{basename}_stylized.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        videoWriter = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # 얼굴 파싱 모델 로드
        parsingpredictor = BiSeNet(n_classes=19)
        parsingpredictor.load_state_dict(torch.load(args.faceparsing_path, map_location=lambda storage, loc: storage))
        parsingpredictor.to(device).eval()

        for i in tqdm(range(num)):
            success, frame = video_cap.read()
            if not success:
                break

            frame_copy = frame.copy()  # 원본 비디오 복사 (모든 얼굴 처리 전)

            # YOLO로 얼굴 탐지
            faces = detect_faces_yolo(yolo_model, frame, confidence_t)
            if not faces:
                videoWriter.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 얼굴 없으면 원본 저장
                continue

            for (x1, y1, x2, y2) in faces:
                # YOLO로 탐지된 얼굴 영역만 잘라서 dlib으로 랜드마크 추출
                landmarks = detect_landmarks_dlib(frame, predictor, x1, y1, x2, y2)
                if landmarks is None:
                    continue

                # YOLO로 탐지된 얼굴 정렬 및 쿼드 좌표 반환
                aligned_face, quad, face_center = align_face(frame, landmarks)
                face_tensor = transform(aligned_face).unsqueeze(dim=0).to(device)

                # 얼굴 파싱 정보 예측
                with torch.no_grad():
                    x_p = F.interpolate(parsingpredictor(2 * (F.interpolate(face_tensor, scale_factor=2, mode='bilinear', align_corners=False)))[0],
                                        scale_factor=0.5, recompute_scale_factor=False).detach()

                # 얼굴 이미지와 파싱 정보 결합하여 스타일 적용
                inputs = torch.cat((face_tensor, x_p / 16.), dim=1)

                with torch.no_grad():
                    s_w = pspencoder(face_tensor)
                    s_w = vtoonify.zplus2wplus(s_w)
                    if args.backbone == 'dualstylegan':
                        s_w[:, :7] = exstyle[:, :7]  # 스타일 적용
                    y_tilde = vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s=args.style_degree)
                    y_tilde = torch.clamp(y_tilde, -1, 1)

                # PIL 이미지를 OpenCV로 변환할 때 RGB -> BGR 변환
                stylized_face_np = tensor2cv2(y_tilde[0].cpu())
                stylized_face_np_bgr = cv2.cvtColor(stylized_face_np, cv2.COLOR_RGB2BGR)  # 변환 명확화

                # 얼굴 크기 조정 (스케일링 없이 고정된 크기로 조정)
                face_width = int(quad[2][0] - quad[0][0])
                face_height = int(quad[2][1] - quad[0][1])

                # 스타일화된 얼굴 크기 조정 (스케일링 없이)
                stylized_face_resized = cv2.resize(stylized_face_np_bgr, (face_width, face_height), interpolation=cv2.INTER_LINEAR)

                # 마스크를 적용해 얼굴을 원본 이미지에 부드럽게 합성
                frame_copy = apply_masked_face(frame_copy, stylized_face_resized, face_center, (face_width, face_height))

            # 변환된 프레임을 출력 비디오로 저장
            frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
            videoWriter.write(cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR))  # 결과 저장

        videoWriter.release()
        video_cap.release()

    print('Transfer style successfully!')
