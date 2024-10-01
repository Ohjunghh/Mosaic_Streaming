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


def detect_faces_yolo(model, img, confidence_t=0.5, face_class=4):
    """YOLO로 얼굴 탐지"""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)


    faces = []
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        if conf >= confidence_t and cls == face_class:
            faces.append((int(x1), int(y1), int(x2), int(y2)))  # 얼굴 영역 좌표 추가
    return faces


def detect_landmarks_dlib(image, predictor, x1, y1, x2, y2):
    """YOLO로 탐지한 얼굴 영역에 대해 dlib으로 랜드마크 탐지"""
    face_roi = image[y1:y2, x1:x2]  # YOLO로 탐지된 얼굴 영역만 잘라냄
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
   
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1)
   
    if len(rects) == 0:
        print("No landmarks detected in the face region")
        return None


    for rect in rects:
        shape = predictor(gray, rect)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])


    # 원본 이미지 좌표로 변환
    landmarks[:, 0] += x1
    landmarks[:, 1] += y1


    return landmarks


def align_face(image, landmarks):
    """얼굴을 정렬하는 함수, 랜드마크 기반"""
    lm_eye_left = landmarks[36:42]  # 왼쪽 눈
    lm_eye_right = landmarks[42:48]  # 오른쪽 눈


    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (landmarks[48] + landmarks[54]) * 0.5  # 입 좌우 평균
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
   
    return img


if __name__ == "__main__":
    parser = TestOptions()
    args = parser.parse()
    print('*'*98)
   
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
                # YOLO로 잡은 얼굴 영역만 잘라서 dlib으로 랜드마크 추출
                landmarks = detect_landmarks_dlib(frame, predictor, x1, y1, x2, y2)
                if landmarks is None:
                    continue


                # YOLO로 탐지된 얼굴 정렬
                aligned_face = align_face(frame, landmarks)
                face_tensor = transform(aligned_face).unsqueeze(dim=0).to(device)


                # 얼굴 파싱 정보 예측 (19채널)
                with torch.no_grad():
                    x_p = F.interpolate(parsingpredictor(2 * (F.interpolate(face_tensor, scale_factor=2, mode='bilinear', align_corners=False)))[0],
                                        scale_factor=0.5, recompute_scale_factor=False).detach()


                # 얼굴 이미지 (3채널)와 얼굴 파싱 정보 (19채널) 결합하여 22채널 생성
                inputs = torch.cat((face_tensor, x_p / 16.), dim=1)


                # VToonify 스타일 적용
                with torch.no_grad():
                    s_w = pspencoder(face_tensor)
                    s_w = vtoonify.zplus2wplus(s_w)
                    if args.backbone == 'dualstylegan':
                        s_w[:, :7] = exstyle[:, :7]  # 스타일 적용
                       
                    # VToonify 모델에 22채널을 입력하여 스타일 적용
                    y_tilde = vtoonify(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s=args.style_degree)
                    y_tilde = torch.clamp(y_tilde, -1, 1)


                # 원본 프레임에 스타일 적용된 얼굴 결합
                stylized_face_np = tensor2cv2(y_tilde[0].cpu())


                # 얼굴 영역만 변환된 스타일 적용 (색 공간 변환 후 일치시킴)
                stylized_face_np_bgr = cv2.cvtColor(stylized_face_np, cv2.COLOR_RGB2BGR)
                frame_copy[y1:y2, x1:x2] = cv2.resize(stylized_face_np_bgr, (x2 - x1, y2 - y1))


            # 원본 프레임의 색상 변환을 통일 (전체 프레임 색상 변환)
            frame_bgr = cv2.cvtColor(frame_copy, cv2.COLOR_RGB2BGR)
            videoWriter.write(cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR))  # 결과 저장


        videoWriter.release()
        video_cap.release()


    print('Transfer style successfully!')
