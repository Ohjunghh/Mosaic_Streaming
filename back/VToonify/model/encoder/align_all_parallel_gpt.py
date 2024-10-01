"""
brief: face alignment with FFHQ method (https://github.com/NVlabs/ffhq-dataset)
author: lzhbrian (https://lzhbrian.me)
date: 2020.1.5
note: code is heavily borrowed from 
    https://github.com/NVlabs/ffhq-dataset
    http://dlib.net/face_landmark_detection.py.html

requirements:
    apt install cmake
    conda install Pillow numpy scipy
    pip install dlib
    # download face landmark model from:
    # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
"""
from argparse import ArgumentParser
import time
import numpy as np
import PIL
import PIL.Image
import os
import scipy
import scipy.ndimage
import dlib
import multiprocessing as mp
import math

SHAPE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'

# # 얼굴 랜드마크 모델이 존재하는지 확인
# if not os.path.exists(SHAPE_PREDICTOR_PATH):
#     raise ValueError(f"Predictor model not found at {SHAPE_PREDICTOR_PATH}")


def get_landmark(img, predictor):
    """dlib를 사용하여 모든 얼굴의 랜드마크를 가져옵니다.
    :return: 각 얼굴에 대한 np.array의 리스트, 각각 shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()  # 얼굴 탐지기 설정
    dets = detector(img, 1)  # 얼굴 탐지

    if len(dets) == 0:
        print('Error: no face detected!')
        return None

    landmarks_list = []
    for d in dets:
        print(f"d object: {d}, type: {type(d)}")  # 디버깅용 출력
        shape = predictor(img, d)  # 각 얼굴의 랜드마크를 예측 시도
        print(f"shape object: {shape}, type: {type(shape)}")  # 랜드마크 출력
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])  # 랜드마크 포인트 추출
        landmarks_list.append(landmarks)  # 각 얼굴의 랜드마크를 리스트에 추가

    return landmarks_list  # 다중 얼굴 랜드마크 리스트 반환

# def align_facess(img, predictor):
#     """
#     :param img: 이미지 배열
#     :return: 정렬된 각 얼굴에 대한 PIL 이미지 리스트
#     """
#     landmarks_list = get_landmark(img, predictor)
#     if landmarks_list is None:
#         return None

#     aligned_faces = []  # 다중 얼굴을 저장할 리스트

#     for lm in landmarks_list:  # 각 얼굴의 랜드마크에 대해 반복
#         # 랜드마크 좌표로부터 얼굴 정렬 처리
#         lm_chin = lm[0: 17]  # left-right
#         lm_eyebrow_left = lm[17: 22]  # left-right
#         lm_eyebrow_right = lm[22: 27]  # left-right
#         lm_nose = lm[27: 31]  # top-down
#         lm_nostrils = lm[31: 36]  # top-down
#         lm_eye_left = lm[36: 42]  # left-clockwise
#         lm_eye_right = lm[42: 48]  # left-clockwise
#         lm_mouth_outer = lm[48: 60]  # left-clockwise
#         lm_mouth_inner = lm[60: 68]  # left-clockwise

#         # 눈과 입 좌표를 기준으로 얼굴 정렬
#         eye_left = np.mean(lm_eye_left, axis=0)
#         eye_right = np.mean(lm_eye_right, axis=0)
#         eye_avg = (eye_left + eye_right) * 0.5
#         eye_to_eye = eye_right - eye_left
#         mouth_left = lm_mouth_outer[0]
#         mouth_right = lm_mouth_outer[6]
#         mouth_avg = (mouth_left + mouth_right) * 0.5
#         eye_to_mouth = mouth_avg - eye_avg

#         # 방향성 있는 크롭 사각형 계산
#         x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
#         x /= np.hypot(*x)
#         x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
#         y = np.flipud(x) * [-1, 1]
#         c = eye_avg + eye_to_mouth * 0.1
#         quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
#         qsize = np.hypot(*x) * 2

#         output_size = 256
#         transform_size = 256
#         enable_padding = True

#         # 이미지 크기 조정
#         shrink = int(np.floor(qsize / output_size * 0.5))
#         if shrink > 1:
#             rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
#             img = img.resize(rsize, PIL.Image.ANTIALIAS)
#             quad /= shrink
#             qsize /= shrink

#         # 크롭
#         border = max(int(np.rint(qsize * 0.1)), 3)
#         crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
#                 int(np.ceil(max(quad[:, 1]))))
#         crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
#                 min(crop[3] + border, img.size[1]))
#         if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
#             img = img.crop(crop)
#             quad -= crop[0:2]

#         # 패딩
#         pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
#                int(np.ceil(max(quad[:, 1]))))
#         pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
#                max(pad[3] - img.size[1] + border, 0))
#         if enable_padding and max(pad) > border - 4:
#             pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
#             img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
#             h, w, _ = img.shape
#             y, x, _ = np.ogrid[:h, :w, :1]
#             mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
#                               1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
#             blur = qsize * 0.02
#             img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
#             img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
#             img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
#             quad += pad[:2]

#         # 변환
#         img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
#         if output_size < transform_size:
#             img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

#         aligned_faces.append(img)  # 정렬된 얼굴을 리스트에 추가

#     return aligned_faces  # 모든 얼굴 이미지 반환
def align_facess(img, landmarks):
    """
    :param img: 이미지 배열
    :param landmarks: 얼굴의 랜드마크 좌표 배열
    :return: 정렬된 얼굴 이미지
    """
    aligned_faces = []  # 다중 얼굴을 저장할 리스트

    # 랜드마크 좌표로부터 얼굴 정렬 처리
    lm_chin = landmarks[0: 17]  # left-right
    lm_eyebrow_left = landmarks[17: 22]  # left-right
    lm_eyebrow_right = landmarks[22: 27]  # left-right
    lm_nose = landmarks[27: 31]  # top-down
    lm_nostrils = landmarks[31: 36]  # top-down
    lm_eye_left = landmarks[36: 42]  # left-clockwise
    lm_eye_right = landmarks[42: 48]  # left-clockwise
    lm_mouth_outer = landmarks[48: 60]  # left-clockwise
    lm_mouth_inner = landmarks[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    if isinstance(img, str):
        img = PIL.Image.open(img)
    else:
        img = PIL.Image.fromarray(img)

    output_size = 256
    transform_size = 256
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    aligned_faces.append(img)  # 정렬된 얼굴을 리스트에 추가

    return aligned_faces  # 모든 얼굴 이미지 반환


def chunks(lst, n):
    """리스트를 n개의 청크로 나누기"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def extract_on_paths(file_paths):
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    pid = mp.current_process().name
    print(f'\t{pid} is starting to extract on #{len(file_paths)} images')

    for file_path, res_path in file_paths:
        try:
            img = dlib.load_rgb_image(file_path)
            res = align_facess(img, predictor)
            res = [face.convert('RGB') for face in res]
            os.makedirs(os.path.dirname(res_path), exist_ok=True)
            for i, face in enumerate(res):
                face.save(f"{res_path}_{i}.jpg")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    print(f'\t{pid} Done!')


def parse_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--root_path', type=str, default='')
    args = parser.parse_args()
    return args


def run(args):
    root_path = args.root_path
    out_crops_path = root_path + '_crops'
    if not os.path.exists(out_crops_path):
        os.makedirs(out_crops_path, exist_ok=True)

    file_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_path = os.path.join(root, file)
            fname = os.path.join(out_crops_path, os.path.relpath(file_path, root_path))
            res_path = '{}.jpg'.format(os.path.splitext(fname)[0])
            if os.path.splitext(file_path)[1] == '.txt' or os.path.exists(res_path):
                continue
            file_paths.append((file_path, res_path))

    file_chunks = list(chunks(file_paths, int(math.ceil(len(file_paths) / args.num_threads))))
    print(f'Running on {len(file_paths)} paths')
    tic = time.time()
    pool = mp.Pool(args.num_threads)
    pool.map(extract_on_paths, file_chunks)
    toc = time.time()
    print(f'Mischief managed in {toc - tic}s')


if __name__ == '__main__':
    args = parse_args()
    run(args)
