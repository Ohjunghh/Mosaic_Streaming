from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_GET
import cv2
import numpy as np
import torch
import os
import signal
from time import time, sleep
from multiprocessing import Process, Manager

# 가능 리스트 설정
CONN_LIMIT = 10
manager = Manager()
possible_list = manager.list([True] * CONN_LIMIT)

# 현재 작동 중인 프로세스 리스트 조회
@require_GET
def get_process_list(request):
    process_data_list = []
    for i in range(CONN_LIMIT):
        if not possible_list[i]:
            process_data = {
                'result': 'true',
                'id': i
            }
            process_data_list.append(process_data)

    json_data = {
        'list': process_data_list
    }
    return JsonResponse(json_data)

# 촬영 시작 -> 지정 번호 get
@require_GET
def get_id(request):
    id = find_possible_id()
    if id != -1:
        json_data = {
            'result': 'true',
            'id': id
        }
    else:
        json_data = {
            'result': 'false',
            'message': 'connection limit...'
        }
    return JsonResponse(json_data)

def find_possible_id():
    for i in range(CONN_LIMIT):
        if possible_list[i]:
            possible_list[i] = False
            return i
    return -1

# 촬영 종료 -> 지정 번호 release
@require_GET
def release_number(request, id):
    possible_list[int(id)] = True
    print('release id : ', id)
    return JsonResponse({'result': 'true'})

# 모자이크 시작
@require_GET
def mosaic(request, id):
    process = Process(target=work, args=(id,))
    process.start()
    return JsonResponse({'result': 'true'})

def work(id):
    mosaic_object = MosaicObject()
    mosaic_object.__init__()

    base_url = "rtmp://15.164.170.6/"
    cap = cv2.VideoCapture(base_url + f"live/{id}")
    rtmp_out_url = base_url + f"live-out/{id}"

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    process = (
        ffmpeg
        .input('pipe:', r='6')
        .output(rtmp_out_url, 
            vcodec='libx264', 
            pix_fmt='yuv420p', 
            preset='veryfast', 
            r='20', g='50', 
            video_bitrate='1.4M', 
            maxrate='2M', 
            bufsize='2M', 
            segment_time='6',
            format='flv')
        .run_async(pipe_stdin=True)
    )

    sleep(1)

    while cap.isOpened():
        start_time = time()

        status, frame = cap.read()
        if not status:
            print('can not read!')
            break

        results = mosaic_object.score_frame(frame)
        frame = mosaic_object.mosaic_frame(results, frame)

        end_time = time()
        fps = 1 / np.round(end_time - start_time, 3)
        print(f"FPS = {fps}")

        status2, buf = cv2.imencode('.png', frame)
        process.stdin.write(frame.tobytes())

    cap.release()
    print(f'cap release! id={id}')
    cv2.destroyAllWindows()

class MosaicObject:

    def __init__(self):
        self.model = self.load_model()
        self.classes = self.model.names

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        return model

    def score_frame(self, frame):
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
        return labels, cord

    def class_to_label(self, x):
        return self.classes[int(x)]

    def mosaic_frame(self, results, frame):
        labels, cord = results
        n = len(labels)

        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.6:
                left = int(row[0] * x_shape)
                top = int(row[1] * y_shape)
                right = int(row[2] * x_shape)
                bottom = int(row[3] * y_shape)

                mosaic_part = frame[top:bottom, left:right]
                mosaic_part = cv2.blur(mosaic_part, (50, 50))

                frame[top:bottom, left:right] = mosaic_part

        return frame
