import torch
import logging

yolo_model = None
logger = logging.getLogger(__name__)

def load_global_model():
    """ YOLO 모델을 전역적으로 로드하는 함수 """
    global yolo_model
    if yolo_model is None:
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f'Using device: {device}')
            yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='./face_recognition_2/926.pt', force_reload=True)
            yolo_model.to(device)
            yolo_model.conf = 0.5  # 신뢰도 설정
            yolo_model.classes = None
            logger.info('YOLO model loaded successfully.')
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            yolo_model = None
            raise
    return yolo_model
