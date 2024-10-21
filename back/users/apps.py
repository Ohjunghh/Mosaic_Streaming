from django.apps import AppConfig
import torch

class UsersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'users'

    def ready(self):
        from yolo_loader import load_global_model
        # 서버가 시작될 때 YOLO 모델을 한 번만 로드
        load_global_model()