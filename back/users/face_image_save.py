from django.core.files.storage import default_storage
import os
import pickle
from .train import encode_faces


def save_encoding_vector(user_id):
    user_encodings_dir = os.path.join('media/encodings', str(user_id))  # 사용자별 인코딩 디렉토리 경로
    os.makedirs(user_encodings_dir, exist_ok=True)  # 디렉토리 생성, 이미 존재하면 무시
    file_path = os.path.join(user_encodings_dir, 'encoding_vector.pkl')  # 저장할 파일 경로
    
    encoding_dict = encode_faces(user_id)

    if encoding_dict:  # 인코딩이 비어있지 않을 경우에만 저장
        with open(file_path, 'wb') as file:  # 파일을 쓰기 모드로 열기
            pickle.dump(encoding_dict, file)  # 인코딩 벡터를 파일에 저장

    return file_path  # 저장된 파일의 경로 반환
