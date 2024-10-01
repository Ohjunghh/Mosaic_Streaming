from django.core.files.storage import default_storage
import os
import pickle
from .train import encode_faces

# def save_face_images(image_files, user_id):
#     user_faces_dir = os.path.join('media/faces', str(user_id))  # 사용자별 디렉토리 경로
#     os.makedirs(user_faces_dir, exist_ok=True)  # 디렉토리 생성, 이미 존재하면 무시

#     file_paths = []
#     for idx, image_file in enumerate(image_files, start=1):  # enumerate를 사용해 1부터 시작하는 인덱스 생성
#         file_name = f'{idx}.jpg'  # 파일명을 idx.jpg로 설정
#         file_path = os.path.join(user_faces_dir, file_name)  # 저장할 파일 경로

#         with default_storage.open(file_path, 'wb+') as destination:  # 파일을 쓰기 모드로 열기
#             for chunk in image_file.chunks():  # 파일의 청크를 반복하면서
#                 destination.write(chunk)  # 청크를 파일에 씀
#         file_paths.append(file_path)  # 각 파일의 경로를 리스트에 저장

#     return file_paths  # 저장된 모든 파일의 경로 리스트를 반환

def save_encoding_vector(user_id):
    user_encodings_dir = os.path.join('media/encodings', str(user_id))  # 사용자별 인코딩 디렉토리 경로
    os.makedirs(user_encodings_dir, exist_ok=True)  # 디렉토리 생성, 이미 존재하면 무시
    file_path = os.path.join(user_encodings_dir, 'encoding_vector.pkl')  # 저장할 파일 경로
    
    encoding_dict = encode_faces(user_id)

    if encoding_dict:  # 인코딩이 비어있지 않을 경우에만 저장
        with open(file_path, 'wb') as file:  # 파일을 쓰기 모드로 열기
            pickle.dump(encoding_dict, file)  # 인코딩 벡터를 파일에 저장

    return file_path  # 저장된 파일의 경로 반환
