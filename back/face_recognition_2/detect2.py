import cv2
import numpy as np
import mtcnn
from architecture import *
from train_v2_1 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle
import time
from sort import Sort  # Sort 클래스 임포트

confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def calculate_cosine_similarity(embedding1, embedding2):
    return cosine(embedding1, embedding2)

def apply_mosaic(image, pt_1, pt_2, kernel_size=15):
    x1, y1 = pt_1
    x2, y2 = pt_2
    face_height, face_width, _ = image[y1:y2, x1:x2].shape
    
    face = image[y1:y1+face_height, x1:x1+face_width]
    
    face = cv2.resize(face, (kernel_size, kernel_size), interpolation=cv2.INTER_LINEAR)
    face = cv2.resize(face, (face_width, face_height), interpolation=cv2.INTER_NEAREST)
    
    image[y1:y1+face_height, x1:x1+face_width] = face
    
    return image

def detect(img, detector, encoder, encoding_dict):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    faces = []
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        
        # Calculate cosine similarity with known encodings
        best_match = None
        min_distance = float('inf')
        for name, known_encode in encoding_dict.items():
            distance = calculate_cosine_similarity(encode, known_encode)
            if distance < min_distance:
                min_distance = distance
                best_match = name
        
        # If the best match meets recognition threshold, add to faces
        if min_distance <= recognition_t:
            faces.append((face, pt_1, pt_2, encode, best_match))
    
    return faces

if __name__ == "__main__":
    required_shape = (160,160)
    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings/encodings.pkl'
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(encodings_path)

    cap = cv2.VideoCapture(0)
    tracker = Sort(max_age=30, min_hits=5, iou_threshold=0.2)  # Sort 객체 생성

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            print("CAM NOT OPENED")
            break

        faces = detect(frame, face_detector, face_encoder, encoding_dict)
        detections = []
        for i, (face, pt_1, pt_2, encode, name) in enumerate(faces):
            detections.append([pt_1[0], pt_1[1], pt_2[0], pt_2[1]])

        if len(detections) > 0:
            detections = np.array(detections)
            trackers = tracker.update(detections)  # Sort로 추적 업데이트

            for d in trackers:
                d = d.astype(np.int32)
                pt_1 = (d[0], d[1])
                pt_2 = (d[2], d[3])
                cv2.rectangle(frame, pt_1, pt_2, (0, 255, 0), 2)
                cv2.putText(frame, "Face", (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)

        cv2.imshow('camera', frame)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
