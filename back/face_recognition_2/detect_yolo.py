import cv2
import numpy as np
import torch
from torchvision import transforms as T
from architecture import InceptionResNetV2
from train_v2_1 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
import pickle
import time

confidence_t = 0.5  # YOLOv5 detection confidence threshold
recognition_t = 0.4  # Face recognition distance threshold
required_size = (160, 160)

# Function to load saved face encodings
def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

# Function to apply mosaic effect
def apply_mosaic(image, pt_1, pt_2, kernel_size=15):
    x1, y1 = pt_1
    x2, y2 = pt_2
    face_height, face_width, _ = image[y1:y2, x1:x2].shape
    face = image[y1:y1 + face_height, x1:x1 + face_width]
    face = cv2.resize(face, (kernel_size, kernel_size), interpolation=cv2.INTER_LINEAR)
    face = cv2.resize(face, (face_width, face_height), interpolation=cv2.INTER_NEAREST)
    image[y1:y1 + face_height, x1:x1 + face_width] = face
    return image

# Function to get face image and coordinates from bounding box
def get_face(img, box):
    x1, y1, x2, y2 = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

# Function to get face encoding using a face recognition model
def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

# Function to detect faces and apply recognition with YOLOv5
def detect(img, model, face_encoder, encoding_dict, person_to_exclude=None):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img_rgb)  # Perform YOLOv5 detection

    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        if conf < confidence_t:
            continue

        # Extract object region
        object_img = img[int(y1):int(y2), int(x1):int(x2)]

        # Check if the detected object is a face (class 0 assumed to be face)
        if cls == 4:
            encode_face = get_encode(face_encoder, object_img, required_size)
            encode_face = l2_normalizer.transform(encode_face.reshape(1, -1))[0]

            # Face recognition
            name = 'unknown'
            distance = float("inf")
            for db_name, db_encode in encoding_dict.items():
                dist = cosine(db_encode, encode_face)
                print(dist)
                if dist < recognition_t and dist < distance:
                    name = db_name
                    distance = dist

            # Display results
            if name == 'unknown':
                img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(img, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                if person_to_exclude is not None and name == person_to_exclude:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)  # Yellow rectangle for excluded person
                    cv2.putText(img, name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                else:
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f'{name} {distance:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Apply mosaic to non-face objects and display labels
            img = apply_mosaic(img, (int(x1), int(y1)), (int(x2), int(y2)))
            label = f'Class {cls} {conf:.2f}'
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return img

if __name__ == "__main__":
    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./627.pt', force_reload=True)
    print("YOLOv5 model loaded successfully.")
    
    # YOLOv5 설정
    model.conf = 0.5  # Detection confidence threshold
    model.classes = None  # 모든 클래스 사용
    model.agnostic_nms = False  # 클래스 독립적인 NMS 설정
    
    required_shape = (160,160)
    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = './encodings/encodings.pkl'
    face_detector = model
    encoding_dict = load_pickle(encodings_path)
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    # Define the codec and create VideoWriter object for recording
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image from camera.")
            break

        # Perform face detection and recognition
        frame = detect(frame, face_detector, face_encoder, encoding_dict)

        # Write the frame to the file for recording
        out.write(frame)
        
        cv2.imshow('camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
