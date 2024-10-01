import cv2
import numpy as np
import mtcnn
from architecture import *
from train_v2_1 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle
import time

confidence_t = 0.99
recognition_t = 0.5
required_size = (160, 160)
mean_shift_interval = 5  # 매 5프레임마다 Mean Shift를 적용

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

def apply_mosaic(image, pt_1, pt_2, kernel_size=15):
    x1, y1 = pt_1
    x2, y2 = pt_2
    face_height, face_width, _ = image[y1:y2, x1:x2].shape
    
    face = image[y1:y1+face_height, x1:x1+face_width]
    
    face = cv2.resize(face, (kernel_size, kernel_size), interpolation=cv2.INTER_LINEAR)
    face = cv2.resize(face, (face_width, face_height), interpolation=cv2.INTER_NEAREST)
    
    image[y1:y1+face_height, x1:x1+face_width] = face
    
    return image

def mean_shift_v2(img, hist, bbox):
    # Convert inputs to proper types
    img = np.asarray(img, dtype=np.uint8)
    hist = np.asarray(hist, dtype=np.float32)
    x, y, w, h = bbox
    
    # Define histogram parameters
    channels = [0]
    hsize = [64]
    range1 = [0, 180]
    histRange = [range1, [0, 256], [0, 256]]  # Hue, Saturation, Value
    
    # Create region of interest (ROI) using given bbox
    roi = (x, y, w, h)
    img_ROI = img[y:y+h, x:x+w]
    
    # Convert image to HSV color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_ROI_hsv = cv2.cvtColor(img_ROI, cv2.COLOR_BGR2HSV)
    
    # Create mask for histogram calculation
    img_mask = cv2.inRange(img_ROI_hsv, (0., 60., 60.), (180., 255., 255.))
    
    # Calculate histogram of ROI
    objectHistogram = cv2.calcHist([img_ROI_hsv], channels, img_mask, hsize, histRange)
    objectHistogram = cv2.normalize(objectHistogram, objectHistogram, 0, 255, cv2.NORM_MINMAX)
    
    # Back project the histogram
    bp = cv2.calcBackProject([img_hsv], channels, objectHistogram, histRange, 1)
    
    # Mean shift tracking
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    _, roi = cv2.meanShift(bp, roi, term_crit)
    
    # Extract coordinates from ROI
    x, y, w, h = roi
    
    # Create and return updated bbox
    updated_bbox = [x, y, w, h]
    return updated_bbox


def mean_shift_tracking(img, hist, bbox):
    img_height, img_width, _ = img.shape
    x, y, w, h = bbox
    
    # Call mean_shift_v2 to get updated bbox
    updated_bbox = mean_shift_v2(img, hist, bbox)
    
    # Extract updated coordinates from updated bbox
    x_new, y_new, w_new, h_new = updated_bbox
    
    # Convert float to int for pixel coordinates
    x_new, y_new, w_new, h_new = int(x_new), int(y_new), int(w_new), int(h_new)
    
    # Update bbox with new coordinates
    bbox = [x_new, y_new, w_new, h_new]
    
    return bbox

def detect_and_track(img, detector, encoder, encoding_dict, frame_count):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    
    for res in results:
        if res['confidence'] < confidence_t:
            continue
        
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, required_size)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        
        name = 'unknown'
        distance = float("inf")
        
        for db_name, db_encode in encoding_dict.items():
            dist = cosine(db_encode, encode)
            if dist < recognition_t and dist < distance:
                name = db_name
                distance = dist
        
        if name == 'unknown':
            img = apply_mosaic(img, pt_1, pt_2)
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name, (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)
        
        # Apply mean shift tracking every mean_shift_interval frames
        if frame_count % mean_shift_interval == 0:
            bbox = [pt_1[0], pt_1[1], pt_2[0] - pt_1[0], pt_2[1] - pt_1[1]]
            roi = mean_shift_tracking(img, img_rgb, bbox)
            
            # Update pt_1, pt_2 based on mean shift tracking result
            pt_1 = (roi[0], roi[1])
            pt_2 = (roi[0] + roi[2], roi[1] + roi[3])
            
            cv2.rectangle(img, pt_1, pt_2, (255, 0, 0), 2)  # Display mean shift tracking result
    
    return img

if __name__ == "__main__":
    required_size = (160, 160)
    face_encoder = InceptionResNetV2()
    path_m = "facenet_keras_weights.h5"
    face_encoder.load_weights(path_m)
    encodings_path = 'encodings/encodings.pkl'
    face_detector = mtcnn.MTCNN()
    encoding_dict = load_pickle(encodings_path)
    
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output4.avi', fourcc, 20.0, (640, 480))
    
    frame_count = 0
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            print("CAM NOT OPENED")
            break
        
        frame_count += 1

        # Detect faces and perform recognition, and then track using mean shift
        frame = detect_and_track(frame, face_detector, face_encoder, encoding_dict, frame_count)
        
        out.write(frame)
        cv2.imshow('camera', frame)
        
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f"FPS: {fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Traceback (most recent call last):
#   File "detect3.py", line 165, in <module>
#     frame = detect(frame, face_detector, face_encoder, encoding_dict)
#   File "detect3.py", line 134, in detect
#     roi = mean_shift_tracking(img_rgb, img_rgb, bbox)  # Use img_rgb for both img and hist
#   File "detect3.py", line 93, in mean_shift_tracking
#     updated_bbox = mean_shift_v2(img, hist, bbox)
#   File "detect3.py", line 71, in mean_shift_v2
#     objectHistogram = cv2.calcHist([img_ROI_hsv], channels, img_mask, hsize, histRange)
# cv2.error: OpenCV(4.9.0) :-1: error: (-5:Bad argument) in function 'calcHist'
# > Overload resolution failed:
# >  - Can't parse 'ranges'. Sequence item with index 0 has a wrong type
# >  - Can't parse 'ranges'. Sequence item with index 0 has a wrong type
