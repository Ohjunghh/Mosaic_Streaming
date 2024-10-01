from architecture import * 
import os 
import cv2
import mtcnn
import pickle 
import numpy as np 
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model

######pathsandvairables#########
face_data = 'Faces/'
required_shape = (160,160)
face_encoder = InceptionResNetV2()
path = "facenet_keras_weights.h5"
face_encoder.load_weights(path)
face_detector = mtcnn.MTCNN()
encodes = []
encoding_dict = dict()
l2_normalizer = Normalizer('l2')
###############################


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std


for face_names in os.listdir(face_data):
    person_dir = os.path.join(face_data,face_names)

    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir,image_name)

        img_BGR = cv2.imread(image_path)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

        x = face_detector.detect_faces(img_RGB) # MTCNN을 사용하여 이미지에서 얼굴을 감지하고, 각 얼굴에 대한 바운딩 박스 정보를 얻습니다. 그 다음, 해당 바운딩 박스를 사용하여 원본 이미지에서 얼굴 영역을 자릅니다. 이 영역은 face 변수에 저장됩니다.
        x1, y1, width, height = x[0]['box'] #얼굴 감지기가 여러 얼굴을 찾을 수 있지만, 여기서는 단순화를 위해 첫 번째로 감지된 얼굴만 사용되었습니다. 따라서 x[0]['box']에서 첫 번째 얼굴의 바운딩 박스 정보를 가져옵니다.
        x1, y1 = abs(x1) , abs(y1)
        x2, y2 = x1+width , y1+height
        face = img_RGB[y1:y2 , x1:x2] #따라서 face 변수에는 이미지에서 잘려진 얼굴 영역이 포함되어 있습니다.

        
        
        face = normalize(face)
        face = cv2.resize(face, required_shape)
        face_d = np.expand_dims(face, axis=0)
        encode = face_encoder.predict(face_d)[0]
        encodes.append(encode)

    if encodes:
        encode = np.sum(encodes, axis=0 )
        encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
        encoding_dict[face_names] = encode
    
path = 'encodings/encodings.pkl'
with open(path, 'wb') as file:
    pickle.dump(encoding_dict, file)






