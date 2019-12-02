from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import imutils
import cv2
import numpy as np
import sys
import os

# to use the system GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

haar_cascade = '../detector/haarcascade_frontalface_default.xml'
final_model = '../models/mini_xception.hdf5'
image_location = "/Users/raghav/Development/VideoEmotion/images/test1.jpeg"

face_detection = cv2.CascadeClassifier(haar_cascade)
emotion_classifier = load_model(final_model, compile=False)

orig_frame = cv2.imread(image_location)
frame = cv2.imread(image_location, 0)
faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
if len(faces) > 0:
    faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces
    roi = frame[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    predictions = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(predictions)
    label = EMOTIONS[predictions.argmax()]
    cv2.putText(orig_frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.45, (0, 155, 255), 2)
    cv2.rectangle(orig_frame, (fX, fY), (fX + fW, fY + fH), (20, 155, 255), 2)
else:
    print("LOL")

cv2.imshow('detect', orig_frame)
cv2.imwrite('result/' + image_location.split('/')[-1], orig_frame)
if cv2.waitKey(2000):
    print("Emotions Detected")

cv2.destroyAllWindows()
