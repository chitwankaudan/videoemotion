from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import imutils
import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

haar_cascade = '../detector/haarcascade_frontalface_default.xml'
final_model = '../models/mini_xception.hdf5'

face_detection = cv2.CascadeClassifier(haar_cascade)
emotion_classifier = load_model(final_model, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]


def get_camera():
    cv2.namedWindow('split4ways')
    camera = cv2.VideoCapture(0)
    return camera


def stop_feed(camera):
    camera.release()
    cv2.destroyAllWindows()


def detect_emotion_video():
    camera = get_camera()
    predictions = []
    label = ''
    fX, fY, fH, fW = 0, 0, 0, 0
    while True:
        frame = camera.read()[1]
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)

        canvas = np.zeros((250, 300, 3), dtype="uint8")
        frameClone = frame.copy()

        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
                           key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces

            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            predictions = emotion_classifier.predict(roi)[0]
            label = EMOTIONS[predictions.argmax()]

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, predictions)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
                          (w, (i * 35) + 35), (0, 155, 255), -1)

            cv2.putText(canvas, text, (10, (i * 35) + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 2)

            cv2.putText(frameClone, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 155, 255), 2)

            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                          (20, 155, 255), 2)

        cv2.imshow('split4ways', frameClone)
        cv2.imshow("Probabilities", canvas)
        if cv2.waitKey(1):
            break
        stop_feed(camera)


if __name__ == '__main__':
    detect_emotion_video()
