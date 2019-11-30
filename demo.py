#from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input
import numpy as np

class livedemo:
    def __init__(self, faceDetectionPath, motionClassifierPath, featureExtractor=''):

        # Load face detection    
        self.faceDetector = cv2.CascadeClassifier(faceDetectionPath)

        # Load emotion classifier and emotion classes
        self.emotionClassifier = load_model(motionClassifierPath, compile=False)
        self.emotionsLookup = {0 : "angry", 1 : "disgust", 2 : "fear", 3 : "happy", \
                                4 : "sad", 5 : "surprise", 6 : "neutral"}

        # If applicable, load feature extractor
        if featureExtractor.lower() == 'vgg':
            self.featureExtractor = vgg16.VGG16(weights='imagenet', include_top=False)
            # Pop the last 3 fully connected and softmax layers in vgg-16
            for i in range(33, 41):
                self.featureExtractor.layers.pop()
        else:
            self.featureExtractor = None
    
    def launch(self):

        # Start video capture
        capture = cv2.VideoCapture(0)
        while(True):

            # Capture frame by frame and convert to grayscale
            ret, frame = capture.read()
            frame = imutils.resize(frame, width=500) # resize frame to make processing for face detector
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect face and crop image to just face (input to emotion classifier)
            faces = self.faceDetector.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
            if len(faces) > 0:  # If face is detected...
                for (x, y, w, h) in faces:
                    # Display rectangle around face
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                    # Extract face and resize to optimal input for classifier
                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face, (48, 48))
                    cv2.imwrite('demo.jpg', face)

                # Run emotion classifier
                predict = self.predict(face) #how to viz? need to figure that out
                print(predict)


            # Display the resulting frames
            cv2.imshow('Face', frame)
            #cv2.imshow()

            # Exit video
            if cv2.waitKey(1) & 0xFF == ord('c'):   # break if user types c
                break

        # Release video capture
        capture.release()
        cv2.destroyAllWindows()


    def predict(self, face):

        # If applicable, extract features 
        if self.featureExtractor is not None:
            inputs = self.extractFeatures(face)
        else:
            inputs = face

        # Run prediction
        return self.emotionClassifier.predict(inputs)[0]


    def extractFeatures(self, face):
        #print("Nonwritten shape (Before reshaping)", face.shape)
        # Convert face image to tensor
        face = cv2.imread("demo.jpg")   # must be written and read to use current model
        #print("Written shape (Before reshaping)", face.shape)
        faceTensor = np.expand_dims(face, axis=0)
        #print("Face shape when reading in (Before prepocessing)", faceTensor.shape)
        faceTensor = preprocess_input(faceTensor)

        # Run vgg
        features = self.featureExtractor.predict(faceTensor)
        features = np.expand_dims(features, axis=0)
        return features


def main():
    faceDetectionPath = 'detector/haarcascade_frontalface_default.xml' # may switch out for MTCNN later
    emotionClassifierPath =  './models/model.hdf5'
    demo = livedemo(faceDetectionPath, emotionClassifierPath, 'vgg')
    demo.launch()


if __name__ == '__main__':
    main()