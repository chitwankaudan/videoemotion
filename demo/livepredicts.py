#from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
from keras.applications import vgg16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt




class livepredicts:
    """
    Using live webcam input to run the emotion classifier and build the visualization.
    Written so it can run with all of our classification models
    """
    def __init__(self, faceDetectionPath, motionClassifierPath, featureExtractor=''):
        # Load face detector    
        self.faceDetector = cv2.CascadeClassifier(faceDetectionPath)

        # Load emotion classifier and emotion classes
        self.emotionClassifier = load_model(motionClassifierPath, compile=False)
        self.emotionsLookup = {0 : "angry", 1 : "disgust", 2 : "fear", 3 : "happy", \
                                4 : "sad", 5 : "surprise", 6 : "neutral"}

        # If applicable, load feature extractor
        if featureExtractor.lower() == 'vgg':
            self.featureExtractor = vgg16.VGG16(weights='imagenet', include_top=False)
            
            for i in range(33, 41): # Pop the last 3 layers in vgg-16
                self.featureExtractor.layers.pop()
        else:
            self.featureExtractor = None


    def crop(self, gray, frame, outputPath):
        """
        Takes in a grayscale image and returns a cropped image that just includes the face.
        """
        # Detect face and crop image to just face (input to emotion classifier)
        faces = self.faceDetector.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
        if len(faces) > 0:  # If face is detected...
            for (x, y, w, h) in faces:

                # Display rectangle around face in the frame
                if frame is not None:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

                # Extract face and resize to optimal input for classifier
                face = gray[y:y+h, x:x+w]
                face = imutils.resize(face, width=48)

                cv2.imwrite('demo.jpg', face)   # Ideally, I would like to not have to do this
                return frame


    def predict(self, face):

        # If applicable, extract features 
        if self.featureExtractor is not None:
            inputs = self.extractFeatures(face)
        else:
            inputs = face

        # Run prediction
        return self.emotionClassifier.predict(inputs)[0]


    def extractFeatures(self, face):
        face = cv2.imread("demo.jpg")   # Ideally, I would like to not have to do this

        #print("Nonwritten shape (Before reshaping)", face.shape)
        # Convert face image to tensor
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
    demo = livepredicts(faceDetectionPath, emotionClassifierPath, 'vgg')
    demo.launch()

    """
    # To run cropping for images
    demo = livedemo(faceDetectionPath, emotionClassifierPath, 'vgg')
    image = cv2.imreaad("pathtoimagehere")
    demo.crop(image, frame=None, outputPath="outputPathhere")
    """


if __name__ == '__main__':
    main()