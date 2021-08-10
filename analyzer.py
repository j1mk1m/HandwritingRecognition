import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python analyzer.py model.h5")
    
    #load photo to be analyzed with cv2
    photoArray, names = loadPhoto()
    print("All photos sucessfully loaded")
    #print(photoArray.shape)

    #load model with tensorflow
    model = loadModel(sys.argv[1])
    print("Model successfully loaded")

    #execute model
    print("Predicting photos...")
    resultArray = model.predict(photoArray)

    #return result
    printResult(resultArray, names)
    
    #terminate
    print("All photos predicted. END")


def loadPhoto():
    images = []
    names = []
    for image in os.listdir("Analyze"):
        img = cv2.imread(os.path.join("Analyze",image))
        if (img is not None):
            img = cv2.resize(img, (30,30))
            images.append(img)
            names.append(image)
        else:
            print("Could not find photo named: "+image)
    return np.array(images), names

def loadModel(model_name):
    return tf.keras.models.load_model(model_name)

def printResult(resultArray, names):
    for i in range(len(resultArray)):
        result = np.argmax(resultArray[i])
        print(names[i]+" seems to be a \'", end = "")
        print(chr(result), end = "")
        print("\'")

main()
