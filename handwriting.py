import cv2
import numpy as np
import os
import sys
import tensorflow as tf
# =============================================================================
# from tensorflow import keras
# from tensorflow.keras import layers
# =============================================================================

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
TEST_SIZE = 0.3
NUM_CATEGORIES = 123


def main():
    
    # Check command-line arguments
    if len(sys.argv) not in [1, 2]:
        sys.exit("Usage: python traffic.py [model.h5]")

    #load data
    images, labels= loadData()
    print(labels)
    # split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )
    print(labels)
    #get neural network model
    print(len(x_train))
    model = getModel()

    #fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    #evaluate neural network
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        model.save(filename)
        print("Model saved to "+filename+".")


def loadData():
    print("Loading data...")
    dataFolder = "smallset"
    images = []
    labels = []
    for subfolder in os.listdir(dataFolder):
        if subfolder == ".DS_Store":
            continue
        for typeFolder in os.listdir(os.path.join(dataFolder,subfolder)):
            if typeFolder == ".DS_Store":
                continue
            for charFolder in os.listdir(os.path.join(dataFolder,subfolder,typeFolder)):
                if charFolder == ".DS_Store":
                    continue
                for image in os.listdir(os.path.join(dataFolder,subfolder,typeFolder,charFolder)):
                    img = cv2.imread(os.path.join(dataFolder,subfolder,typeFolder,charFolder,image))
                    if img is not None:
                        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                        images.append(img)
                        labels.append(str(int(charFolder,16)))
                print("All ",charFolder, " characters loaded")
            print("Data from " +typeFolder+" loaded.")
    return images,labels


def getModel():
    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(30,30,3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Dense(64, activation = "relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(NUM_CATEGORIES, activation = "softmax")
        ])
    model.compile(
            optimizer = "adam",
            loss = "categorical_crossentropy",
            metrics = ["accuracy"]
        )
    return model




main()