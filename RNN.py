import librosa
import librosa.display
import numpy
import matplotlib.pyplot as plt

import tensorflow as tf

import dataPipeline
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, GRU, ConvLSTM2D, LSTM

from os import listdir
from os.path import isfile, join

import random

from scipy.io import wavfile

def buildModel(X, Y):
    model = Sequential()

    #Layer 1
    model.add(
        LSTM(
            128,
            dropout=.3,
            return_sequences = True
        )
    )
    model.add(BatchNormalization())
    #Layer 2
    model.add(
        LSTM(
            128,
            dropout=.3,
            return_sequences = True
        )
    )
    model.add(BatchNormalization())

    #layer 3
    model.add(
        LSTM(
            128,
            dropout=.3,
        )
    )
    model.add(BatchNormalization())
    model.add(Dense(128))


    model.add(Dense(len(Y[0])))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    return model

retrain = True
length = 5
X, Y, Xtest, Ytest = dataPipeline.createRNNData("data/genres", length)

X = numpy.array(X)
X = X.reshape(-1, len(X[0]), len(X[0][0]))
Xtest = numpy.array(Xtest).reshape((-1), len(Xtest[0]), len(Xtest[0][0]))

model = buildModel(X, Y)
#Train the model and save the weights:
if retrain:
    model.fit(X, Y, epochs=30, validation_split = .1)
    model.save_weights("weights/RNN/"+str(length))

model.load_weights("weights/RNN/"+str(length))


#Test with outside data:

predictions = model.predict(Xtest)

successCount = 0
failCount = 0

def indexOfMax(inputList):
    max = inputList[0]
    idx = 0
    for i in range(len(inputList)):
        if inputList[i] > max:
            idx = i
    return idx


for i, prediction in enumerate(predictions):
    if Ytest[i][indexOfMax(prediction)] == 1:
        print("Success")
        successCount += 1
    else:
        print("Failure")
        failCount += 1

accuracy = successCount / (successCount+failCount)
print("Accuracy = " + str(accuracy))
print(Ytest)



