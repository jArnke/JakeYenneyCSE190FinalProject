import librosa
import librosa.display
import numpy
import matplotlib.pyplot as plt

import pickle

from os import listdir
from os.path import isfile, join

import random

# import waveform signal:
def getMelSpec(filePath, length=None):
    if length != None:
        signal, sampleRate = librosa.load(filePath, sr=None, duration = length)
    else:
        signal, sampleRate = librosa.load(filePath, sr=None)

    melSpec = librosa.feature.melspectrogram(signal, sr=sampleRate)
    melSpec = librosa.power_to_db(melSpec)

    return melSpec, sampleRate

#Display Mel Spectrogram
def showMelSpec(melSpec, sampleRate):
    plt.figure(figsize=(25,10))
    librosa.display.specshow(melSpec, x_axis="time",y_axis="mel",sr=sampleRate)

    plt.colorbar(format="%+2.f")
    plt.show()

def showSpec(dataFile):
    signal, sr = librosa.load(dataFile)
    D = librosa.stft(signal)
    S_db = librosa.amplitude_to_db(numpy.abs(D))

    plt.figure()
    librosa.display.specshow(S_db)
    plt.colorbar()

def parseAlbum(filePath, durr, outputTuple):

    print("parsing " + filePath)
    onlyfiles = [f for f in listdir(filePath)]
    print(onlyfiles)
    data = [getMelSpec(str(filePath+"/" + file), durr) for file in onlyfiles]

    X = [a for (a, b) in data]

    #Check for mismatched sample rates:


    album = [ (song, outputTuple) for song in X ]
    return album


def createCNNData(dataDir, length):

    #clips vary between 29 to 30 seconds so 30 seconds produces unusable input
    if length >= 30:
        length = 29

    albumsFilepaths = [f for f in listdir(dataDir)]

    data = []
    testData = []

    genreDict = {"blues":(1,0,0,0,0,0,0,0,0,0), "classical":(0,1,0,0,0,0,0,0,0,0), "country":(0,0,1,0,0,0,0,0,0,0),
                   "disco":(0,0,0,1,0,0,0,0,0,0), "hiphop":(0,0,0,0,1,0,0,0,0,0), "jazz":(0,0,0,0,0,1,0,0,0,0),
                   "metal":(0,0,0,0,0,0,1,0,0,0), "pop":(0,0,0,0,0,0,0,1,0,0), "reggae":(0,0,0,0,0,0,0,0,1,0),
                   "rock":(0,0,0,0,0,0,0,0,0,1)}


    for i, f in enumerate(albumsFilepaths):
        outputTup = [0,0,0,0,0,0,0,0,0,0]
        outputTup[i] = 1
        tempData = parseAlbum(dataDir+"/"+f, length, outputTup)
        data += tempData[11:]
        testData += tempData[:10]
        print(len(testData))
    print(len(data))


    #Split/reshape data
    random.shuffle(data)
    random.shuffle(testData)

    X = [a for (a, b) in data]
    Y = [b for (a, b) in data]

    X = numpy.array(X)
    X = X.reshape(-1, len(X[0]), len(X[0][0]), 1)

    Y = numpy.array(Y)

    Xtest = [a for (a, b) in testData]
    Ytest = [b for (a, b) in testData]

    Xtest = numpy.array(Xtest)
    Xtest = Xtest.reshape(-1, len(Xtest[0]), len(Xtest[0][0]), 1)

    Ytest = numpy.array(Ytest)

    #Save data

    with open("X"+str(length)+".pickle", 'wb') as f:
        pickle.dump(X, f)
    with open("Y"+str(length)+".pickle", 'wb') as f:
        pickle.dump(Y, f)

    return X, Y, Xtest, Ytest

def createRNNData(dataDir, length):
    #clips vary between 29 to 30 seconds so 30 seconds produces unusable input
    if length >= 30:
        length = 29

    albumsFilepaths = [f for f in listdir(dataDir)]

    data = []
    testData = []

    for i, f in enumerate(albumsFilepaths):
        outputTup = [0,0,0,0,0,0,0,0,0,0]
        outputTup[i] = 1
        tempData = parseAlbum(dataDir+"/"+f, length, outputTup)
        data += tempData[11:]
        testData += tempData[:10]




    #Split/reshape data
    random.shuffle(data)
    random.shuffle(testData)
    X = [a for (a, b) in data]
    Y = [b for (a, b) in data]

    X = [a.T for a in X]

    Y = numpy.array(Y)

    Xtest = [a for (a, b) in testData]
    Ytest = [b for (a, b) in testData]

    Xtest = [a.T for a in Xtest]

    Ytest = numpy.array(Ytest)

    #Save data

    with open("XRNN"+str(length)+".pickle", 'wb') as f:
        pickle.dump(X, f)
    with open("YRNN"+str(length)+".pickle", 'wb') as f:
        pickle.dump(Y, f)

    return X, Y, Xtest, Ytest
