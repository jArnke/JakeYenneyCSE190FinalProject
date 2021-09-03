import tensorflow as tf
import dataPipeline

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
def buildModel(X, Y):
    model = Sequential()

    # Layer 1
    model.add(
        Conv2D(
            64,
            (3, 3),
            input_shape=X.shape[1:]
        )
    )
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    # Layer 2
    model.add(
        Conv2D(
            128,
            (3, 3),
            input_shape=X.shape[1:]
        )
    )
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Dropout(0.1))

    # layer 3
    model.add(
        Conv2D(
            128,
            (3, 3),
            input_shape=X.shape[1:]
        )
    )
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(128))

    model.add(Dense(len(Y[0])))
    model.add(Activation("softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    return model

#load data:
length = 15
retrain = False
X, Y, Xtest, Ytest = dataPipeline.createCNNData("data/genres", length)

#Train the model and save the weights:
model = buildModel(X, Y)
if retrain:
    model.fit(X, Y, epochs=30, validation_split = .1)
    model.save_weights("weights/CNN/"+str(length))


#Test the model
model.load_weights("weights/CNN/"+str(length))

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

confusion = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

for i, prediction in enumerate(predictions):
    if Ytest[i][indexOfMax(prediction)] == 1:
        print("Success")
        successCount += 1
    else:
        print("Failure")
        print("\t"+str(prediction))
        failCount += 1
    confusion[indexOfMax(Ytest[i])][indexOfMax(prediction)] += 1

accuracy = successCount / (successCount+failCount)
print("Accuracy = " + str(accuracy))

print(confusion)


import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


genreDict = {"blues":(1,0,0,0,0,0,0,0,0,0), "classical":(0,1,0,0,0,0,0,0,0,0), "country":(0,0,1,0,0,0,0,0,0,0),
               "disco":(0,0,0,1,0,0,0,0,0,0), "hiphop":(0,0,0,0,1,0,0,0,0,0), "jazz":(0,0,0,0,0,1,0,0,0,0),
               "metal":(0,0,0,0,0,0,1,0,0,0), "pop":(0,0,0,0,0,0,0,1,0,0), "reggae":(0,0,0,0,0,0,0,0,1,0),
               "rock":(0,0,0,0,0,0,0,0,0,1)}
classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
confusion = np.array(confusion)
plot_confusion_matrix(confusion, classes)

from sklearn import metrics

