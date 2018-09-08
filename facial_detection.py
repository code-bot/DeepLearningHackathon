import load_data as data
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras import optimizers

def buildModel():
    model = Sequential()
    model.add(Dense(100, activation="relu", input_shape=(96*96,)))
    model.add(Activation('relu'))
    model.add(Dense(30))

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

    return model


def trainModel(model, data):
    training, testing = data
    xTrain, yTrain = training
    xTest, yTest = testing

    epochs = 200
    start_time = time.time()
    history = model.fit(xTrain.reshape(yTrain.shape[0], -1), yTrain, 
                     validation_split=0.2, shuffle=True, 
                     epochs=epochs, batch_size=20)
    end_time = time.time() - start_time
    print("--> Total training time: %i sec" % end_time)

    return history.history

def evaluateModel(model, x, y):                     
    evaluation = self.model.evaluate(x, y,verbose=0)
    print("RMSE:", evaluation[0], "\t MSE:", evaluation[1])

def saveModelWeights(model, model_path):
    model.save(model_path)
    print("--> Model saved at path:", model_path)

def loadModelWeights(model_path):
    model = load_model(model_path)
    print("--> Model loaded from path:", model_path)
    return model
