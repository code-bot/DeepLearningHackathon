from load_data import load
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras import optimizers
import time


class Model(object):
    def __init__(self, o_model):
        self.o_model = o_model

    def set_structure(self, lo_features):
        for o_feat in lo_features:
            try:
                self.o_model.add(o_feat)
            except:
                import pdb; pdb.set_trace()

    def set_optimizer(self, o_optimizer):
        self.o_optimizer = o_optimizer

    def compile_model(self, s_loss='mean_squared_error', ls_metrics=['accuracy']):
        self.o_model.compile(optimizer=self.o_optimizer, loss=s_loss, metrics=ls_metrics)
    
    def train_model(self, df_xTrain, df_yTrain, i_epochs=200, f_validation_split=.2, b_shuffle=True, i_batch_size=20, b_time = False):
        if b_time:
            t_start = time.time()
        import pdb; pdb.set_trace()
        history = self.o_model.fit(df_xTrain.reshape(df_yTrain.shape[0],-1) , df_yTrain, validation_split=f_validation_split, shuffle=b_shuffle, epochs=i_epochs, batch_size=i_batch_size)
        self.history = history
        if b_time:
            t_end = time.time()
            t_runtime = t_end - t_start
            return history.history, t_runtime
        return history.history
    
    def test_model(self, df_xTest, i_batch_size=20):
        return self.o_model.predict(df_xTest, batch_size=i_batch_size)
    
    def evaluate_model(self, df_xTest, df_yTest):
        return self.o_model.evaluate(df_xTest, df_yTest)
    
    def get_model(self):
        return self.o_model

# def buildModel():
#     model = Sequential()
#     model.add(Dense(100, activation="relu", input_shape=(96*96,)))
#     model.add(Activation('relu'))
#     model.add(Dense(30))

#     sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

#     return model


# def trainModel(model, data):
#     training, testing = data
#     xTrain, yTrain = training
#     xTest, yTest = testing

#     epochs = 200
#     start_time = time.time()
#     history = model.fit(xTrain.reshape(yTrain.shape[0], -1), yTrain, 
#                      validation_split=0.2, shuffle=True, 
#                      epochs=epochs, batch_size=20)
#     end_time = time.time() - start_time
#     print("--> Total training time: %i sec" % end_time)

#     return history.history

# def evaluateModel(model, x, y):                     
#     evaluation = self.model.evaluate(x, y,verbose=0)
#     print("RMSE:", evaluation[0], "\t MSE:", evaluation[1])

# def saveModelWeights(model, model_path):
#     model.save(model_path)
#     print("--> Model saved at path:", model_path)

# def loadModelWeights(model_path):
#     model = load_model(model_path)
#     print("--> Model loaded from path:", model_path)
#     return model

if __name__=="__main__":
    tna_train, nai_xTest = load()
    nai_xTrain, nai_yTrain = tna_train
    
    o_model1 = Model(Sequential())
    # import pdb; pdb.set_trace()
    lo_model1_features = [Dense(100, activation="relu", input_shape=(96*96,)), Activation('relu'), Dense(nai_yTrain.shape[1])]
    o_optimizer_sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # o_optimizer_
    
    o_model1.set_structure(lo_model1_features)
    o_model1.set_optimizer(o_optimizer_sgd)
    o_model1.compile_model()


    o_model1.train_model(nai_xTrain, nai_yTrain, i_epochs=5)
    df_yPred = o_model1.test_model(nai_xTest)
    import pdb; pdb.set_trace()

