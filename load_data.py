import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_key_features(s_file, b_filter=False):
    s_file = os.path.join(os.getcwd(), 'data', s_file)
    df_data = pd.read_csv(s_file)
    if b_filter:
        df_data = df_data.dropna(axis=0, how='any')
    return df_data


def stringToNAF(s_string):
    return np.reshape(np.fromstring(s_string, dtype=int,sep=' '), (96, 96))

def get_training(b_filter):
    df_data = load_key_features('training.csv', b_filter)
    xTrain = df_data[df_data.columns[-1]]
    yTrain = df_data[df_data.columns[:-1]]
    xTrain = xTrain.apply(stringToNAF)
    return xTrain, yTrain

def get_test():
    df_data = load_key_features('test.csv')
    df_data = df_data.set_index('ImageId')
    xTest = df_data['Image']
    xTest = xTest.apply(stringToNAF)
    return xTest

def load(b_filter=True):
    xTrain, yTrain = get_training(b_filter)
    xTest = get_test()
    return (xTrain, yTrain), xTest


if __name__=="__main__":
    load()