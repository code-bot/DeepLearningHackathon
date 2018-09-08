import load_data
import os
from keras import optimizers

from keras.models import Sequential, load_model
from keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D, Dense, GlobalAveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

def string2image(string):
    """Converts a string to a numpy array."""
    return np.array([int(item) for item in string.split()]).reshape((96, 96))

df = load_data.load_key_features('training.csv', b_filter=False)
# import pdb; pdb.set_trace()   

keypoint_cols = list(df.columns)[:-1]

xy = df.iloc[0][keypoint_cols].values.reshape((15, 2))

fully_annotated = df.dropna()

X = np.stack([string2image(string) for string in fully_annotated['Image']]).astype(np.float)[:, :, :, np.newaxis]

y = np.vstack(fully_annotated[fully_annotated.columns[:-1]].values)

X_train = X / 255.

output_pipe = make_pipeline(
    MinMaxScaler(feature_range=(-1, 1))
)

y_train = output_pipe.fit_transform(y)


# model = Sequential()
# model.add(Dense(100, activation="relu", input_shape=(96*96,)))
# model.add(Activation('relu'))
# model.add(Dense(30))



# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
# epochs = 200
# history = model.fit(X_train.reshape(y_train.shape[0], -1), y_train, 
#                 validation_split=0.2, shuffle=True, 
#                 epochs=epochs, batch_size=20)

# model.save('model_sgd_mse')



def get_score(img):
    model = load_model('model_sgd_mse')

    pred_data =X_train[0, :, :, :].reshape(1, -1)

    results = model.predict(img)
    

    import pdb; pdb.set_trace()

if __name__=="__main__":
    get_score('hello')









