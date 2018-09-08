import os
import pandas as pd
import numpy as np

def load_key_features(s_file):
    s_file = os.path.join(os.getcwd(), 'data', s_file)
    df_data = pd.read_csv(s_file)
    return df_data


if __name__=="__main__":
    load_key_features('training.csv')
    # np.reshape(np.fromstring(df_data.iloc[i_image_id][0], dtype=int,sep=' '), (96, 96))