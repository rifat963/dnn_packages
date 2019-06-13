# utility
import pickle
import datetime, random
from datetime import datetime, timedelta
import logging
import math
import warnings
import os

# pandas
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

# numpy
import numpy as np
from numpy import concatenate

# plot libraries
import matplotlib.pyplot as plt
from matplotlib import pyplot
from plotly.offline import iplot
from plotly import graph_objs as go
import seaborn as sns

# sklearn

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# keras with tensorflow
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from keras.layers import Lambda
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras import regularizers
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers.merge import concatenate as concat
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras_tqdm import TQDMNotebookCallback

from dnn_packages.ml_data_modeling_test.ml_models.utilities import utilities

# facebook prophet package
from fbprophet import Prophet

# XGB model
import xgboost as xgb


class stacked_autoencoder(object):

    def __init__(self):
        pass

    def create_autoencoder_model(self,train_x, params):
        '''
        Autoencoder Layer Structure and Parameters
        Args:
        1) params -> dict with model parameters
        Out:
        1) autoencoder -> Autoencoder Layer Structure and Parameters
        '''

        #nb_epoch = 100
        #batch_size = 128
        input_dim = train_x.shape[1]  # num of columns, 30
        encoding_dim = 14
        hidden_dim = int(encoding_dim / 2)  # i.e. 7
        learning_rate = 1e-7

        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation="tanh",activity_regularizer=regularizers.l1(learning_rate))(input_layer)
        encoder = Dense(hidden_dim, activation="relu")(encoder)
        decoder = Dense(hidden_dim, activation='tanh')(encoder)
        decoder = Dense(input_dim, activation='relu')(decoder)
        autoencoder = Model(inputs=input_layer, outputs=decoder)

        return autoencoder

    def model_training_logging(self,autoencoder, train_x, params):
        """
        Model training and Logging. Below is where we set up the actual run including checkpoints and the tensorboard.

        """
        nb_epoch = 50
        batch_size = 128
        input_dim = train_x.shape[1]  # num of columns, 30
        encoding_dim = 14
        hidden_dim = int(encoding_dim / 2)  # i.e. 7
        learning_rate = 1e-7
        path_to_pickle = 'gdrive/My Drive/DataSet_Colab_NoteBooks/pickle-machine/'

        file = path_to_pickle + "autoencoder_husky50.h5"

        autoencoder.compile(metrics=['accuracy'],
                            loss='mean_squared_error',
                            optimizer='adam')

        cp = ModelCheckpoint(filepath=file,
                             save_best_only=True,
                             verbose=0)

        logs = '/data/tensorflow_logs'

        tb = TensorBoard(log_dir=logs,
                         histogram_freq=0,
                         write_graph=True,
                         write_images=True)

        history = autoencoder.fit(train_x, train_x,
                                  epochs=nb_epoch,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  # validation_data=(test_x, test_x),
                                  validation_split=0.2,
                                  verbose=1,
                                  callbacks=[TQDMNotebookCallback(), cp, tb]).history

        return history

    def prediction_autoencoder(self,autoencoder, test_x):
        """
        This function makes a one-step prediction

        Args:
            1) model --> autoencoder model (fitted by fit_autoencoder function)
            2) y --> original sample
        Out:
            1) prediction --> preduction results
            2) mse --> mean square error based on difference between original and predicted results

        """

        test_x_predictions = autoencoder.predict(test_x)

        # mse values as reconstruction error
        mse = np.mean(np.power(test_x - test_x_predictions, 2), axis=1)

        return test_x_predictions, mse
