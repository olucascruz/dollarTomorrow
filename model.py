# Basic Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
# Keras Libs
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
# Sklearn Libs
from sklearn.metrics import mean_squared_error
from bcb_service import get_currency
def create_timeseries(series, ts_lag =1):
  dataX = []
  dataY = []

  n_rows = len(series) - ts_lag

  for i in range(n_rows-1):
    a = series[i:(i + ts_lag)]
    dataX.append(a)
    dataY.append(series[i + ts_lag])

  X, Y = np.array(dataX), np.array(dataY)
  return X, Y


def data_processing(data):
    size_data = len(data)
    size_data

    train, test = data[0:size_data//2], data[size_data//2:size_data]


    ts_lag = 3
    trainX, trainY = create_timeseries(train, ts_lag)
    testX, testY = create_timeseries(test, ts_lag)

    # reshape input data to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return trainX, trainY, testX, testY




def create_model(input_size):
    model = Sequential()
    model.add(LSTM(200, input_shape = input_size))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    # Compile the model
    model.compile(loss = "mean_absolute_error", optimizer = "adamax")
    return model

def train_model(model,trainX, trainY):
    # Fit the model
    model.fit(trainX, trainY, epochs = 500, batch_size = 2, verbose =0)
    return model



def get_next_day_currency():
    data = get_currency(60)
    data = data["USD"]
    trainX, trainY, testX, testY = data_processing(data)
    model = create_model(trainX[0].shape)
    model = train_model(model, trainX, trainY)
    print("Chwgou")
    result = model.predict([[[data[-3], data[-2], data[-1]]]])
    status = data[-1] < result[0][0]
    return (result[0][0], status)