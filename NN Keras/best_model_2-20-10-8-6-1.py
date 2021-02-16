import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import csv


def get_data():
    constant_train = 72862
    train_x = [[0] * constant_train, [0] * constant_train]
    train_y = [[0] * constant_train]
    constant_test = 8096
    test_x = [[0] * constant_test, [0] * constant_test]
    test_y = [[0] * constant_test]
    constant_data = constant_train + constant_test
    FILENAME = "Training examples.csv"
    with open(FILENAME, "r", newline="") as file:
        reader = csv.reader(file)
        i = 0
        k = 0
        for row in reader:
            if i % 10 == 0 and i < constant_data:
                test_x[0][int(i / 10)] = float(row[0])
                test_x[1][int(i / 10)] = float(row[1])
                test_y[0][int(i / 10)] = float(row[2])
                i += 1
                k += 1
            elif i < constant_data:
                train_x[0][i - k] = float(row[0])
                train_x[1][i - k] = float(row[1])
                train_y[0][i - k] = float(row[2])
                i += 1
    test_x = np.array(test_x).T
    test_y = np.array(test_y).T
    train_x = np.array(train_x).T
    train_y = np.array(train_y).T
    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = get_data()
"""
Train loss: 0.21995222568511963
Train accuracy: 93.96805167198181
--------------------------------------------------
--------------------------------------------------
Test loss: 0.2240344136953354
Test accuracy: 93.94763112068176
"""


def get_model(train_x=train_x, train_y=train_y):
    inputs = keras.Input(shape=(2,))
    x = layers.Dense(20, activation="relu")(inputs)
    x = layers.Dense(10, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(6, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=["accuracy"])
    model.fit(train_x, train_y, epochs=20, batch_size=128)
    return model


model_1 = get_model()
model_2 = get_model()
model_3 = get_model()
model_4 = get_model()
model_5 = get_model()
model_6 = get_model()
model_7 = get_model()
model_8 = get_model()
model_9 = get_model()
model_10 = get_model()

inputs = keras.Input(shape=(2,))

y1 = model_1(inputs)
y2 = model_2(inputs)
y3 = model_3(inputs)
y4 = model_4(inputs)
y5 = model_5(inputs)
y6 = model_1(inputs)
y7 = model_2(inputs)
y8 = model_3(inputs)
y9 = model_4(inputs)
y10 = model_5(inputs)

all_y = layers.concatenate([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10])
x = layers.Dense(10, activation="relu")(all_y)
x = layers.Dense(10, activation="relu")(x)
x = layers.Dense(10, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
ensemble_model = keras.Model(inputs=inputs, outputs=outputs)
ensemble = ensemble_model.compile(loss=keras.losses.binary_crossentropy,
                                  optimizer=keras.optimizers.Adam(),
                                  metrics=["accuracy"])
ensemble_model.fit(train_x, train_y, epochs=100, batch_size=128)
train_scores = ensemble_model.evaluate(train_x, train_y)
print("-" * 50)
print("Train loss:", train_scores[0])
print("Train accuracy:", 100 * train_scores[1])
print("-" * 50)
test_scores = ensemble_model.evaluate(test_x, test_y)
print("-" * 50)
print("Test loss:", test_scores[0])
print("Test accuracy:", 100 * test_scores[1])
print("-" * 50)
ensemble_model.save("best_model")
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import csv


def get_data():
    constant_train = 72862
    train_x = [[0] * constant_train, [0] * constant_train]
    train_y = [[0] * constant_train]
    constant_test = 8096
    test_x = [[0] * constant_test, [0] * constant_test]
    test_y = [[0] * constant_test]
    constant_data = constant_train + constant_test
    FILENAME = "Training examples.csv"
    with open(FILENAME, "r", newline="") as file:
        reader = csv.reader(file)
        i = 0
        k = 0
        for row in reader:
            if i % 10 == 0 and i < constant_data:
                test_x[0][int(i / 10)] = float(row[0])
                test_x[1][int(i / 10)] = float(row[1])
                test_y[0][int(i / 10)] = float(row[2])
                i += 1
                k += 1
            elif i < constant_data:
                train_x[0][i - k] = float(row[0])
                train_x[1][i - k] = float(row[1])
                train_y[0][i - k] = float(row[2])
                i += 1
    test_x = np.array(test_x).T
    test_y = np.array(test_y).T
    train_x = np.array(train_x).T
    train_y = np.array(train_y).T
    return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = get_data()

"""
"""
Train loss: 0.21995222568511963
Train accuracy: 93.96805167198181
--------------------------------------------------
253/253 [==============================] - 2s 5ms/step - loss: 0.2240 - accuracy: 0.9395
--------------------------------------------------
Test loss: 0.2240344136953354
Test accuracy: 93.94763112068176
"""
"""

def get_model(train_x=train_x, train_y=train_y):
    inputs = keras.Input(shape=(2,))
    x = layers.Dense(20, activation="relu")(inputs)
    x = layers.Dense(10, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(x)
    x = layers.Dense(6, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=["accuracy"])
    model.fit(train_x, train_y, epochs=5000, batch_size=80000)
    return model


model_1 = get_model()
model_2 = get_model()
model_3 = get_model()
model_4 = get_model()
model_5 = get_model()
inputs = keras.Input(shape=(2,))
y1 = model_1(inputs)
print("Complete y1")
y2 = model_2(inputs)
print("Complete y2")
y3 = model_3(inputs)
print("Complete y3")
y4 = model_4(inputs)
print("Complete y4")
y5 = model_5(inputs)
print("Complete y5")
all_y = layers.concatenate([y1, y2, y3, y4, y5])
x = layers.Dense(25, activation="relu")(all_y)
x = layers.Dense(25, activation="relu")(x)
x = layers.Dense(25, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
ensemble_model = keras.Model(inputs=inputs, outputs=outputs)
ensemble = ensemble_model.compile(loss=keras.losses.binary_crossentropy,
                                  optimizer=keras.optimizers.Adam(),
                                  metrics=["accuracy"])
ensemble_model.fit(train_x, train_y, epochs=5000, batch_size=80000)
train_scores = ensemble_model.evaluate(train_x, train_y)
print("-" * 50)
print("Train loss:", train_scores[0])
print("Train accuracy:", 100*train_scores[1])
print("-" * 50)
test_scores = ensemble_model.evaluate(test_x, test_y)
print("-" * 50)
print("Test loss:", test_scores[0])
print("Test accuracy:", 100*test_scores[1])
print("-" * 50)
ensemble_model.save("ensemble_model")
"""
