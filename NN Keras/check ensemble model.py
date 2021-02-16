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
ensemble_model = keras.models.load_model("2-20-10-8-6-1")
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
