from catboost import CatBoostClassifier, CatBoostRegressor
import numpy
import csv
import numpy as np
#from sklearn import cross_validation
from sklearn.metrics import accuracy_score
constant_train = 72862
train_x = [[0] * constant_train, [0] * constant_train]
train_y = [0] * constant_train
constant_test = 8096
test_x = [[0] * constant_test, [0] * constant_test]
test_y = [0] * constant_test
constant_full = constant_test + constant_train
FILENAME = "Training examples.csv"
with open(FILENAME, "r", newline="") as file:
    reader = csv.reader(file)
    i = 0
    k = 0
    for row in reader:
        if i % 10 == 0 and i < constant_full:
            test_x[0][int(i / 10)] = float(row[0])
            test_x[1][int(i / 10)] = float(row[1])
            test_y[int(i / 10)] = float(row[2])
            i += 1
            k += 1
        elif i < constant_full:
            train_x[0][i - k] = float(row[0])
            train_x[1][i - k] = float(row[1])
            train_y[i - k] = float(row[2])
            i += 1
test_x = np.array(test_x).T
test_y = np.array(test_y)

train_x = np.array(train_x).T
train_y = np.array(train_y)
"""
depth=10
iterations=1000
Train accuracy: 94.15195849688452
Test accuracy: 93.35474308300395
depth=10
iterations=5000
Train accuracy: 94.19313222255771
Test accuracy: 93.18181818181817
"""
cat = CatBoostClassifier()
cat.load_model("cat.cbm")
y_prediction = cat.predict(train_x)
print(cat.predict(train_x))
predictions = [round(value) for value in y_prediction]
accuracy = accuracy_score(train_y, predictions)
print("Train accuracy: %s" % (accuracy * 100.0))

y_prediction = cat.predict(test_x)
predictions = [round(value) for value in y_prediction]
accuracy = accuracy_score(test_y, predictions)
print("Test accuracy: %s" % (accuracy * 100.0))

cat.set_params(learning_rate=0.5)

y_prediction = cat.predict(train_x)
print(cat.predict(train_x))
predictions = [round(value) for value in y_prediction]
accuracy = accuracy_score(train_y, predictions)
print("Train accuracy: %s" % (accuracy * 100.0))

y_prediction = cat.predict(test_x)
predictions = [round(value) for value in y_prediction]
accuracy = accuracy_score(test_y, predictions)
print("Test accuracy: %s" % (accuracy * 100.0))
cat.save("cat.cbm")