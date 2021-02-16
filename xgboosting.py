import numpy
import xgboost
import csv
import numpy as np
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

model = xgboost.XGBClassifier(learning_rate=0.1, n_estimators=10000,
                              tree_method="gpu_hist", max_depth=20)

model.fit(train_x, train_y, verbose=True)
print(model)

y_prediction = model.predict(train_x)
predictions = [round(value) for value in y_prediction]
accuracy = accuracy_score(train_y, predictions)
print("Train accuracy: %s" % (accuracy * 100.0))

y_prediction = model.predict(test_x)
predictions = [round(value) for value in y_prediction]
accuracy = accuracy_score(test_y, predictions)
print("Test accuracy: %s" % (accuracy * 100.0))
