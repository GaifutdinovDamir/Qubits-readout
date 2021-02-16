from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy as np
from sklearn.metrics import accuracy_score

def get_data(a):
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
            if i % 10 == a and i < constant_data:
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
    test_y = np.array(test_y).T.ravel()
    train_x = np.array(train_x).T
    train_y = np.array(train_y).T.ravel()
    return train_x, train_y, test_x, test_y


s1 = 0
s2 = 0
for test in range(8):
    print(test)
    train_x, train_y, test_x, test_y = get_data(test)
    """
    test=0
    13
    0.9393236529329417
    0.9391057312252964
    """
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(train_x, train_y)

    prediction = neigh.predict(train_x)
    print(accuracy_score(prediction, train_y))

    prediction = neigh.predict(test_x)
    print(accuracy_score(prediction, test_y))
s1 = s1 / 8
s2 = s2 / 8
print(s1 * 100 / 72862)
print(s2 * 100 / 8096)
