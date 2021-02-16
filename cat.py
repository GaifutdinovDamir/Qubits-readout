from catboost import CatBoostClassifier, CatBoostRegressor
import numpy
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


cat = CatBoostClassifier(iterations=10000, task_type='GPU',
                         devices='0-3', depth=4, learning_rate=0.01,
                         loss_function='CrossEntropy')
cat.fit(train_x, train_y, save_snapshot=True, snapshot_file="catboost_info/cat9.cbsnapshot")

y_prediction = cat.predict(train_x)
predictions = [value for value in y_prediction]
accuracy = accuracy_score(train_y, predictions)
print("Train accuracy: %s" % (accuracy * 100.0))

y_prediction = cat.predict(test_x)
predictions = [value for value in y_prediction]
accuracy = accuracy_score(test_y, predictions)
print("Test accuracy: %s" % (accuracy * 100.0))


"""
***
CatBoostClassifier
***
---snapshot_file=catboost_info/cat.cbsnapshot---
depth=10
iterations=15000
learning_rate=1

---snapshot_file=catboost_info/cat2.cbsnapshot---
depth=16
iterations=5000
learning_rate=1
94.1931322225577---93.1941699604743

---snapshot_file=catboost_info/cat3.cbsnapshot---
depth=8
iterations=50000
learning_rate=1

---snapshot_file=catboost_info/cat4.cbsnapshot---
depth=6
iterations=20000
learning_rate=1
94.19038730751284---93.30533596837944
iterations=30000
learning_rate=0.5
94.19175976503527---93.29298418972331

---snapshot_file=catboost_info/cat5.cbsnapshot---
depth=4
iterations=10000
learning_rate=1
94.09157036589717---93.52766798418972
iterations=20000
94.13274409157036---93.36709486166008

---snapshot_file=catboost_info/cat6.cbsnapshot---
depth=10
iterations=5000
learning_rate=1
94.19313222255771---93.16946640316206

---snapshot_file=catboost_info/cat9.cbsnapshot---
cat = CatBoostClassifier(iterations=10000, task_type='GPU', 
devices='0-3', epth=4, learning_rate=0.01, loss_function='CrossEntropy')
93.88021190744146---93.84881422924902
iterations=15000
learning_rate=0.1
93.94060003842881---93.82411067193675
iterations=20000
learning_rate=0.1
93.96393181631029---93.79940711462451
iterations=30000
learning_rate=0.1
94.00373308446103---93.79940711462451
iterations=40000
learning_rate=0.1
94.01196782959568---93.75

---snapshot_file=catboost_info/cat10.cbsnapshot---
cat = CatBoostClassifier(iterations=5000, task_type='GPU', 
devices='0-3',depth=6, learning_rate=0.1, loss_function='CrossEntropy')
94.00373308446103---93.79940711462451
iterations=10000
learning_rate=0.1
94.04216189508935---93.76235177865613

---snapshot_file=catboost_info/cat11.cbsnapshot---
cat = CatBoostClassifier(iterations=5000, task_type='GPU', devices='0-3',
                         depth=6, learning_rate=0.1, loss_function='CrossEntropy',
                         l2_leaf_reg=0.1)
94.050396640224---93.67588932806325
iterations=10000
learning_rate=0.1
94.08608053580741---93.65118577075098

---snapshot_file=catboost_info/cat12.cbsnapshot---
cat = CatBoostClassifier(iterations=10000, task_type='GPU', devices='0-3',
                         depth=6, learning_rate=0.1, loss_function='CrossEntropy',
                         l2_leaf_reg=0.5)
94.0682385880157---93.70059288537549

---snapshot_file=catboost_info/cat13.cbsnapshot---
cat = CatBoostClassifier(iterations=20000, task_type='GPU', devices='0-3',
                         depth=6, learning_rate=0.1, loss_function='CrossEntropy',
                         l2_leaf_reg=30, random_strength=30)
94.00922291455079---93.76235177865613

---snapshot_file=catboost_info/cat14.cbsnapshot---
cat = CatBoostClassifier(iterations=10000, task_type='GPU', devices='0-3',
                         depth=8, learning_rate=0.1, loss_function='CrossEntropy',
                         l2_leaf_reg=25, random_strength=25)
94.016085202163---93.73764822134387
iterations=30000
94.08608053580741---93.63883399209486

---snapshot_file=catboost_info/cat16.cbsnapshot---
cat = CatBoostClassifier(iterations=10000, task_type='GPU', devices='0-3',
                         depth=8, learning_rate=0.1, loss_function='MultiClass',
                         l2_leaf_reg=25, random_strength=25)
93.947462326041---93.83646245059289
iterations=15000
93.96255935878784---93.82411067193675
iterations=20000
93.97765639153468---93.77470355731225

---snapshot_file=catboost_info/cat17.cbsnapshot---
cat = CatBoostClassifier(iterations=10000, task_type='GPU', devices='0-3',
                         depth=8, learning_rate=0.1, loss_function='MultiClass',
                         l2_leaf_reg=30, random_strength=30)
93.94197249595125---93.86116600790514
iterations=20000
93.97216656144492---93.77470355731225

---snapshot_file=catboost_info/cat18.cbsnapshot---
cat = CatBoostClassifier(iterations=5000, task_type='GPU', devices='0-3',
                         depth=8, learning_rate=0.1, loss_function='MultiClass',
                         l2_leaf_reg=30, random_strength=30)
93.92413054815954---93.88586956521739

---snapshot_file=catboost_info/cat20.cbsnapshot---
cat = CatBoostClassifier(iterations=50000, task_type='GPU', devices='0-3',
                         depth=10, learning_rate=0.1, loss_function='MultiClass',
                         l2_leaf_reg=25, random_strength=25)
94.06549367297082---93.67588932806325

---snapshot_file=catboost_info/cat21.cbsnapshot---
cat = CatBoostClassifier(iterations=10000, task_type='GPU', devices='0-3',
                         depth=10, learning_rate=1, loss_function='MultiClass',
                         l2_leaf_reg=5, random_strength=5)

iterations=10000
learning_rate=2

iterations=30000
learning_rate=2
93.39966935961822---92.99309415529729

---snapshot_file=catboost_info/cat22.cbsnapshot---
cat = CatBoostClassifier(iterations=10000, task_type='GPU', devices='0-3',
                         depth=7, learning_rate=1, loss_function='MultiClass',
                         l2_leaf_reg=5, random_strength=5)
94.1368614641377---93.45355731225297

---snapshot_file=catboost_info/cat24.cbsnapshot---
cat = CatBoostClassifier(iterations=10000, task_type='GPU', devices='0-3',
                         depth=7, learning_rate=1, loss_function='MultiClass',
                         l2_leaf_reg=50, random_strength=50)
94.02157503225274---93.71294466403161
iterations=20000
94.05588647031374---93.73764822134387
"""

"""
***
CatBoostRegressor
***
---snapshot_file=catboost_info/cat7.cbsnapshot---
depth=4
iterations=5000
learning_rate=1
83.82558809804837---83.83152173913044

"""
