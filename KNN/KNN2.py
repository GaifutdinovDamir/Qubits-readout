from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


all_data = pd.read_csv("Training examples.csv")
data = all_data[all_data.columns[1:3]].values
labels = all_data[all_data.columns[-1]].values
train_x, test_x, train_y, test_y = train_test_split(data, labels,
                                                    test_size=0.1)

model = KNeighborsClassifier(n_neighbors=1, metric="euclidean",
                             weights="distance")
model.fit(train_x, train_y)

y_pred1 = model.predict(train_x)
print(accuracy_score(y_pred1, train_y))

y_pred2 = model.predict(test_x)
print(accuracy_score(y_pred2, test_y))
