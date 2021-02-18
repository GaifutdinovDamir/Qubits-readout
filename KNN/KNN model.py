from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
# Получаем данные
all_data = pd.read_csv("Training examples1.csv")
data = all_data[all_data.columns[0:2]].values
labels = all_data[all_data.columns[-1]].values
train_x, test_x, train_y, test_y = train_test_split(data, labels,
                                                    test_size=0.1)
# Учимся
model = KNeighborsClassifier(n_neighbors=4, metric="manhattan",
                             weights="uniform")
param_grid = {
    "n_neighbors": np.arange(1, 25),
    "metric": ["euclidean","manhattan","minkowski"],
    "weights": ["distance", "uniform"]
}

# создадим объект GridSearchCV
#search = GridSearchCV(model, param_grid, n_jobs=-1, cv=5, refit=True, scoring='accuracy')
model.fit(train_x, train_y)


# Тестируемся
y_pred1 = model.predict(train_x)
print('Train accuracy:', 100 * accuracy_score(y_pred1, train_y), "%")

y_pred2 = model.predict(test_x)
print('Test accuracy:', 100 * accuracy_score(y_pred2, test_y), "%")
# Сохраняемся
"""
filename = 'result_model.sav'
pickle.dump(model, open(filename, 'wb'))
"""
