from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
# Получаем данные
all_data = pd.read_csv("Training examples.csv")
data = all_data[all_data.columns[1:3]].values
labels = all_data[all_data.columns[-1]].values
train_x, test_x, train_y, test_y = train_test_split(data, labels,
                                                    test_size=0.9,
                                                    random_state=42)
# Получаем модель
filename = 'result_model.sav'
model = pickle.load(open(filename, 'rb'))
# Тестируемся
y_pred1 = model.predict(train_x)
print('Train accuracy:', 100 * accuracy_score(y_pred1, train_y), "%")

y_pred2 = model.predict(test_x)
print('Test accuracy:', 100 * accuracy_score(y_pred2, test_y), "%")



