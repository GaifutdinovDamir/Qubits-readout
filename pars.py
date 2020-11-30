import csv
import random

with open("zeros2.txt", "r") as file:
    a = file.readlines()
    zeros = []
    label = []
    for elem in a[1:]:
        zeros.append(complex(elem.strip()))
        label.append(0)
with open("ones2.txt", "r") as file:
    a = file.readlines()
    ones = []
    for elem in a[1:]:
        ones.append(complex(elem.strip()))
        label.append(1)
data = zeros + ones
for j in range(len(data)):
    r = random.randint(0, len(data))
    data[j], data[r] = data[r], data[j]
    label[j], label[r] = label[r], label[j]
with open("Training examples.csv", "a") as file:
    file_writer = csv.writer(file, delimiter=",", lineterminator="\r")
    for i in range(len(data)):
        file_writer.writerow([data[i].real, data[i].imag, label[i]])
