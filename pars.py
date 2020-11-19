import csv
with open("zeros.txt", "r") as file:
    a = file.readlines()
    zeros = []
    for elem in a[1:]:
        zeros.append(complex(elem.strip()))
with open("ones.txt", "r") as file:
    a = file.readlines()
    ones = []
    for elem in a[1:]:
        ones.append(complex(elem.strip()))
with open("Training examples.csv", "a") as file:
    file_writer = csv.writer(file, delimiter=",", lineterminator="\r")
    for i in range(len(zeros)):
        file_writer.writerow([zeros[i].real, zeros[i].imag, 0])
    for i in range(len(ones)):
        file_writer.writerow([ones[i].real, ones[i].imag, 1])