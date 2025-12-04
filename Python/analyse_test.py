import csv
import numpy as np
import matplotlib.pyplot as plt
import pickle

predictions = []

with open("KNN_1.csv", "r") as f :
    reader = csv.reader(f)

    for row in reader :
        if row[0] != "ID" :
            predictions.append(row[1])

predictions = np.array(predictions)
classes, count = np.unique(predictions, return_counts=True)
plt.figure()
plt.bar(classes, count)
plt.xlabel('Labels')
plt.ylabel('Count')
plt.title('Compte des labels')
plt.show()