"""
Imports
"""

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
Setting up data
"""

path = os.getcwd() + '/train_data.pkl'
with open(path, 'rb') as f :
    data = pickle.load(f)

images = data['images']
labels = data['labels']


"""
Analyzing data type and structure
"""

print('Types des images : ', type(images))
print('Type des labels : ', type(labels))

print('Dimension des images : ', images.shape)
print('Dimension des labels : ', labels.shape)

print('Type des données : ', images.dtype)
print('Type des labels : ', labels.dtype)

print('Extremum des images : Min = ', images.min(), '-- Max = ', images.max())
print('Labels ', np.unique(labels))


"""
Distribution des labels
"""
classes, count = np.unique(labels, return_counts=True)
plt.figure()
plt.bar(classes, count)
plt.xlabel('Labels')
plt.ylabel('Count')
plt.title('Compte des labels')
plt.show()






