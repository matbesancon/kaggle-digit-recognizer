# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:16:30 2017

@author: mbesancon
"""

import sys
import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# read which matrix to read as optional argument
if len(sys.argv) > 1:
    matrix_file = sys.argv[1]
else:
    matrix_file = "data/matrix.csv"

# Minkowski distance power
if len(sys.argv) > 2:
    distance_metric = sys.argv[2]
else:
    distance_metric = '2'
if distance_metric not in {'1', '2'}:
    distance_metric = 2
else:
    distance_metric = int(distance_metric)

matrix = pd.read_csv(matrix_file, header=None).values
labels = pd.read_csv("data/labels.csv", header=None).values[:, 0]

confusion_results = []
for _ in range(20):
    X_train, X_test, y_train, y_test = train_test_split(
        matrix, labels, test_size=0.4, random_state=0)
    clf = neighbors.KNeighborsClassifier(weights='distance', p=distance_metric)
    clf.fit(X_train, y_train)
    z = clf.predict(X_test)
    confusion_results.append(np.array(confusion_matrix(y_test, z)))

final_res = np.zeros((10, 10), dtype=int)
for cr in confusion_results:
    final_res += cr

accuracy_knn = 1 - final_res.diagonal().sum() / final_res.sum().sum()
print("Distance KNN accuracy: ", accuracy_knn)

confusion_results_unif = []
for _ in range(20):
    X_train, X_test, y_train, y_test = train_test_split(
        matrix, labels, test_size=0.4, random_state=0)
    clf = neighbors.KNeighborsClassifier(weights='uniform')
    clf.fit(X_train, y_train)
    z = clf.predict(X_test)
    confusion_results_unif.append(np.array(confusion_matrix(y_test, z)))

final_res_unif = np.zeros((10, 10), dtype=int)
for cr in confusion_results_unif:
    final_res_unif += cr

accuracy_knn_unif = 1 - final_res_unif.diagonal().sum() /\
                    final_res_unif.sum().sum()
print("Uniform KNN accuracy: ", accuracy_knn_unif)

# write results to csv
if '/' in matrix_file:
    matrix_file_name = matrix_file.split('/')[-1]
else:
    matrix_file_name = matrix_file

res_frame = pd.DataFrame(final_res)
res_frame['weights'] = 'distance'

res_frame_unif = pd.DataFrame(final_res_unif)
res_frame_unif['weights'] = 'uniform'

res_tot = pd.concat([res_frame, res_frame_unif])
res_tot.to_csv("results/" + str(int(pd.Timestamp('now').timestamp())) +
               "_" + matrix_file_name + "_knn.csv")
