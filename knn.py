# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:16:30 2017

@author: mbesancon
"""

import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv("data/train.csv")
matrix = raw_data.iloc[:, 1:].values
labels = raw_data['label']

pca_res = PCA(n_components=50).fit(matrix)
matrix_pca = pca_res.transform(matrix)

confusion_results = []
for _ in range(20):
    print("New iteration")
    X_train, X_test, y_train, y_test = train_test_split(
        matrix, labels, test_size=0.4, random_state=0)
    clf = neighbors.KNeighborsClassifier(weights='distance')
    clf.fit(X_train, y_train)
    z = clf.predict(X_test)
    confusion_results.append(np.array(confusion_matrix(y_test, z)))

final_res = np.zeros((10, 10), dtype=int)
for cr in confusion_results:
    final_res += cr

accuracy_knn = 1 - final_res.diagonal().sum() / final_res.sum().sum()

confusion_results_pca = []
for it in range(20):
    print("Iteration ", it)
    X_train, X_test, y_train, y_test = train_test_split(
        matrix_pca, labels, test_size=0.4, random_state=0)
    clf = neighbors.KNeighborsClassifier(weights='distance')
    clf.fit(X_train, y_train)
    z = clf.predict(X_test)
    confusion_results_pca.append(np.array(confusion_matrix(y_test, z)))

final_res_pca = np.zeros((10, 10), dtype=int)
for cr in confusion_results_pca:
    final_res_pca += cr

accuracy_knn_pca = 1 - final_res_pca.diagonal().sum() / final_res_pca.sum().sum()
