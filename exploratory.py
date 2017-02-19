# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 18:58:58 2017

@author: mbesancon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

raw_data = pd.read_csv("data/train.csv")
matrix = raw_data.iloc[:, 1:].values
labels = raw_data['label']


def resquare(array, dim=28):
    """
    builds square (dim x dim) image from an array
    """
    img = np.zeros((dim, dim))
    for i in range(dim):
        img[i, :] = array[dim * i:dim * (i + 1)]
    return img


def viz_pixel(array, dim=28):
    """
    plot the image from an array of pixels
    """
    img = resquare(array, dim)
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    pca_res = PCA(n_components=50).fit(matrix)

    plt.figure()
    plt.plot(pca_res.explained_variance_ratio_)

    matrix_pca = pca_res.transform(matrix)

    plt.figure()
    for label in range(10):
        x = [matrix_pca[i, 0] for i, l in enumerate(labels) if l == label]
        y = [matrix_pca[i, 1] for i, l in enumerate(labels) if l == label]
        plt.plot(x, y, 'o', label=label, alpha=0.6)
    plt.grid()
    plt.legend()
