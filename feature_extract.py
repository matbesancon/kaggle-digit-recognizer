# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 17:09:01 2017

@author: mbesancon
"""

import numpy as np
import matplotlib.pyplot as plt
from exploratory import matrix, resquare, viz_pixel
from skimage.morphology import skeletonize, medial_axis


def binarize(array, threshold=200):
    """
    Sets all pixels in array to 255 if >= threshold, 0 otherwise
    """
    return np.array([255 * (e > threshold) for e in array])


def skeleton_image(img_array, median=False, dim=28):
    """
    Thinners the foreground of an image to one pixel
    """
    img = resquare(binarize(img_array)//255, dim)
    if median:
        return medial_axis(img).flatten()
    return skeletonize(img).flatten()


def compare_res(img_array, dim=28):
    """
    Plot results of raw image, binarize, skeleton_image
    with and without median axis
    """
    for res_tup in zip([img_array, binarize(img_array),
                        skeleton_image(img_array),
                        skeleton_image(img_array, median=True)],
                       ["Raw", "Binarized", "Skeleton", "Median axis"]):
        plt.figure()
        viz_pixel(res_tup[0], dim)
        plt.title(res_tup[1])
    plt.show()

if __name__ == "__main__":
    # random index generator
    index = (np.random.randint(0, matrix.shape[0]) for _ in range(5))
    for idx in index:
        compare_res(matrix[idx, :])
