# Kaggle digit recognizer challenge

This repo contains some experiments with the
[digit-recognizer challenge](https://www.kaggle.com/c/digit-recognizer) from Kaggle.

## Objective

Given flattened 28x28 grey-scale images of hand-drawn figures, the
goal is to predict the real figure.  

Personal objectives include exploring image feature extraction
techniques, define models for figures within images and test
non-linear transformations on prediction performance.

## Content

* *exploratory.py* - first visualizations from raw data
downloaded from Kaggle, separation from the pixel matrix and
labels. Not much cleaning and pre-processing is required to start
playing around.
* *knn.py* - test of the K nearest neighbors scikit-learn
implementation, reaching ~97% accuracy, with and without
Principal Components Analysis as pre-processing.
* *feature_extract.py* - image processing techniques to make the
figures easier to distinguish.
