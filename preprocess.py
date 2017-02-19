# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 20:57:56 2017

@author: mbesancon
"""

import pandas as _pd

raw_data = _pd.read_csv("data/train.csv")
matrix = raw_data.iloc[:, 1:].values
labels = raw_data["label"]
matrix_test = _pd.read_csv("data/test.csv").values

if __name__ == "__main__":
    raw_data.iloc[:, 1:].to_csv("data/matrix.csv", index = False, header = False)
    labels.to_csv("data/labels.csv", index = False, header = False)
