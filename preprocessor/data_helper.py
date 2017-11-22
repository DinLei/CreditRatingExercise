#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2017/11/21 12:11
# @Author  : HeHonXue

import numpy as np
from preprocessor.data_cleaning import data_clean


class DataHelper:
    """
    是否逾期数据预处理
    """
    def __init__(self, data_file):
        self.data_file = data_file
        self.features = None
        self.x_matrix = None
        self.y_vector = None
        self.x_data = None
        self.y_data = None

    def preprocessor(self, row_limit=None):
        self.x_data, self.y_data = data_clean(
            self.data_file, row_limit=row_limit)
        self.features = self.x_data.columns

    def prepare_training_data(self, selected_features=None):
        if selected_features:
            x_tr = self.x_data[selected_features]
        else:
            x_tr = self.x_data
        self.x_matrix = x_tr.as_matrix().astype(np.float32)
        self.y_vector = self.y_data.tolist()


def data_splits(x_matrix, y_vector,
                test_size=0.3,
                random_state=0,
                shuffle=True):
    from sklearn.model_selection import train_test_split
    return train_test_split(
        x_matrix, y_vector,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
    )
