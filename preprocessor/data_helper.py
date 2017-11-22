#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2017/11/21 12:11
# @Author  : HeHonXue

import numpy as np
from config_file.features_config import discrete_columns
from preprocessor.data_cleaning import data_clean, one_hot_encoder


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
        self._clipped_data = None
        self._clipped_data_one_hot = None

    def preprocessor(self, row_limit=None):
        self.x_data, self.y_data = data_clean(
            self.data_file, row_limit=row_limit)
        self.features = self.x_data.columns

    def prepare_training_data(self,
                              one_hot_encode=True,
                              selected_features=None,
                              discrete_cols=discrete_columns):
        if selected_features:
            self._clipped_data = self.x_data[selected_features]
        else:
            self._clipped_data = self.x_data
        if one_hot_encode:
            x_tr = one_hot_encoder(self._clipped_data, discrete_cols=discrete_cols)
            self._clipped_data_one_hot = x_tr
        else:
            x_tr = self._clipped_data
        self.x_matrix = x_tr.as_matrix().astype(np.float32)
        self.y_vector = [int(x) for x in self.y_data.tolist()]

    @property
    def clipped_data(self):
        return self._clipped_data

    @property
    def clipped_data_one_hot(self):
        return self._clipped_data_one_hot


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
