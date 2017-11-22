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
        self.user_id = None
        self._clipped_data = None
        self._clipped_data_one_hot = None

    def preprocessor(self, row_limit=None):
        self.x_data, self.y_data, self.user_id = data_clean(
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

    def reorganize_data(self, data_file,
                        row_limit=None,
                        for_predict=False,
                        one_hot_encode=True,
                        discrete_cols=discrete_columns):
        """
        用于检验、预测。将新数据转换成与训练数据格式一致
        如果 for_predict=False， 代码中的tmp_value指代 (sloanapplyid, 因变量)；反之 只代表 sloanapplyid
        :param data_file: 
        :param row_limit: 
        :param for_predict: 新数据用于检验，False; 用于预测（需要打上标签），True
        :param one_hot_encode: 
        :param discrete_cols: 
        :return: 
        """
        if for_predict:
            new_data, user_id = data_clean(data_file,
                                           row_limit=row_limit,
                                           for_predict=for_predict)
            tmp_value = user_id
        else:
            new_data, y_true, user_id = data_clean(data_file,
                                                   row_limit=row_limit,
                                                   for_predict=for_predict)
            tmp_value = list(zip(user_id, y_true))
        model_feats = set(self.clipped_data.columns)
        current_feats = new_data.columns
        for mf in model_feats:
            if mf not in current_feats:
                new_data[mf] = 0
        new_data = new_data[self.clipped_data.columns]
        if one_hot_encode:
            new_data = one_hot_encoder(
                new_data,
                discrete_cols=discrete_cols
            )
            model_feats = set(self.clipped_data_one_hot.columns)
            current_feats = new_data.columns
            for mf in model_feats:
                if mf not in current_feats:
                    new_data[mf] = 0
            new_data = new_data[self.clipped_data_one_hot.columns]
        return new_data.as_matrix().astype(np.float32), tmp_value

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
