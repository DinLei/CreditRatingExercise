#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2017/11/22 11:51
# @Author  : HeHonXue

import numpy as np


def predict(data_file,
            row_limit=None,
            for_predict=False,
            one_hot_encode=True):
    from app_utils.io_util import load_pickle
    from preprocessor.data_helper import DataHelper
    dh_er = load_pickle("./model/clean_data.obj")
    model = load_pickle("./model/exceed_time.judge")
    assert isinstance(dh_er, DataHelper)
    target_data, user_id = dh_er.reorganize_data(
        data_file=data_file,
        row_limit=row_limit,
        for_predict=for_predict,
        one_hot_encode=one_hot_encode
    )
    outcome = model.predict(target_data)
    return list(zip(user_id, outcome))


def predict_report(self_outcome):
    from sklearn.metrics import classification_report
    y_test = []
    y_predict = []
    for ele in self_outcome:
        if isinstance(ele[0], tuple):
            y_test.append(ele[0][1])
        else:
            y_test.append(ele[0])
        y_predict.append(ele[1])
    print(classification_report(y_test, y_predict))
    return np.array(y_test), np.array(y_predict)


def test_index(self_outcome):
    y_test, y_predict = predict_report(self_outcome)
    total = len(y_predict)
    in_time_predict = sum(y_predict==0)
    false_true = sum((y_predict==0) & (y_test==1))

    pass_rate = (in_time_predict/total) * 100
    exceed_time_rate = (false_true/in_time_predict) * 100
    print("通过率: {:.2f}% ; 逾期率: {:.2f}%".format(pass_rate, exceed_time_rate))


if __name__ == "__main__":
    test_outcome = predict("./test_data/train.csv", row_limit=None)
    test_index(test_outcome)

