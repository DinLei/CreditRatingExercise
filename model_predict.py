#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2017/11/22 11:51
# @Author  : HeHonXue


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


if __name__ == "__main__":
    test_outcome = predict("./test_data/train.csv", row_limit=None)
    predict_report(test_outcome)
