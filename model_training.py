#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2017/11/21 18:28
# @Author  : HeHonXue


if __name__ == "__main__":

    from preprocessor.data_helper import *
    from sklearn.metrics import classification_report
    from classifier.classification_algorithms import *
    from app_utils.io_util import save_as_pickle
    from preprocessor.data_cleaning import features_select_rf

    test_file = "./test_data/train.csv"

    dh_er = DataHelper(data_file=test_file)
    dh_er.preprocessor(row_limit=None)
    # dh_er = load_pickle("./model/clean_data.obj")
    selected_feats = features_select_rf(
        x_data=dh_er.x_data, y_data=dh_er.y_data, top_n=0.8)

    dh_er.prepare_training_data(
        one_hot_encode=True,
        selected_features=selected_feats
    )

    x_max, y_vec = dh_er.x_matrix, dh_er.y_vector

    x_train, x_dec, y_train, y_dev = data_splits(
        x_matrix=x_max, y_vector=y_vec, test_size=0.2)

    # 模型测试
    lr_model = logistic_regression_classifier(x_train, y_train)
    y_predict3 = lr_model.predict(x_dec)

    print("### lr classifier | test | outcome ###")
    print(classification_report(y_dev, y_predict3))

    save_as_pickle(lr_model, "exceed_time.judge", "./model")
    save_as_pickle(dh_er, "clean_data.obj", "./model")

