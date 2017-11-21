#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2017/11/21 18:28
# @Author  : HeHonXue


if __name__ == "__main__":

    from preprocessor.data_helper import *
    from sklearn.metrics import classification_report
    from classifier.classification_algorithms import *
    from preprocessor.data_cleaning import features_select_rf

    test_file = "./test_data/train.csv"

    dh_er = DataHelper(data_file=test_file)
    dh_er.preprocessor()
    selected_feats = features_select_rf(
        x_data=dh_er.x_data, y_data=dh_er.y_data, top_n=0.7)
    dh_er.prepare_training_data(selected_features=selected_feats)

    x_max, y_vec = dh_er.x_matrix, dh_er.y_vector

    x_train, x_dec, y_train, y_dev = data_splits(
        x_matrix=x_max, y_vector=y_vec, test_size=0.2)

    lr_model = logistic_regression_classifier(x_train, y_train)
    y_pred = lr_model.predict(x_dec)

    print("### logistic classifier | test | outcome ###")
    print(classification_report(y_dev, y_pred))

