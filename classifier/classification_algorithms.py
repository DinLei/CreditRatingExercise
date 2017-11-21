#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2017/10/27 14:57
# @Author  : BigDin
# @Contact : dinglei_1107@outlook.com

import re


# neural network Classifier
def neural_network_classifier(train_x, train_y):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(50,), alpha=1e-4,
                          solver='sgd', verbose=10, tol=1e-4, random_state=1,
                          learning_rate_init=.1)
    model.fit(train_x, train_y)
    return model


# Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y, model_type="m"):
    """
    
    :param train_x: 自变量矩阵
    :param train_y: 因变量
    :param model_type: "b"代表伯努利事件模型，"m"代表多项式事件模型
    :return: 
    """
    assert model_type in {"m", "b"}
    if model_type == "m":
        from sklearn.naive_bayes import MultinomialNB as NbModel
    else:
        from sklearn.naive_bayes import BernoulliNB as NbModel
    model = NbModel(alpha=0.01)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y,
                             n_estimators=110, max_features='sqrt', max_depth=None,
                             min_samples_split=4, min_samples_leaf=1,
                             max_leaf_nodes=None, random_state=50, n_jobs=4):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth,
                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                   max_leaf_nodes=max_leaf_nodes, random_state=random_state, n_jobs=n_jobs)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in list(best_parameters.items()):
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


class Classifier:
    classifiers = {'NB': naive_bayes_classifier,
                   'KNN': knn_classifier,
                   'LR': logistic_regression_classifier,
                   'RF': random_forest_classifier,
                   'DT': decision_tree_classifier,
                   'SVM': svm_classifier,
                   'SVMCV': svm_cross_validation,
                   'GBDT': gradient_boosting_classifier,
                   'ANN': neural_network_classifier
                   }

    def __init__(self):
        print(list(Classifier.classifiers.keys()))

    @staticmethod
    def get_classifier(classifier_name):
        name = classifier_name.upper()
        name = re.sub("\s", "", name)
        return Classifier.classifiers[name]


if __name__ == '__main__':
    Classifier()
