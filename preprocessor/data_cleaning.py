#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time    : 2017/11/21 12:11
# @Author  : HeHonXue

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


# 离散数据one-hot-encode处理
def one_hot_encoder(data, discrete_cols):
    """
    对离散变量进行one-hot编码
    :param data: pd.DataFrame
    :param discrete_cols: 
    :return: 变换后的data，pd.DataFrame
    """
    if not discrete_cols:
        return data
    assert isinstance(discrete_cols, list)
    assert isinstance(data, pd.core.frame.DataFrame)

    current_cols = set(data.columns)
    for column in discrete_cols:
        if column == "is_yq":
            continue
        if column not in current_cols:
            continue
        print("\tDiscretize the feature:【{}】...".format(column))
        tmp = pd.get_dummies(data[column], prefix=column)
        data = pd.concat([data, tmp], axis=1)
        del data[column]
    print("Features discretization task completed!")
    return data


# 缺失值处理
def missing_value_processor(data):
    """
    对数值型缺失值NaN处理
    :param data: pd.DataFrame
    :return: 变换后的data，pd.DataFrame
    """
    from sklearn.preprocessing import Imputer
    assert isinstance(data, pd.core.frame.DataFrame)
    print("Missing value processing...")
    data.select_dtypes(include=[np.number]).isnull().sum().sort_values(ascending=False)
    sub_columns = data.select_dtypes(include=[np.number]).columns
    imr = Imputer(missing_values='NaN', strategy='median', axis=0)
    imr = imr.fit(data[sub_columns])
    data[sub_columns] = imr.transform(data[sub_columns])
    return data


# 数据变形
def data_transform(data):
    """
    将DataFrame中的数值转成float，并进行相关特征离散化
    :param data: 
    :return: 
    """
    assert isinstance(data, pd.core.frame.DataFrame)
    print("Data transforming...")
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    # 数据清洗-连续型数据分箱
    train_data = data.replace(["", "\s"], np.nan, regex=True)
    train_data = train_data.apply(lambda x: pd.to_numeric(x, errors='ignore', downcast='float'))

    # 借款期限处理
    train_data.loc[train_data.iloanperiod == 20, 'iloanperiod'] = 1
    train_data.loc[train_data.iloanperiod == 30, 'iloanperiod'] = 2
    train_data.loc[train_data.iloanperiod == 45, 'iloanperiod'] = 3
    # 年龄分段
    train_data.loc[train_data.sage <= 23, 'sage'] = 1
    train_data.loc[(train_data.sage > 23) & (train_data.sage <= 30), 'sage'] = 2
    train_data.loc[(train_data.sage > 30) & (train_data.sage <= 36), 'sage'] = 3
    train_data.loc[train_data.sage > 36, 'sage'] = 4
    # 下载渠道
    train_data.loc[train_data.channel_nm == 'tengxun', 'is_gw_qd'] = 1
    train_data.loc[train_data.channel_nm != 'tengxun', 'is_gw_qd'] = 0

    # 版本号
    train_data.loc[(train_data.version_num == 7) | (train_data.version_num == 10), 'is_gw_version'] = 1
    train_data.loc[(train_data.version_num != 7) & (train_data.version_num != 10), 'is_gw_version'] = 0
    # 芝麻身份证风险等级
    train_data.loc[train_data.zhima_idno_risk_level <= 1, 'is_gw_zhima'] = 1
    train_data.loc[train_data.zhima_idno_risk_level > 1, 'is_gw_zhima'] = 0

    del train_data['channel_nm']
    del train_data['version_num']
    del train_data['sloanapplyid']
    del train_data['zhima_idno_risk_level']
    return train_data


# 数据归一化
def data_normalization(data, scale_type="max_min", except_cols=None):
    """
    最大最小值归一
    :return: 
    """
    assert isinstance(data, pd.core.frame.DataFrame)
    print("Normalizing...")
    if except_cols:
        if "is_yq" not in except_cols:
            except_cols.append("is_yq")
        selected_cols = [x for x in data.columns if x not in except_cols]
    else:
        selected_cols = data.columns
    if scale_type == "max_min":
        from sklearn.preprocessing import minmax_scale
        data[selected_cols] = data[selected_cols].apply(minmax_scale, axis=0)
    elif scale_type == "z_score":
        from sklearn.preprocessing import scale
        data[selected_cols] = data[selected_cols].apply(scale, axis=0)
    elif scale_type == "normalize":
        data[selected_cols] = data[selected_cols].apply(normalize, axis=0)
    return data


def normalize(arr):
    """
    将每个样本缩放到单位范数, 配合data_normalization方法使用
    :param arr: np.array或者list
    :return: 
    """
    from sklearn.preprocessing import normalize
    x = np.reshape(np.array(arr), (1, len(arr)))
    x = normalize(x)
    return x.tolist()[0]


# 特征选择，随机森林
def features_select_rf(x_data, y_data, top_n=1.0,
                       n_estimators=100, random_state=0, n_jobs=-1):
    print("Features selecting...")
    from sklearn.ensemble import RandomForestClassifier
    assert isinstance(x_data, pd.core.frame.DataFrame)
    assert isinstance(y_data, pd.core.series.Series)

    feat_labels = x_data.columns
    x_mat = x_data.as_matrix().astype(np.float32)
    y_vec = y_data.tolist()
    # print(feat_labels)
    # print(np.any(np.isnan(x_mat)))
    # print(np.all(np.isfinite(x_mat)))

    forest = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
    forest.fit(x_mat, y_vec)

    importance = forest.feature_importances_
    indices = np.argsort(importance)[::-1]

    features_ranking = list(zip(feat_labels, indices))
    features_ranking = sorted(features_ranking, key=lambda k: k[1], reverse=True)

    if isinstance(top_n, float):
        assert 0 <= top_n <= 1
        top_n = int(top_n * len(features_ranking))
    elif isinstance(top_n, int):
        top_n = min(len(features_ranking), top_n)
    else:
        top_n = len(features_ranking)
    return [f[0] for f in features_ranking][: top_n]


# 综合函数
def data_clean(data_file='../test_data/train.csv', row_limit=300):
    import os
    assert os.path.exists(data_file)
    from config_file.features_config import discrete_columns

    train_data = pd.read_csv(open(data_file, 'r'))[:row_limit]
    if row_limit:
        assert isinstance(row_limit, int)
        train_data = train_data[:row_limit]

    train_data = data_normalization(
            missing_value_processor(
                data_transform(train_data)
            ), except_cols=discrete_columns
        )
    return train_data.drop('is_yq', 1), pd.to_numeric(train_data['is_yq'], downcast='signed')

if __name__ == "__main__":
    x_data1, y_data1 = data_clean()
    print(features_select_rf(x_data1, y_data1, 20))


