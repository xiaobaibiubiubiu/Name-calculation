#!/usr/local/bin/python
# -*- coding: utf-8 -*-
"""
@Time: 2020/03/31 11:00
@Function：发票贷模型V1.0
@Version: V1.0
"""
import pandas as pd
import numpy as np
import time
import os
import codecs
import xgboost
import time
import datetime
import sklearn2pmml
from sklearn.externals import joblib
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from dateutil.parser import parse


# ks计算函数
def compute_ks(target, proba):
    from scipy.stats import ks_2samp
    return ks_2samp(proba[target == 1], proba[target != 1]).statistic

# 模型函数
def run_model(df_main):
    cate_feat = []
    #n_estimators: 拟合的树的棵树，相当于训练轮数
    #learning_rate#学习率
    #max_depth=-1#最大树的深度
    #num_leaves, default=31, type=int, alias=num_leaf  一棵树上的叶子数
    #is_unbalance 默认是ture 不平衡数据集
    #categorical_feature, default="", type=string, alias=categorical_column, cat_feature, cat_column
        # 指定分类特征
        # 用数字做索引, e.g. categorical_feature=0,1,2 意味着 column_0, column_1 和 column_2 是分类特征
        # 为列名添加前缀 name:, e.g. categorical_feature=name:c1,c2,c3 意味着 c1, c2 和 c3 是分类特征
        # Note: 只支持分类与 int type. 索引从 0 开始. 同时它不包括标签栏
        # Note: 负值的值将被视为 missing values
    #subsample_for_bin,  bin_construct_sample_cnt, 默认为200000, 也称subsample_for_bin。用来构建直方图的数据的数量。
    #n_jobs=-1 全部CPU工作
    #colsample_bytree colsample_bytree：系统默认值为1。我们一般设置成0.8左右。用来控制每棵随机采样的列数的占比(每一列是一个特征)。 典型值：0.5-1范围: (0,1]
    # min_child_weight 定义了一个子集的所有观察值的最小权重和。这个可以用来减少过拟合，但是过高的值也会导致欠拟合，因此可以通过CV来调整min_child_weight。
    #min_split_gain, 默认为0, type=double, 也称min_gain_to_split`。执行切分的最小增益 
    #verbose=-1 不打印过程
    #silent 提升时打印
    clf = LGBMClassifier(
        n_estimators=50, learning_rate=0.05, max_depth=5, num_leaves=12, is_unbalance=True, categorical_feature=cate_feat,
        subsample_for_bin=100, n_jobs=-1, colsample_bytree=0.3, min_child_weight=5, min_split_gain=2, verbose=-1, random_state=0, silent=False)
    df_train, df_test, y_train, y_test = train_test_split(df_main, df_main["type"], test_size=0.3, random_state=0)  # 生成测试集训练集及相应标签
    print("train %s, test %s" % (df_train.shape, df_test.shape))
    # 模型训练
    clf.fit(df_train[var_list], df_train["type"])
    # 输出模型特征重要度
    df_importance = pd.DataFrame({'var_name': var_list})
    #输出各个特征的重要性
    df_importance['var_importance'] = clf.feature_importances_
    #重要性的排序降序
    df_importance = df_importance.sort_values(by=['var_importance'], ascending=False)

    df_importance[['var_importance', 'var_name']].to_csv(os.path.join('/Users/apple/Desktop', 'var_importance_%s.csv' % today), encoding='utf_8_sig', index=None)

    pvalue = clf.predict_proba(df_main[var_list])[:, -1]  # 模型预测值
    df_main['pvalue'] = pvalue
    df_main.to_csv(os.path.join('/Users/apple/Desktop', 'pvalue_%s.csv' % today), encoding='gbk', index=None)
    # 输出模型pmml文件
    pipeline = sklearn2pmml.PMMLPipeline([('classifier', clf)])
    pipeline.fit(df_train[var_list], df_train["type"])
    sklearn2pmml.sklearn2pmml(pipeline, os.path.join('/Users/apple/Desktop', 'lgbm_%s.pmml' % today))
    # 计算模型在训练集和测试集的KS和auc
    print("train auc: %s" % roc_auc_score(y_train, clf.predict_proba(df_train[var_list])[:, -1]))
    print("test auc: %s" % roc_auc_score(y_test, clf.predict_proba(df_test[var_list])[:, -1]))
    print("train ks: %s" % compute_ks(y_train, clf.predict_proba(df_train[var_list])[:, -1]))
    print("test ks: %s" % compute_ks(y_test, clf.predict_proba(df_test[var_list])[:, -1]))


if __name__ == '__main__':
    #读取数据集
    df_main = pd.read_csv('/Users/apple/Desktop/sample1011_f.csv',encoding='gbk')
    #获取今天的时间戳
    today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    #100003000001  近12个月开票金额	valid_amt_y	近12个月有效开票金额，有效开票金额=蓝票金额+红票金额	取数截止申请日期上月月末
    #100003000051	近12个月开票总张数	l12_count	近12个月蓝票张数+红票张数总和（即月度统计表中sum(count)）	取数截止申请日期上月月末
    #100005000003	近12个月开票月份数	l12_monthcount	近12个月有开票的月份总数	取数截止申请日期上月月末
    #100003000061	近12个月红冲票张数	l12_hc_count		取数截止申请日期上月月末
    #100003000011	近12个月红冲票总额	l12_hc_amount	取红冲金额绝对值	取数截止申请日期上月月末
    #100004001003	近12个月红冲发票金额/近12个月蓝票金额		取红冲金额的绝对值计算，即近12个月红冲发票金额/近12个月蓝票金额	取数截止申请日期上月月末
    #100003000071	近12个月作废票张数	l12_invalid_count		取数截止申请日期上月月末
    #100004000023	近12个月作废票金额占比	l12_invalid_amount_rate	近12个月作废发票金额/(近12个月作废发票金额+近12个月有效开票金额）	取数截止申请日期上月月末
    #100004000003	近12个月有效票金额增长率	growth_rate_valid_amt_y2y	（近12个月有效开票金额-近13至24个月有效开票金额）/近13至24个月有效开票金额	取数截止申请日期上月月末
    #100004000024	近24个月作废票金额占比	l24_invalid_amount_rate	近24个月作废发票金额/(近24个月作废发票金额+近24个月有效开票金额）	取数截止申请日期上月月末
    #100004000052	近6个月有效开票金额环比	ratio_valid_amt_m2m_hy2hy	近6个月有效开票金额/近7至12个月有效开票金额	取数截止申请日期上月月末
    var_list = ['100003000001', '100003000051', '100005000003', '100003000061', '100003000011',
                '100004001003', '100003000071', '100004000023', '100004000003', '100004000024',
                '100004000052']
    run_model(df_main)