"""
 -*- coding: utf-8 -*-
time: 2023/12/20 17:02
author: suyunsen
email: suyunsen2023@gmail.com
"""


import numpy as np


def preprocess_data(X, y):
    # 数据清洗
    # TODO: 处理缺失值和异常值

    # 特征选择
    # TODO: 根据需求选择合适的特征

    # 数据标准化
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    return X, y


def initialize_parameters(n_features):
    # 使用零来初始化参数
    w = np.zeros((n_features, 1))
    b = 0

    return w, b


# 计算代价函数和梯度
def compute_cost_gradient(X, y, w, b):
    m = X.shape[0]

    # 计算预测值
    z = np.dot(X, w) + b

    a = np.sigmoid(z)

    # 计算代价函数
    #交叉商损失函数 -[sum(y_t*log(y_p) + (1-y_t)*log(1-y_p))]
    cost = -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / m

    # 计算梯度
    dw = np.dot(X.T, (a - y)) / m
    db = np.sum(a - y) / m

    return cost, dw, db


# 更新参数
def update_parameters(w, b, dw, db, learning_rate):
    w = w - learning_rate * dw
    b = b - learning_rate * db

    return w, b


# 迭代优化
def optimize(X, y, w, b, num_iterations, learning_rate):
    costs = []

    for i in range(num_iterations):
        cost, dw, db = compute_cost_gradient(X, y, w, b)
        w, b = update_parameters(w, b, dw, db, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            print("Cost after iteration %i: %f" % (i, cost))

    return w, b, costs


def predict(X, w, b):
    z = np.dot(X, w) + b
    a = np.sigmoid(z)
    y_pred = (a > 0.5).astype(int)

    return y_pred


# 完整的逻辑回归模型
class LogisticRegression:
    def __init__(self, num_iterations=1000, learning_rate=0.01):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.w = None
        self.b = None

    def fit(self, X, y):
        X, y = preprocess_data(X, y)
        n_features = X.shape[1]
        self.w, self.b = initialize_parameters(n_features)
        self.w, self.b, costs = optimize(X, y, self.w, self.b, self.num_iterations, self.learning_rate)
        return costs

    def predict(self, X):
        X, _ = preprocess_data(X, np.zeros(X.shape[0]))
        y_pred = predict(X, self.w, self.b)
        return y_pred


if __name__ == '__main__':
    print(initialize_parameters(5))

