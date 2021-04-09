# -*- coding:UTF-8 -*-
import joblib
import pandas as pd
from sklearn import cluster, datasets

from readfile import *


def t01():
    iris = datasets.load_iris()
    X_iris = iris.data
    y_iris = iris.target
    k_means = cluster.KMeans(n_clusters=3)
    k_means.fit(X_iris)
    print(k_means.labels_[::10])


tkword = ['第', '章', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']


def transformLine(line):
    x = line[0].strip()
    line[1] = 0
    if len(line[0].strip()) > 20 or len(line[0].strip()) < 3:
        line[0] = 0
    else:
        f1 = 0
        f2 = 0
        i = 0
        for tk in tkword:
            if x.find(tk) > -1:
                f1 = f1 + 8
                if i < 2:
                    f2 = f2 + i
            i = i + 1
        if 24 <= f1 <= 96:
            f1 = 1
        else:
            f1 = 0
        line[0] = f1
        line[1] = f2


def tox(x):
    if len(x) > 20 or len(x) < 3:
        return 0
    else:
        return 1


def getArticleFeature():
    lines = read("./resource/article.txt")
    x_alias = [transformLine(x) for x in lines]

    data = pd.DataFrame(lines, columns=["feature1"])
    data['feature2'] = data['feature1']
    data.apply(transformLine, axis=1)
    data['valid'] = [tox(x) for x in lines]
    return data, lines


def train():
    data, lines = getArticleFeature()
    k_means = cluster.KMeans(n_clusters=3)
    k_means.fit(data)
    joblib.dump(k_means, 'saved_model/article.pkl')
    data["article"] = lines
    data["predicat"] = k_means.labels_
    d2 = data.loc[data.predicat == 1]
    y = 1


def use():
    k_means = joblib.load('saved_model/article.pkl')
    data, lines = getArticleFeature()
    k_means.predict(data)
    data["article"] = lines
    data["predicat"] = k_means.labels_
    d2 = data.loc[data.predicat == 1]
    y = 1


if __name__ == "__main__":
    use()
