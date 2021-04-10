# -*- coding:UTF-8 -*-
import math
import multiprocessing
# '''文章关键字抽取'''
import os
from time import time

import gensim
import jieba.posseg as pseg
from collections import Counter
import jieba
from jieba.analyse import tfidf
from pyecharts import options as opts
from pyecharts.charts import WordCloud
from gensim import corpora
from gensim import models
import pandas as pd

rmcx = ['v', 'vd', 'vn', 'vshi', 'vyou', 'vf', 'vx', 'vi', 'vg',
        'vl', 'a', 'ad', 'an', 'ag', 'al', 'z', 'r', 'rr', 'rz', 'rzt',
        'rzs', 'rzv', 'ry', 'ryt', 'rys', 'ryv', 'rg', 'm', 'mq',
        'q', 'qv', 'qt', 'd', 'u', 'uzhe', 'ule', 'uguo', 'ude1', 'ude2', 'ude3',
        'usuo', 'udeng', 'uyy', 'udh', 'uls', 'uzhi', 'ulian', 'e', 'y', 'o', 'wm',
        ]

jieba.load_userdict("./resource/jieba.txt")


def transformLine(line):
    words = pseg.lcut(line)
    s = []
    for word, flag in words:
        if len(word) > 1:
            if flag not in rmcx:
                s.append(word)
    return s


# class MyCorpus(object):
#     def __init__(self):
#         self.dictionary = corpora.Dictionary(texts)
#     def __iter__(self):
#         for line in open('./resource/article.txt'):
#             yield self.dictionary.doc2bow(transformLine(line))


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for root, dirs, files in os.walk(self.dirname):
            for filename in files:
                file_path = root + '/' + filename
                for line in open(file_path):
                    try:
                        sline = line.strip()
                        if sline == "":
                            continue
                        ###rline = cleanhtml(sline)
                        tokenized_line = ''.join(sline)
                        ###print(tokenized_line)
                        word_line = transformLine(tokenized_line)
                        yield word_line
                    except Exception:
                        print("catch exception")
                        yield ""


def stopwordslist():
    stopwords = [line.strip() for line in
                 open('./resource/stopwords/baidu_stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords


def k01():
    begin = time()
    stopwords = stopwordslist()
    sentences = MySentences("./resource/a1")
    wordsCounter = Counter()
    for s in sentences:
        wp = []
        for word in s:
            if word not in stopwords:
                wp.append(word)
        wordsCounter.update(wp)
    wd = dict(wordsCounter)
    print(len(wd))
    wdc = {}
    for key in wd:
        if wd[key] > 100:
            wdc[key] = wd[key]
    wordsCounter = Counter(wdc)
    wordslist = wordsCounter.most_common(1000)
    wordcloud = WordCloud()
    wordcloud.add('', wordslist, shape='circle')
    ### 渲染图片
    # wordcloud.to_file("./resource/cy.png")
    # wordcloud.show()
    wordcloud.render()
    ### 指定渲染图片存放的路径
    ### mywordcloud.render('E:/wordcloud.html')

    # model = gensim.models.Word2Vec(sentences,window=15, min_count=10, workers=multiprocessing.cpu_count())
    # model.save("model/word2vec_gensim")
    # model.wv.save_word2vec_format("model/word2vec_org",
    #                               "model/vocabulary",
    #                               binary=False)

    end = time()
    print("Total procesing time: %d seconds" % (end - begin))
    # lines = read("./resource/article.txt")
    # texts = [transformLine(x) for x in lines]
    # dictionary = corpora.Dictionary(texts)
    # corpus = [dictionary.doc2bow(text) for text in texts]
    # tfidf = models.TfidfModel(corpus)


# word可以通过count得到，count可以通过countlist得到

# count[word]可以得到每个单词的词频， sum(count.values())得到整个句子的单词总数
def gtf(word, count):
    return count[word] / sum(count.values())


# 统计的是含有该单词的句子数
def gn_containing(word, article):
    return sum(1 for sentence in article if word in sentence)


# len(count_list)是指句子的总数，n_containing(word, count_list)是指含有该单词的句子的总数，加1是为了防止分母为0
def gidf(word, count_list):
    return math.log(len(count_list) / (1 + gn_containing(word, count_list)))


# 将tf和idf相乘
def gtfidf(word, count, count_list):
    f = gidf(word, count_list)
    return gtf(word, count) * f


if __name__ == '__main__':
    begin = time()
    stopwords = stopwordslist()
    article = []
    sentences = MySentences("./resource/a1")
    wordsCounter = Counter()
    for s in sentences:
        wp = []
        article.append(s)
        for word in s:
            if word not in stopwords:
                wp.append(word)
        wordsCounter.update(wp)
    wd = dict(wordsCounter)
    t1 = pd.DataFrame(columns=["kw", "count", "tfidf"])
    t1['kw'] = list(wd.keys())
    t1['count'] = list(wd.values())
    g = []
    for word in wd:
        g.append(gtfidf(word, wordsCounter, article))
    t1['tfidf'] = g
    t1 = t1.sort_values(by=['tfidf'], ascending=[False])
    t1 = t1.head(20)
    subjects = {}
    for index, row in t1.iterrows():
        print("文章主题 %s %s" % (row['kw'],row['count']))
        subjects[row['kw']] = row['count']
    # del t1['tfidf']
    wordsCounter = Counter(subjects)
    wordcloud = WordCloud()
    wordslist = wordsCounter.most_common(1000)
    wordcloud.add('', wordslist, shape='star')
    wordcloud.render()
    # temp = t1.head(300)
    # del temp['tfidf']
    print(t1)
