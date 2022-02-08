import jieba
import jieba.analyse
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

def transform(data):
    x = []
    y = []
    for row in data:
        x.append(row[0])
        y.append(row[1])
    return [x,y]

def draw(data):
    print("Drawing Scatter Plot...")
    plt.suptitle('PCA of LSI')
    plt.scatter(data[0], data[1])
    plt.savefig("PCA of LSI.png")
    plt.show()

def P_C_A(data):
    print("PCAing...")
    pca = PCA(n_components=2)
    pca.fit(data)
    draw(transform(pca.fit_transform(data)))

def Lsi(keywords, name_list):
    print("LSIing...")
    svd = TruncatedSVD(n_components=100)
    norm = Normalizer(copy=False)
    lsi = make_pipeline(svd, norm)
    all_topic_vector = lsi.fit_transform(keywords)
    P_C_A(np.array(all_topic_vector, dtype=object))
    
def tfidf(data, names):
    print("TF*iDFing...")
    vec = TfidfVectorizer(ngram_range=(1, 1), use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words=None)
    keywords = vec.fit_transform(data)
    Lsi(keywords, names)

def read_file(file_name):
    print("Reading files...")
    data = list()
    with open(file_name, encoding='utf-8') as f:
        reader = list(csv.reader(f)) 
        length = len(reader[0])
        for i in range(length):
            data.append("")
        for column in reader[1:]:
            for i in range(1, length):
                data[i] += str(column[i]) + " "
    tfidf(data[1:], reader[0][1:])


def main():
    read_file(r'D:\Shaw\Documents\Miscellany\网课\R计划数据挖掘方向\课程\Homework 1\tf table_Ch.csv')
    print("Finish!")

main()