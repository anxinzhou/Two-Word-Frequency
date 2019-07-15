import pickle
import os
import time
import re
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz
import gmpy
import numpy as np
from math import sqrt
from numpy import linalg as LA
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# use to check if is a word
# import enchant
# from nltk.stem.porter import PorterStemmer
# porter_stemmer = PorterStemmer()
# enDict = enchant.Dict('en_US')


rootDir = "maildir"
targetFiles = "target.data"
contentFile = "content.data"
dictionaryPath = "dictionary.data"
prunedDictionaryPath = "prunedDictionary.data"

vectorPath = "countVector.data"
prunedVectorPath = "prunedCountVector.data"
oneWordPath = "oneWord.data"
prunedOneWordPath = "prunedOneWord.data"
twoWordPath = "twoWord.data.npy"
reverseIndexPath = "reverseIndex.data"

calibratedBitMapPath = "calibratedVector.data"
wordIDVectorPath = "wordIDVector.npy"
kmeansLabelPath = "kmeansLabel.npy"
pcaPath = "pca.npy"

egPath = "eg.data"

stop_words = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
              'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
              'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the',
              'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
              'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both',
              'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and',
              'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',
              'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too',
              'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my',
              'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}
fileTargetAmount = 100000
wordTargetAmount = 5000
sampleCount = 200
clusterNum = 32

prob0To1 = 0.3
prob1To0 = 0.1


def merge_cluster(ts, ws, index):
    new_ts = []
    new_ws = []
    base = ts[index]
    base_ws = ws[index]
    for i, t in enumerate(ts):
        if i == index:
            continue
        r = np.sum(np.logical_and(base, t))
        if r == 0:
            new_ts.append(t)
            new_ws.append(ws[i])
        else:
            base = np.logical_xor(base, t)
            base_ws.extend(ws[i])
    new_ws.append(base_ws)
    new_ts.append(base)
    ts[:] = new_ts
    ws[:] = new_ws


def cluster_reverse_index(word_frequency, reverse_index):
    s = time.time()
    ts = []
    ws = []
    for i, w in enumerate(reverse_index):
        merged = False
        for j, t in enumerate(ts):
            r = np.sum(np.logical_and(t, w))
            if r != 0:
                ts[j] = np.logical_xor(ts[j], w)
                ws[j].append(word_frequency[i])
                merge_cluster(ts, ws, j)
                merged = True
                break
        if not merged:  # a new cluster
            ts.append(w)
            ws.append([word_frequency[i]])
    e = time.time()
    print("cluster time:", e - s, "s")
    return ws


def test_two_word():
    # get files from path
    files = get_files(rootDir, True, targetFiles, fileTargetAmount)

    # build corpus from file
    corpus = get_corpus(files, save=True, file_path=contentFile)

    # build dictionary from corpus
    dictionary = build_dictionary(corpus, save=True, file_path=dictionaryPath)
    print("total word counts", len(dictionary))

    # build bit map
    bitMap = count_corpus(corpus, dictionary, save=True, vector_path=vectorPath)

    # build one word vector
    oneWord = build_one_word_vector(bitMap, save=False, one_word_path=oneWordPath)
    # print(oneWord)
    # calibrate dictionary
    # print(list(dictionary.items())[:100])
    wordFrequency = [[w, c] for w, c in zip(dictionary, oneWord)]
    # print(wordFrequency[:100])
    wordFrequency.sort(key=lambda x: x[1], reverse=True)
    wordFrequency = wordFrequency[sampleCount:wordTargetAmount + sampleCount]
    # print(list(w.items())[:10])
    # for w,c in wordFrequency[200:300]:
    #     print(w,c)

    # build reverse index
    # reverseIndex = cal_reverse_index(w, corpus, save=True, reverse_index_path=reverseIndexPath)
    # s = time.time()
    # reverseIndex = reverseIndex.todense()

    # build two word vector
    pathPrefix = str(wordTargetAmount) + "_"
    w = {w[0]: i for i, w in enumerate(wordFrequency[:wordTargetAmount])}
    bitMap2 = count_calibrated_corpus(corpus, w, save=True, vector_path=pathPrefix + calibratedBitMapPath, dp=False)
    oneWord2 = build_one_word_vector2(bitMap2)
    print(oneWord2)
    with open(pathPrefix + 'word_frequency.txt', 'w') as f:
        for c in oneWord2:
            f.write(str(c) + "\n")
    print("len of bitmap2", len(bitMap2))
    print("calibrate words length", len(w))
    twoWord = build_two_word_vector(bitMap2, corpus, save=True, two_word_path=pathPrefix + twoWordPath)
    sampleWords = 1
    print(wordFrequency[sampleWords][0], wordFrequency[sampleWords][1])
    with open(pathPrefix + "cocurrency.txt", 'w') as f:
        for c in twoWord[0]:
            f.write(str(c) + "\n")
    w = cal_eg(twoWord, save=True, eg_path=pathPrefix + egPath)
    with open(pathPrefix + 'Eigenvalues.txt', 'w') as f:
        for c in w:
            f.write(str(c) + "\n")


def build_word_id_vector(words, cps, save=True, vector_path=wordIDVectorPath):
    if os.path.exists(vector_path):
        print("word id vector already exist")
        return np.load(vector_path)
    start = time.time()
    w_id_vector = np.zeros((len(words), len(cps)))
    for j, cp in enumerate(cps):
        cp_words = cp.split()
        cp_words_set = set(cp_words)
        for i, w in enumerate(words):
            if w in cp_words_set:
                w_id_vector[i][j] = 1
    if save:
        print("save calibrated bitmap to file")
        np.save(vector_path, w_id_vector)
    end = time.time()
    print("build word id vector time", end - start, 's')
    return w_id_vector


def test_word_vector_cluster():
    # get files from path
    files = get_files(rootDir, True, targetFiles, fileTargetAmount)

    # build corpus from file
    corpus = get_corpus(files, save=True, file_path=contentFile)

    # build dictionary from corpus
    dictionary = build_dictionary(corpus, save=True, file_path=dictionaryPath)
    print("total word counts", len(dictionary))

    # build bit map
    bitMap = count_corpus(corpus, dictionary, save=True, vector_path=vectorPath)

    # build one word vector
    oneWord = build_one_word_vector(bitMap, save=False, one_word_path=oneWordPath)
    # calibrate dictionary
    # print(list(dictionary.items())[:100])
    wordFrequency = [[w, c] for w, c in zip(dictionary, oneWord)]
    # print(wordFrequency[:100])
    wordFrequency.sort(key=lambda x: x[1], reverse=True)
    wordFrequency = wordFrequency[sampleCount:wordTargetAmount + sampleCount]
    # print(list(w.items())[:10])
    # for w,c in wordFrequency[200:300]:
    #     print(w,c)

    # build reverse index
    # reverseIndex = cal_reverse_index(w, corpus, save=True, reverse_index_path=reverseIndexPath)
    # s = time.time()
    # reverseIndex = reverseIndex.todense()

    # build two word vector
    pathPrefix = str(wordTargetAmount) + "_"
    w = [w[0] for _, w in enumerate(wordFrequency[:wordTargetAmount])]
    word_id_vector = build_word_id_vector(w, corpus, save=True, vector_path=pathPrefix + wordIDVectorPath)
    kmeans_label_path = pathPrefix + str(clusterNum)+"_" + kmeansLabelPath

    w = {w[0]: i for i, w in enumerate(wordFrequency[:wordTargetAmount])}
    bitMap2 = count_calibrated_corpus(corpus, w, save=True, vector_path=pathPrefix + calibratedBitMapPath, dp=False)
    oneWord2 = build_one_word_vector2(bitMap2)
    print("oneword2",oneWord2)

    if os.path.exists(kmeans_label_path):
        print("kmeans label already exist")
        labels = np.load(kmeans_label_path)
    else:
        start = time.time()
        print("calculate kmeans")
        kmeans = KMeans(n_clusters=clusterNum).fit(word_id_vector)
        labels = kmeans.labels_
        np.save(kmeans_label_path, labels)
        end = time.time()
        print("time for kmeans", end - start, 's')

    print("labels")
    with open(pathPrefix + str(clusterNum)+"_" + "labelsFrequence.txt", 'w') as f:
        for i, l in enumerate(labels):
            f.write(str(l) + " " + str(wordFrequency[i][1]) + "\n")
    pca_path = pathPrefix + pcaPath
    if os.path.exists(pca_path):
        print("pca path already exist")
        truncated_word_id_vector = np.load(pca_path)
    else:
        start = time.time()
        print("calculate pca time")
        pca = PCA(n_components=3)
        truncated_word_id_vector = pca.fit_transform(word_id_vector)
        end = time.time()
        print("time for pca", end - start, 's')
        np.save(pca_path, truncated_word_id_vector)

    # draw
    ax = plt.axes(projection='3d')
    color = ['b', 'r', 'w', 'c', 'm', 'y', 'k', 'g'] * (clusterNum // 8 + 1)

    words_map = [[] for _ in range(clusterNum)]
    for i, l in enumerate(labels):
        words_map[l].append(bitMap2[i])

    count = []
    for i,ws in enumerate(words_map):
        count.append(len(ws))

    for i,l in enumerate(labels):
        if count[l] == 1:
            key = list(w.keys())[i]
            print(key,oneWord2[i])

    total_arr = []
    for i,ws in enumerate(words_map):
        total = 0
        print("length of words map",len(ws))
        for w in ws:
            total ^= w
        total_arr.append(gmpy.popcount(total))

    print("count of each cluster")
    print(count)
    print("max distance in each cluster")
    dis_arr= calculated_max_distance_in_cluster(words_map)
    print(dis_arr)
    print("total in each cluster")
    print(total_arr)
    for i,[c,d,t] in enumerate(zip(count,dis_arr,total_arr)):
        print("cluster",i,"count",c,"max distance",d,"total",t)
    # draw
    ws = [[] for _ in range(clusterNum)]
    for i, l in enumerate(labels):
        ws[l].append(truncated_word_id_vector[i])
    for i, w in enumerate(ws):
        c = color[i]
        w = np.array(w)
        print("length of different word", len(w), "shape of w", w.shape)
        x = w[:, 0]
        y = w[:, 1]
        z = w[:, 2]
        ax.scatter3D(x, y, z, c=c)
    plt.show()


def calculated_max_distance_in_cluster(words_map):
    dis_arr = [0]*len(words_map)
    for i, ws in enumerate(words_map):
        max_dis = -1
        for j in range(len(ws) - 1):
            for k in range(j + 1, len(ws)):
                dis = j ^ k
                if dis > max_dis:
                    max_dis = dis
        dis_arr[i] = max_dis
    return dis_arr


test_word_vector_cluster()
# test_two_word()

# for c in w:
#     print(c)
# print(w)


# csr_matrix((data, (row, col)), shape=(len(word_list), len(file_corpus)))
