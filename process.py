# from sklearn.feature_extraction.text import CountVectorizer
# from scipy.sparse import coo_matrix
# from scipy.sparse import save_npz
# from scipy.sparse import load_npz
# import numpy as np

import pickle
import os
import time
import re
from math import sqrt
import gmpy


# use to check if is a word
# import enchant
# from nltk.stem.porter import PorterStemmer
# porter_stemmer = PorterStemmer()
# enDict = enchant.Dict('en_US')


rootDir = "/Users/anxin/Downloads/maildir"
targetFiles = "./target.data"
contentFile = "./content.data"
dictionaryPath = "./dictionary.data"
prunedDictionaryPath = "./prunedDictionary.data"

vectorPath = "./countVector.data"
prunedVectorPath = "./prunedCountVector.data"
oneWordPath = "./oneWord.data"
prunedOneWordPath = "./prunedOneWord.data"
twoWordPath = "./twoWord.data"

stop_words = {'the', 'and', 'to', 'of', 'i', 'a', 'in', 'it', 'that', 'is',
              'you', 'my', 'with', 'not', 'his', 'this', 'but', 'for',
              'me', 's', 'he', 'be', 'as', 'so', 'him', 'your'}
targetAmount = 30000
maxRatio = 0.9
minRatio = 0.001
twoWordRatio = 0.001


def get_files(path, save=True, file_path=targetFiles, target_amount=targetAmount):
    if os.path.exists(file_path):
        print("file paths already saved in file, skip doing again")
        return load_target_files(file_path)
    print("geting files from root dir")
    files = []
    dfs_dir(files, path, target_amount)
    if save:
        print("write to " + file_path)
        save_target_files(files, file_path)
    return files


def dfs_dir(files, path, target_amount):
    if len(files) == target_amount:
        return
    if os.path.isdir(path) and not path.startswith('.'):
        subs = os.listdir(path)
        for sub in subs:
            dfs_dir(files, os.path.join(path, sub), target_amount)
            if len(files) == target_amount:
                break
    elif os.path.isfile(path) and path.endswith('.'):
        files.append(path)
    else:
        print("skip " + path)


def save_target_files(fs, path):
    with open(path, 'wb') as fileHandler:
        pickle.dump(fs, fileHandler)


def load_target_files(path):
    with open(path, 'rb') as fileHandler:
        return pickle.load(fileHandler)


def parse_file(path):
    with open(path, 'r') as f:
        try:
            content = f.read()
        except UnicodeDecodeError:
            print(path)
            return ''
        content = re.sub(r"[^A-Za-z]", " ", content).lower()
        # c_lists = content.split()
        # content = ' '.join([c for c in c_lists if enDict.check(c)]) # check if is a word
    return content


def get_corpus(fs, save=True, file_path=contentFile):
    if os.path.exists(contentFile):
        print("corpus already saved in file, skip doing again")
        return load_target_files(file_path)
    print("geting corpus from files")
    cps = []
    skip_count = 0
    for i in range(len(fs)):
        content = parse_file(fs[i])
        if len(content) == 0:
            skip_count += 1
            continue
        cps.append(content)
    print("skip:", skip_count)
    print("total:", len(cps))
    if save:
        print("write content to " + contentFile)
        save_target_files(cps, file_path)
    return cps


def build_dictionary(cps, save=True, file_path=dictionaryPath):
    if os.path.exists(file_path):
        print("dic already exists, load from file")
        return load_target_files(file_path)
    print("building dictionary")
    dic = dict()
    for i, cp in enumerate(cps):
        words = cp.split()
        for w in words:
            if w in stop_words or len(w) <= 2 or w in dic:
                continue
            dic[w] = len(dic)

    # save
    if save:
        save_target_files(dic, file_path)
    return dic


def count_corpus(cps, dic, save=True, vector_path=vectorPath):
    if os.path.exists(vector_path):
        print("counter vector already exists, load from file")
        return load_target_files(vector_path)
    print("counting corpus")
    start = time.time()
    bit_map = [0] * len(dic)
    for i in range(len(cps)):
        words = cps[i].split()
        seen = set()
        for w in words:
            if w in dic and w not in seen:
                if i == 0:
                    addition = 1
                else:
                    addition = 2 << (i - 1)
                bit_map[dic[w]] += addition
                seen.add(w)
    end = time.time()
    print("counting corpus consuming time", end - start, 's')
    if save:
        save_target_files(bit_map, vector_path)
    return bit_map


def build_one_word_vector(bit_map, save=True, one_word_path=oneWordPath):
    # use sparse matrix
    if os.path.exists(one_word_path):
        print("one word already exists, load from file")
        return load_target_files(one_word_path)
    start = time.time()
    print("building one word")
    one_word = [0] * len(bit_map)
    for i, v in enumerate(bit_map):
        one_word[i] = gmpy.popcount(bit_map[i])
    end = time.time()
    print("building one word consuming time", end - start, "s")
    if save:
        save_target_files(one_word, one_word_path)
    return one_word


def build_two_word_vector(bit_map, cps, save=True, two_word_path=twoWordPath):
    if os.path.exists(two_word_path):
        print("two word already exists")
        return load_target_files(two_word_path)
    print("building two word")
    print("len of bitmap", len(bit_map))
    length = len(bit_map)
    cap = len(bit_map) * int(sqrt(len(bit_map)))
    data = [0] * cap
    row = [0] * cap
    col = [0] * cap
    start = time.time()
    cur_number = 0
    skip_count = 0
    for i in range(length - 1):
        if i % 500 == 0:
            print("building two vector", i, "consuming time", time.time() - start, "s", "cur number", cur_number,
                  "skip count", skip_count)
        for j in range(i + 1, length):
            count = gmpy.popcount(bit_map[i] & bit_map[j])
            if count != 0:
                # print(count)
                if count < len(cps) * twoWordRatio:
                    # print("low frequency skip")
                    skip_count += 1
                    continue
                if cur_number == cap:  # extend pre-allocated space
                    data.extend([0] * (cap // 2))
                    row.extend([0] * (cap // 2))
                    col.extend([0] * (cap // 2))
                    cap += (cap // 2)
                data[cur_number] = count
                row[cur_number] = i
                col[cur_number] = j
                cur_number += 1
    data = data[:cur_number]
    row = row[:cur_number]
    col = col[:cur_number]
    print("two word total time", time.time() - start, "s")
    two_word = [data, row, col]
    if save:
        print("save two word to file")
        save_target_files(two_word, two_word_path)
    return two_word


def build_pruned_dictionary(cps, save=True, pruned_dictionary_path=prunedDictionaryPath):
    if os.path.exists(pruned_dictionary_path):
        print("pruned dictionary already exists load from file")
        return load_target_files(pruned_dictionary_path)
    print("building pruned dictionary")
    dictionary = build_dictionary(cps, dictionaryPath)
    counter = count_corpus(cps, dictionary)

    one_word_vector = build_one_word_vector(counter)

    length = len(cps)
    new_dic = dict()
    for k in list(dictionary.keys()):
        count = one_word_vector[dictionary[k]]
        if length * minRatio <= count < length * maxRatio:
            new_dic[k] = len(new_dic)
    dictionary = new_dic
    if save:
        save_target_files(dictionary, pruned_dictionary_path)
    return dictionary


files = get_files(rootDir)
corpus = get_corpus(files)
prunedDictionary = build_pruned_dictionary(corpus)
print("total words:", len(prunedDictionary))
# print(prunedDictionary.keys())

# get bitmap
bitMap = count_corpus(corpus, prunedDictionary, save=True, vector_path=prunedVectorPath)

# calculate one_word
oneWord = build_one_word_vector(bitMap, save=True, one_word_path=prunedOneWordPath)
# calculate two_word
twoWord = build_two_word_vector(bitMap, corpus)
print("len two word:", len(twoWord[0]))
