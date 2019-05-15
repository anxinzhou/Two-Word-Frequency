import pickle
import os
import time
import re
from math import sqrt
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse import load_npz
import gmpy

# use to check if is a word
# import enchant
# from nltk.stem.porter import PorterStemmer
# porter_stemmer = PorterStemmer()
# enDict = enchant.Dict('en_US')


rootDir = "maildir"
targetFiles = "./target.data"
contentFile = "./content.data"
dictionaryPath = "./dictionary.data"
prunedDictionaryPath = "./prunedDictionary.data"

vectorPath = "./countVector.data"
prunedVectorPath = "./prunedCountVector.data"
oneWordPath = "./oneWord.data"
prunedOneWordPath = "./prunedOneWord.data"
twoWordPath = "./twoWord.data"
reverseIndexPath = "./reverseIndex.data"

stop_words = {'the', 'and', 'to', 'of', 'i', 'a', 'in', 'it', 'that', 'is',
              'you', 'my', 'with', 'not', 'his', 'this', 'but', 'for',
              'me', 's', 'he', 'be', 'as', 'so', 'him', 'your'}
fileTargetAmount = 50000
wordTargetAmount = 10000


def get_files(path, save=True, file_path=targetFiles, target_amount=fileTargetAmount):
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
        c = []
        try:
            lines = f.readlines()
            # print(lines)
            reach_head = False
            for line in lines:
                if line.startswith('X-FileName'):
                    reach_head = True
                    continue
                # skip mail header
                if not reach_head:
                    continue
                # skip mail forward and appended mail
                if 'Forwarded by' in line:
                    continue
                if 'Original Message' in line:
                    continue
                if 'From:' in line:
                    continue
                if 'To:' in line:
                    continue
                if 'Cc:' in line:
                    continue
                if 'Sent:' in line:
                    continue
                if 'Subject:' in line:
                    continue
                if 'cc:' in line:
                    continue
                if 'subject:' in line:
                    continue
                if 'Subject:' in line:
                    continue
                if 'from:' in line:
                    continue
                # line = line.replace('\n',' ')
                line = re.sub(r"[^\s]*@[^\s]*", " ", line)
                line = re.sub(r"[^A-Za-z]", " ", line).lower()
                # print(line.split())
                tmp = line.split()
                line = [l for l in tmp if len(l) >= 2 and l not in stop_words]
                line = ' '.join(line)
                if len(line) != 0:
                    c.append(line)
                    # print(line)
        except UnicodeDecodeError:
            print(path)
            return ''
        # c_lists = content.split()
        # content = ' '.join([c for c in c_lists if enDict.check(c)]) # check if is a word
    return ' '.join(c)


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
            if w in dic:
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


def cal_reverse_index(word_list, file_corpus, save=True,reverse_index_path=reverseIndexPath):
    if os.path.exists(reverse_index_path):
        print("corpus already saved in file, skip doing again")
        return load_npz(reverse_index_path)
    row = []
    col = []
    data = []
    for j, cp in enumerate(file_corpus):
        cp = cp.split()
        seen = set()
        for c in cp:
            if c in word_list and c not in seen:
                i = word_list[c]
                row.append(i)
                col.append(j)
                data.append(1)
                seen.add(c)
    res = csr_matrix((data, (row, col)), shape=(len(word_list), len(file_corpus)))
    if save:
        save_npz(reverse_index_path, res)
    return res


files = get_files(rootDir,targetFiles,fileTargetAmount)
corpus = get_corpus(files,save=True,file_path=contentFile)
dictionary = build_dictionary(corpus,save=True,file_path=dictionaryPath)
bitMap = count_corpus(corpus, dictionary,save=True,vector_path=vectorPath)
oneWord = build_one_word_vector(bitMap,save=True,one_word_path=oneWordPath)
wordFrequency = [[w, c] for w, c in zip(dictionary, oneWord)]
wordFrequency.sort(key=lambda x: x[1], reverse=True)
w = {w[0]: i for i, w in enumerate(wordFrequency[:wordTargetAmount])}
# print(wordFrequency)
print("len of dictionary", len(dictionary))
print("len of one word", len(oneWord))
print("max frequency", max(oneWord))
reverseIndex = cal_reverse_index(w, corpus, save=True, reverse_index_path= reverseIndexPath)
print(reverseIndex)