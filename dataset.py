from abc import ABC, abstractmethod
import util
import gmpy
import numpy as np
import time
import os
from scipy.sparse import csr_matrix
import re
import random


class DataSet(ABC):
    stopWords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
                 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such',
                 'into',
                 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each',
                 'the',
                 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
                 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both',
                 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and',
                 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',
                 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where',
                 'too',
                 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my',
                 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}

    def __init__(self, cps, dic):
        self.cps = cps
        self.dic = dic

    @staticmethod
    @util.time_profiler
    @util.file_saver
    def build_dictionary(cps, **kwargs):
        dic = dict()
        for i, cp in enumerate(cps):
            words = cp.split()
            for w in words:
                if w in dic:
                    continue
                dic[w] = len(dic)
        return dic

    @util.time_profiler
    @util.file_saver
    def build_bitmap(self, **kwargs):
        dic = self.dic
        cps = self.cps
        bitmap = [0] * len(dic)
        for i in range(len(cps)):
            words = cps[i].split()
            seen = set()
            for w in words:
                if w in dic and w not in seen:
                    if i == 0:
                        addition = 1
                    else:
                        addition = 2 << (i - 1)
                    bitmap[dic[w]] += addition
                    seen.add(w)
        return bitmap

    @staticmethod
    @util.time_profiler
    @util.file_saver
    def bitmap_from_wid(wid, **kwargs):
        wid = wid.tocoo()
        bitmap = [0] * wid.shape[0]
        row = wid.row
        col = wid.col
        for i in range(len(row)):
            if col[i] == 0:
                addition = 1
            else:
                addition = 2 << (int(col[i]) - 1)
            bitmap[row[i]] += addition
        return bitmap

    @util.time_profiler
    @util.file_saver
    def build_word_id_vector(self, **kwargs):
        cps = self.cps
        dic = self.dic
        row = []
        col = []
        data = []
        for j, cp in enumerate(cps):
            cp = cp.split()
            seen = set()
            for c in cp:
                if c in dic and c not in seen:
                    i = dic[c]
                    row.append(i)
                    col.append(j)
                    data.append(1)
                    seen.add(c)
        return csr_matrix((data, (row, col)), shape=(len(dic), len(cps)))

    @staticmethod
    def build_one_word_vector(bitmap, **kwargs):
        one_word = [0] * len(bitmap)
        for i, v in enumerate(bitmap):
            one_word[i] = gmpy.popcount(bitmap[i])
        return one_word

    def sample_top_k_words(self, one_word, k, head_filter_count=200):
        dic = self.dic
        words = dic.keys()
        m = [[w, c] for w, c in zip(words, one_word)]
        m.sort(key=lambda x: x[1], reverse=True)
        return [b[0] for b in m[head_filter_count:head_filter_count + k]]

    def random_sample_words(self,one_word, k, head_filter_count=200):
        dic = self.dic
        words = dic.keys()
        tail_filter_count = len(dic)*0.1
        m = [[w, c] for w, c in zip(words, one_word)]
        m.sort(key=lambda x: x[1], reverse=True)
        m = m[head_filter_count:int(len(dic)-tail_filter_count)]
        keep_list = random.sample(range(len(m)),k)
        res = [m[i][0] for i in keep_list]
        return res


    @staticmethod
    def build_two_word_vector(bitmap, **kwargs):
        cur_number = 0
        skip_count = 0
        length = len(bitmap)
        two_word = np.zeros((length, length))
        start = time.time()
        for i in range(length):
            if i % 500 == 0:
                print("building two vector", i, "consuming time", time.time() - start, "s", "cur number", cur_number,
                      "skip count", skip_count)
            for j in range(length):
                count = gmpy.popcount(bitmap[i] & bitmap[j])
                if count != 0:
                    two_word[i][j] = count
                    cur_number += 1
        return two_word

    @classmethod
    def files_from_dir(cls, path, target_amount):
        files = []

        def dfs_dir(target_path):
            if len(files) == target_amount:
                return
            if os.path.isdir(target_path) and not cls.dir_filter(target_path):
                subs = os.listdir(target_path)
                for sub in subs:
                    dfs_dir(os.path.join(target_path, sub))
                    if len(files) == target_amount:
                        break
            elif os.path.isfile(target_path) and not cls.file_filter(target_path):
                files.append(target_path)
            else:
                print("skip " + target_path)

        dfs_dir(path)
        return files

    @classmethod
    @abstractmethod
    def dir_filter(cls, file):
        pass

    @classmethod
    @abstractmethod
    def file_filter(cls, file):
        pass

    @classmethod
    @util.time_profiler
    @util.file_saver
    def get_corpus_from_dir(cls, path, target_amount, **kwargs):
        fs = cls.files_from_dir(path, target_amount + target_amount // 50)
        print("geting corpus from files")
        cps = []
        skip_count = 0
        for i in range(len(fs)):
            content = cls.file_parser(fs[i])
            if len(content) == 0:
                skip_count += 1
                continue
            cps.append(content)
        print("skip:", skip_count)
        print("total:", len(cps))
        return cps[:target_amount]

    @classmethod
    @abstractmethod
    def file_parser(cls, file):
        pass

    @staticmethod
    def process(init_data_set, sample_data_set, process_data_set):
        data_set = init_data_set()
        sampled_data_set = sample_data_set(data_set)
        process_data_set(sampled_data_set)


class EnronDataSet(DataSet):
    def __init__(self, cps, dic):
        super().__init__(cps, dic)

    @classmethod
    def file_parser(cls, file):
        with open(file, 'r') as f:
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

                    # remove duplicate words
                    seen = set()
                    tmp = line.split()
                    line = []
                    for l in tmp:
                        if len(l) >= 2 and l not in cls.stopWords and l not in seen:
                            seen.add(l)
                            line.append(l)
                    line = ' '.join(line)
                    if len(line) != 0:
                        c.append(line)
                        # print(line)
            except UnicodeDecodeError:
                print(file)
                return ''
            # c_lists = content.split()
            # content = ' '.join([c for c in c_lists if enDict.check(c)]) # check if is a word
        return ' '.join(c)

    @classmethod
    def dir_filter(cls, file):
        return file.startswith(".")

    @classmethod
    def file_filter(cls, file):
        return not file.endswith(".")


class ApacheDataSet(DataSet):
    def __init__(self, cps, dic):
        super().__init__(cps, dic)

    def file_parser(cls, file):
        pass

    def dir_filter(cls, file):
        pass

    def file_filter(cls, file):
        pass
