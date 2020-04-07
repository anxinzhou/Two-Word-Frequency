import os
import re
import gmpy
from scipy.sparse import csr_matrix
import numpy as np
from numpy import linalg as LA
import random
import math
from utility.attack import build_dictionary, build_revert_dictionary, \
    build_bitmap, recover_by_cocounts, build_one_word_vector, random_sample_words,build_word_id_vector,\
bitmap_from_wid
from utility import file_helper


def file_defense_padding(wid):
    len_w = len(wid)
    len_id = len(wid[0])
    file_count = math.ceil(math.log(len_id,2))
    to_pad_file_id = random.sample(range(len_id),file_count)
    for order in range(len_w):
        count = 0
        while order!=0:
            if order&1 == 1:
                file_id = to_pad_file_id[count]
                wid[order][file_id] = 1
            order= order>>1
            count += 1
    return wid


def test_file_defense(bitmap, true_bitmap):
    recover_index = recover_by_cocounts(bitmap, true_bitmap)
    return recover_index

# files = files_from_dir("imdb",1000000000)
# all = len(files)
# all=75000
# import time
# random.seed(time.time())
import argparse

total = 10000
parser = argparse.ArgumentParser(description='input name of dir to process')
parser.add_argument('--db', help='three databases, blog, imdb, enron')
args = parser.parse_args()
dbname = args.db
if (dbname == 'blog'):
    cps = file_helper.blog_get_corpus_from_dir("pre-processed-blogs", total)
elif (dbname == 'imdb'):
    cps = file_helper.imdb_get_corpus_from_dir("imdb", total)
elif (dbname == 'enron'):
    cps = file_helper.enron_get_corpus_from_dir("maildir", total)
else:
    raise Exception("Unknown db choise")
if len(cps)<total:
    raise Exception("not enough")
print("enron")
print("total files:",total)
# print("target amount:",target)
cps = cps[:total]
dic = build_dictionary(cps)
print("total words",len(dic))
bitmap = build_bitmap(dic,cps)
one_word = build_one_word_vector(bitmap)

for strategy in ["random", "top"]:
    print("sample strategy:",strategy)
    for sample_words in [2000,4000,6000,8000,10000]:
        print("sample words", sample_words)
        sampled_dic = random_sample_words(dic, one_word, sample_words, strategy)
        wid_before = build_word_id_vector(cps, sampled_dic)
        wid_tmp = wid_before.copy()
        wid_after = file_defense_padding(wid_tmp)
        bitmap_before = bitmap_from_wid(wid_before)
        bitmap_after = bitmap_from_wid(wid_after)

        reverse_sample_dict = dict()
        for k, v in sampled_dic.items():
            reverse_sample_dict[v] = k

        total_recover_set = test_file_defense(bitmap_before, bitmap_after)
        print("total recover rate", len(total_recover_set) / len(sampled_dic))