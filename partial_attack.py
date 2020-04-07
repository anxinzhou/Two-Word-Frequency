import os
import re
import gmpy
from scipy.sparse import csr_matrix
import numpy as np
from numpy import linalg as LA
import random
import math
from utility.attack import build_dictionary, build_revert_dictionary, \
    build_bitmap, recover_by_cocounts, build_one_word_vector, random_sample_words,attack_paritial_db
from utility import file_helper


def test_recover(cps, start, end):
    cps = cps[start:end]
    dic = build_dictionary(cps)
    revert_dic = build_revert_dictionary(cps)
    bitmap = build_bitmap(dic, cps)
    recover_index = recover_by_cocounts(bitmap, bitmap)
    return set([revert_dic[i] for i in recover_index])


def test_recover2(cps, dic, revert_dic, start, end):
    cps = cps[start:end]
    bitmap = build_bitmap(dic, cps)
    one_word = build_one_word_vector(bitmap)
    # print(one_word)
    recover_index = recover_by_cocounts(bitmap, bitmap, unique_by='count')
    return set([revert_dic[i] for i in recover_index])


import argparse

total = 10000
sample_words = 10000
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
    raise Exception("Unknown db choice")

if len(cps) < total:
    raise Exception("not enough")
cps = cps[:total]
dic = build_dictionary(cps)
print("total words", len(dic))
bitmap = build_bitmap(dic, cps)
one_word = build_one_word_vector(bitmap)


sampled_dic = random_sample_words(dic, one_word, sample_words,strategy="top")
sampled_bitmap = build_bitmap(sampled_dic, cps)
reverse_sample_dict = dict()
for k, v in sampled_dic.items():
    reverse_sample_dict[v] = k

for drop_rate in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    print("drop rate", drop_rate)
    match_pair = attack_paritial_db(sampled_bitmap, drop_rate)
    res = set()
    for p in match_pair:
        if p[0] == p[1]:
            res.add(p[0])
    # print(match_pair)
    print("total recover rate", len(res) / len(sampled_dic))

# total_recover_set = test_recover2(cps, sampled_dic, reverse_sample_dict, 0, total)
# total_set = set(sampled_dic.keys())
# uncovered_set = total_set - total_recover_set
#
# print("total recover rate", len(total_recover_set) / len(sampled_dic))