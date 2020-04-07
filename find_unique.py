import os
import re
import random
import gmpy
import math

from utility.attack import build_dictionary, build_revert_dictionary, \
    build_bitmap, recover_by_cocounts, build_one_word_vector, random_sample_words
from utility import file_helper

sample_words = 2000

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
if len(cps) < total:
    raise Exception("not enough")
target = total // 4 * 3
print("total files:", total)
print("target amount:", target)
cps = cps[:total]
dic = build_dictionary(cps)
print("total words", len(dic))
bitmap = build_bitmap(dic, cps)
one_word = build_one_word_vector(bitmap)


def test_recover2(cps, dic, revert_dic, start, end):
    cps = cps[start:end]
    bitmap = build_bitmap(dic, cps)
    recover_index = recover_by_cocounts(bitmap, bitmap)
    return set([revert_dic[i] for i in recover_index])


for strategy in ["random", "top"]:
    print(strategy)
    sampled_dic = random_sample_words(dic, one_word, sample_words, strategy)
    reverse_sample_dict = dict()
    for k, v in sampled_dic.items():
        reverse_sample_dict[v] = k

    total_recover_set = test_recover2(cps, sampled_dic, reverse_sample_dict, 0, total)
    print("total recover rate", len(total_recover_set) / len(sampled_dic))
