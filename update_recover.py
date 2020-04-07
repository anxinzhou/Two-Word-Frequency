import os
import re
import gmpy
from scipy.sparse import csr_matrix
import numpy as np
from numpy import linalg as LA
import random
from utility.attack import build_dictionary, build_revert_dictionary, \
    build_bitmap, recover_by_cocounts, build_one_word_vector,\
    random_sample_words,count_map_from_bitmap,co_counts,update_two_level_recover
from utility import file_helper
random.seed(1)


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
    recover_index = recover_by_cocounts(bitmap, bitmap)
    return set([revert_dic[i] for i in recover_index])


def test_update_recover(cps, dic, revert_dic, recovered_words_pair):
    bitmap = build_bitmap(dic, cps)
    true_bitmap = build_bitmap(dic, cps)
    cmap = count_map_from_bitmap(bitmap)
    true_cmap = count_map_from_bitmap(bitmap)
    cover_pair = []

    # get cover_pair for recovered_words
    confirmed_set = set()
    true_confirmed_set = set()
    for pair in recovered_words_pair:
        w = pair[0]
        true_w = pair[1]
        if w in dic:
            confirmed_set.add(dic[w])
        if true_w in dic:
            true_confirmed_set.add(dic[true_w])
        if w in dic and true_w in dic:
            cover_pair.append([dic[w], dic[true_w]])

    for c in cmap:
        if c not in true_cmap:
            continue
        # trim cmap
        cset = set(cmap[c])
        true_cset = set(true_cmap[c])
        trimed_cset = cset.copy()
        true_trimed_cset = true_cset.copy()
        for ele in cset:
            if ele in confirmed_set:
                trimed_cset.remove(ele)
        for ele in true_cset:
            if ele in true_confirmed_set:
                true_trimed_cset.remove(ele)

        if len(trimed_cset) == 1 and len(true_trimed_cset) == 1:
            a = trimed_cset.pop()
            b = true_trimed_cset.pop()
            cover_pair.append([a, b])
            recovered_words_pair.append([revert_dic[a], revert_dic[b]])
    # print("length of cover_pair",len(cover_pair),"length of words pair",len(recovered_words_pair))
    return update_two_level_recover(cover_pair, bitmap, true_bitmap, dic, revert_dic, recovered_words_pair)


import argparse
total = 10000

sample_words = 2000

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
print("total files:", total)
cps = cps[:total]
dic = build_dictionary(cps)
print("total words", len(dic))
bitmap = build_bitmap(dic, cps)
one_word = build_one_word_vector(bitmap)

sampled_dic = random_sample_words(dic, one_word, sample_words)

reverse_sample_dict = dict()
for k, v in sampled_dic.items():
    reverse_sample_dict[v] = k
total_recover_set = test_recover2(cps, sampled_dic, reverse_sample_dict, 0, total)
print("total recover rate", len(total_recover_set) / len(sampled_dic))


start_loc = 0
division_part = 10
for division_part in range(5,40,5):
    recovered_words_pair = []
    seg = division_part
    if (total//seg)*seg<total:
        seg+=1
    for i in range(seg):
        end_loc = (total // division_part) * (i + 1)
        updated_dic = build_dictionary(cps[start_loc: end_loc])
        updated_sample_dic = updated_dic.copy()
        for k in updated_dic:
            if k not in sampled_dic:
                del updated_sample_dic[k]
        # if i==1:
        #     print(len(updated_dic),len(sampled_dic), end_loc)
        #     print(len(updated_sample_dic.keys()))
        #     break
        updated_sample_dic = {k: i for i, k in enumerate(updated_sample_dic.keys())}
        updated_reverse_sample_dic = dict()
        for k, v in updated_sample_dic.items():
            updated_reverse_sample_dic[v] = k

        test_update_recover(cps[start_loc:end_loc], updated_sample_dic, updated_reverse_sample_dic,
                                               recovered_words_pair)

        # print(len(recovered_words_pair))

        # print(recover_set)
        # recovered_words |= recover_words_from_update
        # print("dynamic recover rate", len(recovered_words)/len(sampled_dic))
    # print("end loc",end_loc)
    amount = 0
    for p in recovered_words_pair:
        if p[0] == p[1]:
            amount += 1
    print("segment number",division_part, "recover rate", amount / len(sampled_dic))