import os
import re
import gmpy
from scipy.sparse import csr_matrix
import numpy as np
from numpy import linalg as LA
import random
import math
from utility import util
import time
from utility.attack import build_dictionary, build_revert_dictionary,co_counts, \
    build_bitmap, recover_by_cocounts, build_one_word_vector, random_sample_words,build_word_id_vector,\
bitmap_from_wid
from utility import file_helper


def pad_wid(wid, to_pad):
    padded_wid = np.copy(wid)
    for i in range(wid.shape[0]):
        potential_padding = []
        for j in range(wid.shape[1]):
            if wid[i][j] == 0:
                potential_padding.append(j)
        amount_to_pad = to_pad[i]
        pad_index = random.sample(potential_padding, amount_to_pad)
        for index in pad_index:
            padded_wid[i][index] = 1
    return padded_wid


def test_padded_recover(bitmap, padded_bitmap, try_amount):
    one_word = build_one_word_vector(bitmap)
    bitmap_count = [[bit, c, index] for index,(bit, c) in enumerate(zip(bitmap, one_word))]
    bitmap_count.sort(key=lambda x: x[1])
    padded_one_word = build_one_word_vector(padded_bitmap)
    padded_bitmap_count = [[bit, c, index] for index, (bit, c) in enumerate(zip(padded_bitmap, padded_one_word))]
    padded_bitmap_count.sort(key=lambda x: x[1])
    total_match = 0
    i=0
    j=0
    match_set = set()
    count_vectors = []
    for i in range(len(bitmap)):
        count_vector = []
        for k in range(len(bitmap)):
            if k==i:
                continue
            count_vector.append(co_counts(bitmap_count[i][0],bitmap_count[k][0]))
        count_vector.sort(reverse=True)
        count_vectors.append(count_vector)
    while j<len(padded_bitmap):
        while i<len(bitmap) and bitmap_count[i][1]<=padded_bitmap_count[j][1]:
            i+=1
        potential_match = []
        padded_count_vector = []
        for q in range(len(padded_bitmap)):
            if q == j:
                continue
            padded_count_vector.append(co_counts(padded_bitmap_count[j][0], padded_bitmap_count[q][0]))
        padded_count_vector.sort(reverse=True)
        for k in range(i):
            if k in match_set:
                continue
            count_vector = count_vectors[k]
            possible = True
            for v1,v2 in zip(count_vector,padded_count_vector):
                if v1>v2:
                    possible = False
                    break
            if possible:
                potential_match.append(k)
        try_count = try_amount
        if try_count>len(potential_match):
            try_count = len(potential_match)

        indexs = random.sample(potential_match,try_count)
        for index in indexs:
            if bitmap_count[index][2] == padded_bitmap_count[j][2]:
                total_match+=1
                match_set.add(index)
                break
        j+=1
    return total_match/len(bitmap)



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

# target = total//4*3
print("total files:", total)
# print("target amount:",target)
cps = cps[:total]
dic = build_dictionary(cps)
print("total words", len(dic))
bitmap = build_bitmap(dic, cps)
one_word = build_one_word_vector(bitmap)


for strategy in ["random", "top"]:
    print(strategy)
    sampled_dic = random_sample_words(dic, one_word, sample_words, strategy)
    wid = build_word_id_vector(cps, sampled_dic)
    bitmap = bitmap_from_wid(wid)
    one_word = build_one_word_vector(bitmap)

    max_try_amount = 20
    for minimal_padding in [250,500,750,1000]:
        origin_length = [[l, index] for index, l in enumerate(one_word)]
        origin_length.sort(key=lambda x: x[0], reverse=True)

        length_to_padding = util.to_padding(origin_length,minimal_padding)
        # print(length_to_padding)
        print("start prepare padded map")
        padded_wid = pad_wid(wid, length_to_padding)
        padded_bitmap = bitmap_from_wid(padded_wid)
        print("end prepare padded map")
        print("start recover")
        start = time.time()
        recover_rate = test_padded_recover(bitmap, padded_bitmap, max_try_amount)
        end = time.time()

        print("consuming time:", end - start)
        print("minimal padding:", minimal_padding)
        print("recover_rate", recover_rate)
# file = open("imdb_bitmap",'w')
# for bit in bitmap:
#     file.write(str(bit)+"\n")
# file.close()
#
# file = open("imdb_padded_bitmap",'w')
# for bit in padded_bitmap:
#     file.write(str(bit)+"\n")
# file.close()