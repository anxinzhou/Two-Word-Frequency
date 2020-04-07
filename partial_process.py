import os
import re
import random
import gmpy
import math
from utility import util

from utility.attack import build_dictionary, build_revert_dictionary, \
    build_bitmap, recover_by_cocounts, build_one_word_vector, random_sample_words
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
    recover_index = recover_by_cocounts(bitmap, bitmap)
    return set([revert_dic[i] for i in recover_index])


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
cps = cps[:total]
dic = build_dictionary(cps)
print("total words",len(dic))
bitmap = build_bitmap(dic,cps)
one_word = build_one_word_vector(bitmap)

# get sampled_dic
sample_cps = cps[:total]
sample_dic_before = build_dictionary(sample_cps)
sample_bitmap = build_bitmap(sample_dic_before, sample_cps)
sample_one_word = build_one_word_vector(sample_bitmap)

# test partil update
#
# for start in range(0, 9001, segmentation):
#     end = start + segmentation
#     print("range:", start, "-", end)
#     sampled_dic = random_sample_words(sample_dic_before, sample_one_word, start, end)
#     reverse_sample_dict = dict()
#     for k, v in sampled_dic.items():
#         reverse_sample_dict[v] = k
#
#     total_recover_set = test_recover2(cps, sampled_dic, reverse_sample_dict, 0, total)
#     print("total recover rate", len(total_recover_set) / len(sampled_dic))

# test padding

amount = 10000
sampled_dic = random_sample_words(sample_dic_before, sample_one_word,  amount)
revert_dic = dict()
for k,v in sampled_dic.items():
    revert_dic[v] = k

bitmap = build_bitmap(sampled_dic, sample_cps)
one_word = build_one_word_vector(bitmap)

word_count = []
for i,count in enumerate(one_word):
    word = revert_dic[i]
    word_count.append([word,count])

word_count.sort(key=lambda x: x[1], reverse= True)

original_size = 0
for v in word_count:
    original_size += v[1]



for times in range(1,11):
    print("times:",times)
    segmentation = amount // times
    for minimal_padding_size in [250, 500, 750, 1000]:
        word_count_map = dict()
        for v in word_count:
            word_count_map[v[0]] = 0
        padding_segment = util.build_padding_map(word_count, minimal_padding_size)
        print("minial padding size",minimal_padding_size)
        count=0
        for start in range(0,  amount-segmentation+1, segmentation):
            count+=1
            end = start + segmentation
            if end+segmentation > amount:
                end = amount
            # print("range:", start, "-", end)
            sampled_dic = random_sample_words(sample_dic_before, sample_one_word, start, end)
            bitmap = build_bitmap(sampled_dic, sample_cps)
            one_word = build_one_word_vector(bitmap)

            revert_dict = dict()
            for w in sampled_dic:
                position = sampled_dic[w]
                num = one_word[position]
                word_count_map[w] += num

            for segment in padding_segment:
                max_length = -1
                for w in segment:
                    if word_count_map[w] > max_length:
                        max_length = word_count_map[w]
                for w in segment:
                    word_count_map[w] = max_length

                if max_length == -1:
                    exit("error case")

        if count!=times:
            raise Exception("count:",count,"times",times)

        total_size = 0
        for  w in word_count_map:
            total_size+=word_count_map[w]
        padding_size = total_size - original_size
        print("padding ratio:",padding_size/ original_size )