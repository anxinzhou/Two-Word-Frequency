import os
import re
import gmpy
from scipy.sparse import csr_matrix
import numpy as np
from numpy import linalg as LA
import random
import math
from utility.attack import build_dictionary, build_revert_dictionary, \
    build_bitmap, recover_by_cocounts, build_one_word_vector, random_sample_words, \
    build_word_id_vector, bitmap_from_wid
from utility import file_helper

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
    raise Exception("Unknown db choice")

if len(cps) < total:
    raise Exception("not enough")
cps = cps[:total]
dic = build_dictionary(cps)
print("total words", len(dic))
bitmap = build_bitmap(dic, cps)
one_word = build_one_word_vector(bitmap)



# recover id first
wid = build_word_id_vector(cps, dic)
idbitmap = bitmap_from_wid(np.transpose(wid))
recover_pair = recover_by_cocounts(idbitmap, idbitmap, unique_by='count', early_stop=True)

## calculate recover rate
idres = set()
for p in recover_pair:
    if p[0] == p[1]:
        idres.add(p[0])
print("id recover length",len(idres))
print("id total recover rate", len(idres) / total)


# for sample_words in range(1000,10001,1000):
#     print("Number of query", sample_words)
#
#     sampled_dic = random_sample_words(dic, one_word, sample_words, strategy='random')
#     sampled_bitmap = build_bitmap(sampled_dic, cps)
#     sampled_one_word = build_one_word_vector(sampled_bitmap)
#     reverse_sample_dict = dict()
#     for k, v in sampled_dic.items():
#         reverse_sample_dict[v] = k
#
#     result_pair = recover_by_cocounts(sampled_bitmap, sampled_bitmap, unique_by='count', early_stop=True)
#
#     ## calculate recover rate
#     res = set()
#     for p in result_pair:
#         if p[0] == p[1]:
#             res.add(p[0])
#     print("words total recover rate", len(res) / len(sampled_dic))
#
#
#     # continue recover by id
#     confirmed_plaintext_set = set(p[0] for p in result_pair)
#     confirmed_encrypted_set = set(p[1] for p in result_pair)
#
#     total_set = set(range(len(sampled_dic)))
#     unconfirmed_plaintext = []
#     unconfirmed_encrypted = []
#     for k in total_set:
#         if k not in confirmed_plaintext_set:
#             unconfirmed_plaintext.append(k)
#         if k not in confirmed_encrypted_set:
#             unconfirmed_encrypted.append(k)
#
#     count = 0
#     d = dict()
#     for p,q in zip(unconfirmed_plaintext,unconfirmed_encrypted):
#         c= sampled_one_word[p]
#         if c not in d:
#             d[c] = 1
#         else:
#             d[c]+=1
#     # print(d)
#         # if sampled_one_word[q]>=2:
#         #     print()
#
#
#     recover_map = {k: v for [k, v] in result_pair}
#     reverse_recover_map = {v: k for [k, v] in result_pair}
#     random.shuffle(unconfirmed_encrypted)
#     random.shuffle(unconfirmed_plaintext)
#
#
#     fit_dic = dict()
#     for p in unconfirmed_plaintext:
#         pid_count = gmpy.popcount(sampled_bitmap[p])
#         pidlist = np.where(wid[p] == 1)[0]
#         for pid in pidlist:
#             if pid not in recover_map:
#                 continue
#             if recover_map[pid] != pid:
#                 continue
#         pidlist = [str(ele) for ele in pidlist]
#         hstr = '#'.join(pidlist)
#         if hstr not in fit_dic:
#             fit_dic[hstr] = [p]
#         else:
#             fit_dic[hstr].append(p)
#
#     for e in unconfirmed_encrypted:
#         eid_count = gmpy.popcount(sampled_bitmap[e])
#         eidlist = np.where(wid[e] == 1)[0]
#         for eid in eidlist:
#             if eid not in reverse_recover_map:
#                 continue
#             if recover_map[eid] != eid:
#                 continue
#         eidlist = [str(ele) for ele in eidlist]
#         hstr = '#'.join(eidlist)
#         if hstr not in fit_dic:
#             fit_dic[hstr] = [e]
#         else:
#             fit_dic[hstr].append(e)
#
#     id_recover_pair = []
#     for k in fit_dic:
#         if len(fit_dic[k])!=2:
#             continue
#         id_recover_pair.append(fit_dic[k])
#
#     # print(fit_dic)
#
#     res_id = set()
#     for p in id_recover_pair:
#         if p[0] == p[1]:
#             res_id.add(p[0])
#     print("total length of id recover pair",len(id_recover_pair))
#     print("recovered from id",len(res_id))
#     print("total recover rate after id recover", (len(res_id) + len(res)) / len(sampled_dic))
#
