import gmpy
import random
import math
import numpy as np


def build_bitmap(dic, cps, **kwargs):
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


def build_dictionary(cps, **kwargs):
    dic = dict()
    for i, cp in enumerate(cps):
        words = cp.split()
        for w in words:
            if w in dic:
                continue
            dic[w] = len(dic)
    return dic


def build_revert_dictionary(cps, **kwargs):
    dic = dict()
    revert_dict = dict()
    for i, cp in enumerate(cps):
        words = cp.split()
        for w in words:
            if w in dic:
                continue
            dic[w] = len(dic)
            revert_dict[len(dic) - 1] = w
    return revert_dict


def build_one_word_vector(bitmap, **kwargs):
    one_word = [0] * len(bitmap)
    for i, v in enumerate(bitmap):
        one_word[i] = gmpy.popcount(bitmap[i])
    return one_word


def count_map_from_bitmap(bitmap):
    cmap = dict()
    for i in range(len(bitmap)):
        count = gmpy.popcount(bitmap[i])
        if count == 0:
            continue
        if count not in cmap:
            cmap[count] = [i]
        else:
            cmap[count].append(i)
    return cmap


def co_counts(a, b):
    return gmpy.popcount(a & b)


# unique_by  'cosine' or 'count'
#
def recover_by_cocounts(bitmap, true_bitmap, unique_by='cosine',early_stop=False):
    cmap = count_map_from_bitmap(bitmap)
    true_cmap = count_map_from_bitmap(true_bitmap)
    if unique_by == 'cosine':
        cover_pair = find_unique_count_cosine(bitmap, cmap)
    elif unique_by == 'count':
        cover_pair = find_unique_count_count(cmap, true_cmap)
    else:
        raise Exception("unknown find unique method")
    count = 0
    for p in cover_pair:
        if p[0] == p[1]:
            count += 1
    # print("length of cover pair", len(cover_pair))
    # print("covered by unique count", count)
    return two_level_cover_set_cocounts(cover_pair, bitmap, true_bitmap,early_stop)


def find_unique_count_count(cmap, true_cmap):
    cover_pair = []
    for c in cmap:
        if len(cmap[c]) != 1 or c not in true_cmap or len(true_cmap[c]) != 1:
            continue
        a = cmap[c][0]
        b = true_cmap[c][0]
        cover_pair.append([a, b])
    return cover_pair


def bits_cosine_distance(a, b):
    x = gmpy.popcount(a & b)
    if x == 0:
        return 0
    y = gmpy.popcount(a) * gmpy.popcount(b)
    return x / math.sqrt(y)


def find_unique_count_cosine(bitmap, cmap):
    unique_pair = []
    cover_set_a = set()
    cover_set_b = set()
    for c in cmap:
        a = cmap[c]
        c_set = set(a)
        reference = 0
        for i in a:
            reference |= bitmap[i]
        reference_count = gmpy.popcount(reference)
        if reference_count == 0:
            print("cosine skip")
            continue
        x_dict = dict()
        y_dict = dict()
        for i in range(len(bitmap)):
            if i in c_set or i in cover_set_a:
                continue
            vx = bits_cosine_distance(reference, bitmap[i])
            if vx == 0:
                continue
            if vx in x_dict:
                x_dict[vx].append(i)
            else:
                x_dict[vx] = [i]
        for i in range(len(bitmap)):
            if i in c_set or i in cover_set_b:
                continue
            vx = bits_cosine_distance(reference, bitmap[i])
            if vx == 0:
                continue
            if vx in y_dict:
                y_dict[vx].append(i)
            else:
                y_dict[vx] = [i]
        for count in x_dict:
            if count not in y_dict or len(x_dict[count]) != 1 or len(y_dict[count]) != 1:
                continue
            unique_pair.append([x_dict[count][0], y_dict[count][0]])
            cover_set_a.add(x_dict[count][0])
            cover_set_b.add(y_dict[count][0])
    return unique_pair


def two_level_cover_set_cocounts(cover_pair, bitmap, true_bitmap, early_stop=False,**kwargs):
    cmap = count_map_from_bitmap(bitmap)
    true_cmap = count_map_from_bitmap(true_bitmap)
    n = len(bitmap)
    cover_set = set([p[0] for p in cover_pair])
    visited = set([p[1] for p in cover_pair])
    result_pair = cover_pair.copy()
    length = len(cover_set)

    while True:
        co_similarity = dict()
        true_co_similarity = dict()
        for j in range(n):
            if j in cover_set:
                continue
            co_similarity[j] = []
            for [index, true_index] in result_pair:
                count = co_counts(bitmap[index], bitmap[j])
                co_similarity[j].append(count)
        for j in range(n):
            if j in visited:
                continue
            true_co_similarity[j] = []
            for [index, true_index] in result_pair:
                true_count = co_counts(true_bitmap[true_index], true_bitmap[j])
                true_co_similarity[j].append(true_count)

        for c in cmap:
            if c not in true_cmap:
                continue
            a = cmap[c][:]
            random.shuffle(a)
            b = true_cmap[c]
            for i in a:
                if i in cover_set:
                    continue
                candidate = []
                for j in b:
                    if j in visited:
                        continue
                    equal = True
                    has_none_zero = False
                    for p, q in zip(co_similarity[i], true_co_similarity[j]):
                        if p != q:
                            equal = False
                            break
                        if p != 0:
                            has_none_zero = True
                    if equal and has_none_zero:
                        candidate.append(j)
                if len(candidate) != 1:
                    continue
                cover_set.add(i)
                result_pair.append([i, candidate[0]])
                visited.add(candidate[0])
        if len(cover_set) == length:
            break
        length = len(cover_set)
    if early_stop:
        return result_pair
    res = set()
    for p in result_pair:
        if p[0] == p[1]:
            res.add(p[0])
    return res


def random_sample_words(dic, one_word, count, strategy = "random",head_filter_count=200):
    words = dic.keys()
    m = [[w, c] for w, c in zip(words, one_word)]
    m.sort(key=lambda x: x[1], reverse=True)
    m = m[head_filter_count:]
    if strategy == 'random':
        keep_set = set(random.sample(range(len(m)), count))
        m = [e for i, e in enumerate(m) if i in keep_set]
    elif strategy == 'top':
        m = m[:count]

    sampled_words = [e[0] for e in m]
    new_dic = {w: i for i, w in enumerate(sampled_words)}
    return new_dic


def update_two_level_recover(cover_pair, bitmap, true_bitmap, dic, revert_dict, recovered_words_pair, **kwargs):
    cmap = count_map_from_bitmap(bitmap)
    true_cmap = count_map_from_bitmap(true_bitmap)
    n = len(bitmap)
    cover_set = set([p[0] for p in cover_pair])
    visited = set([p[1] for p in cover_pair])
    result_pair = cover_pair.copy()
    length = len(cover_set)

    for p in recovered_words_pair:
        w = p[0]
        true_w = p[1]
        if w in dic:
            cover_set.add(dic[w])
        if true_w in dic:
            visited.add(dic[true_w])

    while True:
        co_similarity = dict()
        true_co_similarity = dict()
        for j in range(n):
            if j in cover_set:
                continue
            co_similarity[j] = []
            for [index, true_index] in result_pair:
                count = co_counts(bitmap[index], bitmap[j])
                co_similarity[j].append(count)
        for j in range(n):
            if j in visited:
                continue
            true_co_similarity[j] = []
            for [index, true_index] in result_pair:
                true_count = co_counts(true_bitmap[true_index], true_bitmap[j])
                true_co_similarity[j].append(true_count)

        for c in cmap:
            if c not in true_cmap:
                continue
            a = cmap[c][:]
            random.shuffle(a)
            b = true_cmap[c]
            for i in a:
                if i in cover_set:
                    continue
                candidate = []
                for j in b:
                    if j in visited:
                        continue
                    equal = True
                    has_none_zero = False
                    for p, q in zip(co_similarity[i], true_co_similarity[j]):
                        if p != q:
                            equal = False
                            break
                        if p != 0:
                            has_none_zero = True
                    if equal and has_none_zero:
                        candidate.append(j)
                if len(candidate) != 1:
                    continue
                cover_set.add(i)
                result_pair.append([i, candidate[0]])
                visited.add(candidate[0])
                recovered_words_pair.append([revert_dict[i], revert_dict[candidate[0]]])
        if len(cover_set) == length:
            break
        length = len(cover_set)


def bitmap_from_wid(wid):
    bitmap = [0] * wid.shape[0]
    row = len(wid)
    col = len(wid[0])
    for i in range(row):
        for j in range(col):
            if wid[i][j] == 1:
                if j == 0:
                    addition = 1
                else:
                    addition = 2 << (j - 1)
                bitmap[i] += addition
    return bitmap


def build_word_id_vector(cps, dic):
    wid = np.zeros((len(dic), len(cps)))
    for j, cp in enumerate(cps):
        cp = cp.split()
        seen = set()
        for c in cp:
            if c in dic and c not in seen:
                i = dic[c]
                wid[i][j] = 1
                seen.add(c)
    return wid

def attack_parital_db(bitmap, drop_rate):
    open_index = random.sample(range(len(bitmap)),len(bitmap)*(1-drop_rate))
    encrypted_index = range(len(bitmap))




