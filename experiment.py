from dataset import IMDBDataSet, EnronDataSet
import os
import gmpy
import random
import util
from scipy.sparse import csr_matrix
import math
import time
import numpy as np
from bisect import bisect_left
import sys
from queue import PriorityQueue as PQ


def bits_cosine_distance(a, b):
    x = gmpy.popcount(a & b)
    if x == 0:
        return 0
    y = gmpy.popcount(a) * gmpy.popcount(b)
    return x / math.sqrt(y)


def co_counts(a, b):
    return gmpy.popcount(a & b)


def init_dataset(dataset_path, class_type, file_target_amount, path_prefix):
    def f():
        cps = class_type.get_corpus_from_dir(path=dataset_path, target_amount=file_target_amount, save=True,
                                             saving_path=path_prefix + "cps.data")
        print("length of file", len(cps))
        dic = class_type.build_dictionary(cps, save=True, saving_path=path_prefix + "dic.data")
        print("total words:", len(dic))
        return class_type(cps, dic)

    return f


def sample_dataset(path_prefix, word_target_amount, sample_strategy="deunique"):
    def f(data_set):
        bitmap = data_set.build_bitmap(save=True, saving_path=path_prefix + "temp_bitmap.data")
        one_word = data_set.build_one_word_vector(bitmap)
        if sample_strategy == "topk":
            sampled_words = data_set.sample_top_k_words(one_word, k=word_target_amount)
        elif sample_strategy == "random":
            sampled_words = data_set.random_sample_words(one_word, k=word_target_amount)
        elif sample_strategy == "middle":
            raise Exception("not support")
        elif sample_strategy == "full":
            sampled_words = data_set.random_sample_words(one_word, k=len(bitmap))
        elif sample_strategy == "deunique":
            sampled_words = data_set.deunique_random_sample_words(one_word, k=word_target_amount)
        else:
            raise Exception("wrong sample strategy")
        dic = {w: i for i, w in enumerate(sampled_words)}
        return type(data_set)(data_set.cps, dic)

    return f


def padding_evaluation():
    def process_dataset(path_prefix):
        def f(data_set):
            pass


def static_evaluation():
    def process_dataset(path_prefix):
        def f(data_set):
            # result_lenngth
            bitmap = data_set.build_bitmap()
            one_word = data_set.build_one_word_vector(bitmap)
            with open(path_prefix + "result_length.txt", 'w') as f:
                for c in one_word:
                    f.write(str(c))
                    f.write("\n")
            dic = data_set.dic
            max_v = -1
            max_index = -1
            for i, c in enumerate(one_word):
                if c > max_v:
                    max_v = c
                    max_index = i
            # cal for cocounts for the word
            cocounts = []
            for i in range(len(dic)):
                cocounts.append(co_counts(bitmap[max_index], bitmap[i]))
            with open(path_prefix + "cocounts.txt", 'w') as f:
                for c in cocounts:
                    f.write(str(c))
                    f.write("\n")
            # cal cosine similarity
            cosine = []
            for i in range(len(dic)):
                cosine.append(bits_cosine_distance(bitmap[max_index], bitmap[i]))
            with open(path_prefix + "cosine.txt", 'w') as f:
                for c in cosine:
                    f.write(str(c))
                    f.write("\n")
            # cal eg
            wid = data_set.build_word_id_vector()
            wid = wid.toarray()
            two_word_vector = data_set.build_two_word_vector_by_wid()
            eg = util.cal_eg(two_word_vector)
            with open(path_prefix + "eg.txt", 'w') as f:
                for c in eg:
                    f.write(str(c))
                    f.write("\n")

        return f

    fileTargetAmount = 50000
    word_target_amount = range(5000, 20001, 5000)

    for wordTargetAmount in word_target_amount:
        print(EnronDataSet.__name__, "begin test")
        print("word target amount", wordTargetAmount)
        enron_file_prefix = "./cache/" + EnronDataSet.__name__ + "_" + str(fileTargetAmount) + "_"
        EnronDataSet.process(init_data_set=init_dataset("maildir", EnronDataSet, fileTargetAmount,
                                                        enron_file_prefix),
                             sample_data_set=sample_dataset(enron_file_prefix + str(wordTargetAmount) + "_"
                                                            , wordTargetAmount, sample_strategy="topk"),
                             process_data_set=process_dataset(
                                 enron_file_prefix + str(wordTargetAmount) + "_",
                             )
                             )

    for wordTargetAmount in word_target_amount:
        print(IMDBDataSet.__name__, "begin test")
        print("word target amount", wordTargetAmount)
        imdb_file_prefix = "./cache/" + IMDBDataSet.__name__ + "_" + str(fileTargetAmount) + "_"
        IMDBDataSet.process(init_data_set=init_dataset("imdb", IMDBDataSet, fileTargetAmount,
                                                       imdb_file_prefix),
                            sample_data_set=sample_dataset(imdb_file_prefix + str(wordTargetAmount) + "_",
                                                           wordTargetAmount, sample_strategy="topk"),
                            process_data_set=process_dataset(
                                imdb_file_prefix + str(wordTargetAmount) + "_"
                            )
                            )


def attack_evaluation():
    def process_dataset(path_prefix, keep_percent, repeat_num=1):
        @util.time_profiler
        def recover_by_cocounts(data_set, bitmap, true_bitmap):
            cmap = count_map_from_bitmap(bitmap)
            true_cmap = count_map_from_bitmap(true_bitmap)
            return two_level_cover_set_cocounts(one_level_cover_pair_cocounts(cmap, true_cmap), bitmap, true_bitmap)

        @util.time_profiler
        def one_level_cover_pair_cocounts(cmap, true_cmap):
            cover_pair = []
            for c in cmap:
                if len(cmap[c]) != 1 or c not in true_cmap or len(true_cmap[c]) != 1:
                    continue
                a = cmap[c][0]
                b = true_cmap[c][0]
                cover_pair.append([a, b])
            return cover_pair

        @util.time_profiler
        def two_level_cover_set_cocounts(cover_pair, bitmap, true_bitmap, **kwargs):
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
                    true_co_similarity[j] = []
                    for [index, true_index] in result_pair:
                        count = co_counts(bitmap[index], bitmap[j])
                        true_count = co_counts(true_bitmap[true_index], true_bitmap[j])
                        co_similarity[j].append(count)
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
            res = set()
            for p in result_pair:
                if p[0] == p[1]:
                    res.add(p[0])
            return res

        @util.time_profiler
        @util.file_saver
        def permute_dataset(matrix, true_dataset_percent, **kwargs):
            matrix = matrix.tocoo()
            len_row = matrix.shape[0]
            keep_set = set(random.sample(range(len_row), int(len_row * true_dataset_percent)))
            data = matrix.data
            row = matrix.row
            col = matrix.col
            new_data = []
            new_row = []
            new_col = []
            for i in range(len(data)):
                if row[i] in keep_set:
                    new_data.append(data[i])
                    new_row.append(row[i])
                    new_col.append(col[i])
            return csr_matrix((new_data, (new_row, new_col)), shape=matrix.shape)

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

        @util.time_profiler
        def find_unique_count_count_attack(bitmap, cmap):
            unique_pair = []
            for c in cmap:
                a = cmap[c]
                c_set = set(a)
                b = cmap[c][:]
                a_dict = dict()
                b_dict = dict()
                for i in range(len(a)):
                    x = a[i]
                    has_none_zero = False
                    vector = []
                    for j in range(len(bitmap)):
                        if j in c_set:
                            continue
                        vx = gmpy.popcount(x) & gmpy.popcount(j)
                        vector.append(str(vx))
                        if vx != 0:
                            has_none_zero = True
                    if has_none_zero:
                        str_vector = '#'.join(vector)
                        if str_vector in a_dict:
                            a_dict[str_vector].append(i)
                        else:
                            a_dict[str_vector] = [i]
                for i in range(len(b)):
                    x = b[i]
                    has_none_zero = False
                    vector = []
                    for j in range(len(bitmap)):
                        if j in c_set:
                            continue
                        vx = gmpy.popcount(x) & gmpy.popcount(j)
                        vector.append(str(vx))
                        if vx != 0:
                            has_none_zero = True
                    if has_none_zero:
                        str_vector = '#'.join(vector)
                        if str_vector in b_dict:
                            b_dict[str_vector].append(i)
                        else:
                            b_dict[str_vector] = [i]
                for s in a_dict:
                    if s not in b_dict or len(a_dict[s]) != 1 or len(b_dict[s]) != 1:
                        continue
                    unique_pair.append([a_dict[s][0], b_dict[s][0]])
            return unique_pair

        @util.time_profiler
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

        @util.time_profiler
        def first_find_unique_count_count_attack(bitmap, cmap):
            found = 0
            for c in cmap:
                a = cmap[c]
                c_set = set(a)
                b = cmap[c][:]
                a_dict = dict()
                b_dict = dict()
                for i in range(len(a)):
                    x = a[i]
                    has_none_zero = False
                    vector = []
                    for j in range(len(bitmap)):
                        if j in c_set:
                            continue
                        vx = gmpy.popcount(x) & gmpy.popcount(j)
                        vector.append(str(vx))
                        if vx != 0:
                            has_none_zero = True
                    if has_none_zero:
                        str_vector = '#'.join(vector)
                        if str_vector in a_dict:
                            a_dict[str_vector].append(i)
                        else:
                            a_dict[str_vector] = [i]
                for i in range(len(b)):
                    x = b[i]
                    has_none_zero = False
                    vector = []
                    for j in range(len(bitmap)):
                        if j in c_set:
                            continue
                        vx = gmpy.popcount(x) & gmpy.popcount(j)
                        vector.append(str(vx))
                        if vx != 0:
                            has_none_zero = True
                    if has_none_zero:
                        str_vector = '#'.join(vector)
                        if str_vector in b_dict:
                            b_dict[str_vector].append(i)
                        else:
                            b_dict[str_vector] = [i]
                for s in a_dict:
                    if s not in b_dict or len(a_dict[s]) != 1 or len(b_dict[s]) != 1:
                        continue
                    if a_dict[s][0] == b_dict[s][0]:
                        found += 1
                    break
            return found

        @util.time_profiler
        def first_find_unique_count_cosine(bitmap, cmap):
            found = 0
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
                    if i in c_set:
                        continue
                    vx = bits_cosine_distance(reference, bitmap[i])
                    if vx == 0:
                        continue
                    if vx in x_dict:
                        x_dict[vx].append(i)
                    else:
                        x_dict[vx] = [i]
                for i in range(len(bitmap)):
                    if i in c_set:
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
                    if x_dict[count][0] == y_dict[count][0]:
                        found += 1
                    break
            return found

        def test_unique_find(data_set):
            bitmap = data_set.build_bitmap()
            print("word number:", len(bitmap))
            cmap = count_map_from_bitmap(bitmap)
            start = time.time()
            unique_count = first_find_unique_count_count_attack(bitmap, cmap)
            end = time.time()
            print("count attack found rate", unique_count / len(cmap))
            print("count attack average time", (end - start) / len(cmap))
            start = time.time()
            unique_count = first_find_unique_count_cosine(bitmap, cmap)
            end = time.time()
            print("cosine attack found rate", unique_count / len(cmap))
            print("cosine attack average time", (end - start) / len(cmap))

        #
        @util.time_profiler
        def non_unique_recover_rate(data_set):
            bitmap = data_set.build_bitmap()
            print("word number:", len(bitmap))
            cmap = count_map_from_bitmap(bitmap)
            unique_pair = find_unique_count_count_attack(bitmap, cmap)
            result_set = two_level_cover_set_cocounts(unique_pair, bitmap, bitmap)
            print("count attack recover rate", len(result_set) / len(bitmap))
            unique_pair = find_unique_count_cosine(bitmap, cmap)
            result_set = two_level_cover_set_cocounts(unique_pair, bitmap, bitmap)
            print("cosine attack recover rate", len(result_set) / len(bitmap))

        def eigen_recover_rate(data_set):
            bitmap = data_set.build_bitmap()
            wid = data_set.build_word_id_vector()
            cocounts = data_set.build_two_word_vector_by_wid(wid)
            recover_count = len(bitmap)
            test_unique = dict()
            calculated_set = set()
            for i, k in enumerate(cocounts):
                x = np.sum(cocounts[i])
                y = cocounts[i, i]
                if x == y:
                    if x in test_unique:
                        test_unique[x].append(i)
                    else:
                        test_unique[x] = [i]
            layer_one_count = 0
            for c in test_unique:
                if len(test_unique[c]) != 1:
                    recover_count -= len(test_unique[c])
                    layer_one_count += len(test_unique[c])
                    for e in test_unique[c]:
                        calculated_set.add(e)
            print("later one total count", layer_one_count)
            print("layer one reduce count", len(bitmap) - recover_count)
            tmp_count = recover_count
            test_unique = dict()
            string_map = [0] * len(bitmap)
            for i, k in enumerate(cocounts):
                string_map[i] = '#'.join([str(cocounts[i, j]) for j in range(len(bitmap))])
            for i, e in enumerate(string_map):
                if e not in test_unique:
                    test_unique[e] = [i]
                else:
                    test_unique[e].append(i)
            f = open('cocounts.txt', 'w')
            layer_two_count = 0
            for c in test_unique:
                if len(test_unique[c]) != 1:
                    layer_two_count += len(test_unique[c])
                    count = 0
                    for e in test_unique[c]:
                        f.write(string_map[e])
                        f.write("\n")
                        if e in calculated_set:
                            continue
                        count += 1
                    recover_count -= count
                    f.write("\n")
            print("later two total count", layer_two_count)
            print("layer two reduce count", tmp_count - recover_count)
            print("eigen recover rate:", recover_count / len(bitmap))

        def padding_nonunique_recover_rate(data_set):
            bitmap = data_set.build_bitmap()
            one_word = data_set.build_one_word_vector(bitmap)
            wid = data_set.build_word_id_vector()
            wid = wid.toarray()
            middle = len(bitmap) // 2
            top_max = max(one_word[:middle])
            bottom_max = max(one_word[middle:])
            bottom_max = max(bottom_max, 200)
            print("top max", top_max)
            print("bottom max", bottom_max)
            # padding top half
            for i in range(middle):
                to_pad_amount = top_max - one_word[i]
                while to_pad_amount != 0:
                    pad_list = random.sample(range(len(wid[i])), to_pad_amount)
                    for index in pad_list:
                        if wid[i][index] == 0:
                            wid[i][index] = 1
                            to_pad_amount -= 1
                            if to_pad_amount == 0:
                                break
            # padding bottom half
            for i in range(middle, len(bitmap)):
                to_pad_amount = bottom_max - one_word[i]
                while to_pad_amount != 0:
                    pad_list = random.sample(range(len(wid[i])), to_pad_amount)
                    for index in pad_list:
                        if wid[i][index] == 0:
                            wid[i][index] = 1
                            to_pad_amount -= 1
                            if to_pad_amount == 0:
                                break
            bitmap = data_set.bitmap_from_dense_wid(wid)
            print(data_set.build_one_word_vector(bitmap))
            print("word number:", len(bitmap))
            cmap = count_map_from_bitmap(bitmap)
            unique_pair = find_unique_count_count_attack(bitmap, cmap)
            result_set = two_level_cover_set_cocounts(unique_pair, bitmap, bitmap)
            print("count attack recover rate", len(result_set) / len(bitmap))
            unique_pair = find_unique_count_cosine(bitmap, cmap)
            result_set = two_level_cover_set_cocounts(unique_pair, bitmap, bitmap)
            print("cosine attack recover rate", len(result_set) / len(bitmap))

        return padding_nonunique_recover_rate

        def get_wid(data_set):
            wid = data_set.build_word_id_vector().toarray()
            f = open('wid.txt', 'w')
            for c in wid:
                f.write(' '.join([str(e) for e in c]) + "\n")
            f.close()

        return get_wid

    fileTargetAmount = 50000

    word_target_amount = [1000]

    for wordTargetAmount in word_target_amount:
        print(EnronDataSet.__name__, "begin test")
        print("word target amount", wordTargetAmount)
        enron_file_prefix = "./cache/" + EnronDataSet.__name__ + "_" + str(fileTargetAmount) + "_"
        EnronDataSet.process(init_data_set=init_dataset("maildir", EnronDataSet, fileTargetAmount,
                                                        enron_file_prefix),
                             sample_data_set=sample_dataset(enron_file_prefix + str(wordTargetAmount) + "_"
                                                            , wordTargetAmount, sample_strategy="random"),
                             process_data_set=process_dataset(
                                 enron_file_prefix + str(wordTargetAmount) + "_",
                                 keep_percent=1, repeat_num=1)
                             )

    # for wordTargetAmount in word_target_amount:
    #     print(IMDBDataSet.__name__, "begin test")
    #     print("word target amount", wordTargetAmount)
    #     imdb_file_prefix = "./cache/" + IMDBDataSet.__name__ + "_" + str(fileTargetAmount) + "_"
    #     IMDBDataSet.process(init_data_set=init_dataset("imdb", IMDBDataSet, fileTargetAmount,
    #                                                    imdb_file_prefix),
    #                         sample_data_set=sample_dataset(imdb_file_prefix + str(wordTargetAmount) + "_",
    #                                                        wordTargetAmount, sample_strategy="random"),
    #                         process_data_set=process_dataset(
    #                             imdb_file_prefix + str(wordTargetAmount) + "_",
    #                             keep_percent=1, repeat_num=1)
    #                         )
    # to cal cocounts
    # def f(data_set):
    #     bitmap = data_set.build_bitmap()
    #     word_id_vector = data_set.build_word_id_vector()
    #     co_counts_recover_rate = 0
    #     # cosine_recover_rate = 0
    #     for i in range(repeat_num):
    #         true_word_id_vector = permute_dataset(word_id_vector, keep_percent)
    #         true_bitmap = data_set.bitmap_from_wid(true_word_id_vector)
    #         # cosine_cover_set = recover_by_cosine(data_set, bitmap, true_bitmap)
    #         # print("cosine recover rate:", len(cosine_cover_set) / len(bitmap))
    #         # cosine_recover_rate += len(cosine_cover_set) / len(bitmap)
    #         co_counts_cover_set = recover_by_cocounts(data_set, bitmap, true_bitmap)
    #         print("cocounts recover rate:", len(co_counts_cover_set) / len(bitmap))
    #         co_counts_recover_rate += len(co_counts_cover_set) / len(bitmap)
    #     print("cocounts recover average rate", co_counts_recover_rate / repeat_num)
    #     # print("cosine recover average rate", cosine_recover_rate / repeat_num)
    #
    # return f
