from dataset import ApacheDataSet, EnronDataSet
import os
import gmpy
import random
import util
from scipy.sparse import csr_matrix
import math
from bisect import bisect_left


def init_dataset(dataset_path, class_type, file_target_amount, path_prefix):
    def f():
        cps = class_type.get_corpus_from_dir(path=dataset_path, target_amount=file_target_amount, save=True,
                                             saving_path=path_prefix + "cps.data")
        print("length of file", len(cps))
        dic = class_type.build_dictionary(cps, save=True, saving_path=path_prefix + "dic.data")

        return class_type(cps, dic)

    return f


def sample_dataset(path_prefix, sample_strategy="topk"):
    def f(data_set):
        bitmap = data_set.build_bitmap(save=True, saving_path=path_prefix + "temp_bitmap.data")
        one_word = data_set.build_one_word_vector(bitmap)
        if sample_strategy == "topk":
            sampled_words = data_set.sample_top_k_words(one_word, k=wordTargetAmount)
        elif sample_strategy == "random":
            sampled_words = data_set.random_sample_words(one_word, k=wordTargetAmount)
        else:
            raise Exception("wrong sample strategy")
        dic = {w: i for i, w in enumerate(sampled_words)}
        return type(data_set)(data_set.cps, dic)

    return f


def process_dataset(path_prefix, true_dataset_percent, repeat_num=1):
    @util.time_profiler
    @util.file_saver
    def permute_dataset(matrix, **kwargs):
        len_col = matrix.shape[1]
        keep_set = set(random.sample(range(len_col), len_col * int(true_dataset_percent)))
        matrix = matrix.tocoo()
        data = matrix.data
        row = matrix.row
        col = matrix.col
        new_data = []
        new_row = []
        new_col = []
        for i in range(len(data)):
            if col[i] in keep_set:
                new_data.append(data[i])
                new_row.append(row[i])
                new_col.append(col[i])
        return csr_matrix((new_data, (new_row, new_col)), shape=matrix.shape)

    def count_map_from_bitmap(bitmap):
        cmap = dict()
        for i in range(len(bitmap)):
            count = gmpy.popcount(bitmap[i])
            if count not in cmap:
                cmap[count] = [i]
            else:
                cmap[count].append(i)
        return cmap

    def one_level_cover_pair(cmap, true_cmap):
        cover_pair = []

        def random_pick(l):
            return l[random.randint(0, len(l) - 1)]

        def least_distance_pick(c,true_cmap):
            keys = list(true_cmap.keys())
            i = bisect_left(keys,c)
            if i == len(keys):
                min_distance_c = keys[i-1]
            elif i==0:
                min_distance_c = keys[0]
            else:
                t1 = keys[i] - c
                t2 = c - keys[i-1]
                if t1 == t2:
                    min_distance_c = [keys[i],keys[i-1]][random.randint(0,1)]
                elif t1>t2:
                    min_distance_c = keys[i-1]
                else:
                    min_distance_c = keys[i]
            return min_distance_c
        for c in cmap.keys():
            if len(cmap[c]) != 1:
                continue
            index = cmap[c][0]
            if c in true_cmap:
                true_index = random_pick(true_cmap[c])
            else:
                least_distance_count = least_distance_pick(true_cmap)
                true_index = random_pick(c,true_cmap[least_distance_count])
            cover_pair.append([index, true_index])
        return cover_pair

    def bits_cosine_distance(a, b):
        x = gmpy.popcount(a & b)
        y = gmpy.popcount(a) * gmpy.popcount(b)
        return x / math.sqrt(y)

    def co_counts(a, b):
        return gmpy.popcount(a & b)

    @util.time_profiler
    @util.file_saver
    def two_level_cover_set(cover_pair, bitmap, true_bitmap, metric_func, **kwargs):
        m = len(cover_set)
        n = len(bitmap)
        result_set = set(cover_set)
        for [] in cover_set:
            cmap = dict()
            true_cmap = dict()
            for j in range(n):
                if j in result_set:
                    continue
                count = metric_func(bitmap[r], bitmap[j])
                true_count = metric_func(true_bitmap[r], true_bitmap[j])
                if count in cmap:
                    cmap[count].append(j)
                else:
                    cmap[count] = [j]
                if true_count in true_cmap:
                    true_cmap[count].append(j)
                else:
                    true_cmap[count] = [j]
            result_set |= one_level_cover_pair(cmap, true_cmap)
        return result_set

    def f(data_set):
        bitmap = data_set.build_bitmap(save=True, saving_path=path_prefix + "_bitmap.data")
        word_id_vector = data_set.build_word_id_vector(save=True, saving_path=path_prefix + "_wid.npz")
        co_counts_recover_rate = 0
        cosine_recover_rate = 0
        for i in range(repeat_num):
            true_word_id_vector = permute_dataset(word_id_vector)
            true_bitmap = data_set.bitmap_from_wid(true_word_id_vector)
            cmap = count_map_from_bitmap(bitmap)
            true_cmap = count_map_from_bitmap(true_bitmap)
            cover_pair_layer_one = one_level_cover_pair(cmap, true_cmap)
            co_counts_cover_set = two_level_cover_set(cover_pair_layer_one, bitmap, true_bitmap, co_counts)
            print("cocounts recover rate:", len(co_counts_cover_set) / len(bitmap))
            co_counts_recover_rate += len(co_counts_cover_set) / len(bitmap)
            cosine_cover_set = two_level_cover_set(cover_pair_layer_one, bitmap, true_bitmap, bits_cosine_distance)
            print("cosine recover rate:", len(cosine_cover_set) / len(bitmap))
            cosine_recover_rate += len(cosine_cover_set) / len(bitmap)
        print("cocounts recover average rate", co_counts_recover_rate / repeat_num)
        print("cosine recover average rate", cosine_recover_rate / repeat_num)

    return f


fileTargetAmount = 10000

test_true_dataset_percent = [0.5, 0.6, 0.7, 0.8, 0.9, 1][::-1]
word_target_amount = range(500,2501,500)

for wordTargetAmount in word_target_amount:
    print("word target amount", wordTargetAmount)
    for trueDatasetPercent in test_true_dataset_percent:
        print("true data set percent", str(trueDatasetPercent * 100) + "%")
        enron_file_prefix = "./cache/" + EnronDataSet.__name__ + "_" + str(fileTargetAmount) + "_"
        EnronDataSet.process(init_data_set=init_dataset("maildir", EnronDataSet, fileTargetAmount,
                                                        enron_file_prefix),
                             sample_data_set=sample_dataset(enron_file_prefix + str(wordTargetAmount) + "_",
                                                            sample_strategy="random"),
                             process_data_set=process_dataset(
                                 enron_file_prefix + str(wordTargetAmount) + "_" + str(trueDatasetPercent * 100) + "_",
                                 trueDatasetPercent, repeat_num=5)
                             )
