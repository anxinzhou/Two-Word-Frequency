import time
import functools
import os
import pickle
import numpy as np
from numpy import linalg as LA
from scipy.sparse import save_npz
from scipy.sparse import load_npz


def time_profiler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, "consuming time:", end - start)
        return result

    return wrapper


def file_saver(func):
    def check_file_extentsion(file_path):
        if not use_pickle(file_path) and not use_numpy(file_path) and not use_sparse_matrix(file_path):
            raise Exception("wrong saving file path extension, use .data or .npy")

    def load_content(file_path):
        if use_pickle(file_path):
            return load_content_pickle(file_path)
        elif use_numpy(file_path):
            return load_content_numpy(file_path)
        elif use_sparse_matrix(file_path):
            return load_content_sparse_matrix(file_path)

    def save_content(content, file_path):
        if use_pickle(file_path):
            save_content_pickle(content, file_path)
        elif use_numpy(file_path):
            save_content_numpy(content, file_path)
        elif use_sparse_matrix(file_path):
            save_content_sparse_matrix(content, file_path)

    def use_pickle(file_path):
        return file_path.endswith(".data")

    def use_numpy(file_path):
        return file_path.endswith(".npy")

    def use_sparse_matrix(file_path):
        return file_path.endswith(".npz")

    def load_content_pickle(path):
        with open(path, 'rb') as fileHandler:
            return pickle.load(fileHandler)

    def save_content_pickle(content, path):
        with open(path, 'wb') as fileHandler:
            pickle.dump(content, fileHandler)

    def load_content_numpy(path):
        return np.load(path)

    def save_content_numpy(content, path):
        return np.save(path,content)

    def load_content_sparse_matrix(path):
        return load_npz(path)

    def save_content_sparse_matrix(content, path):
        return save_npz(path, content)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        allow_save = False
        file_path = kwargs["saving_path"] if "saving_path" in kwargs else None
        if file_path:
            check_file_extentsion(file_path)
        if "save" in kwargs:
            allow_save = kwargs["save"]
            if allow_save and not file_path:
                raise Exception("not specify file path when need to save")
        if allow_save:
            if os.path.exists(file_path):
                return load_content(file_path)
        result = func(*args, **kwargs)
        if allow_save:
            save_content(result, file_path)
        return result

    return wrapper


@time_profiler
@file_saver
def cal_eg(arr):
    w, _ = LA.eigh(arr)
    return w


def padding(length, minimal_padding):
    padded_length = []
    i = 0
    total_padding_size = 0
    total_original_size = sum(length)
    while i < len(length):
        # calculate number of element to padding
        j = i + minimal_padding - 1
        if len(length) - (j+1) < minimal_padding:
            j = len(length) - 1
        else:
            while j + 1 < len(length):
                if len(length) - (j + 1) == minimal_padding:
                    break
                if length[j + 1] == length[j]:
                    j += 1
                else:
                    break
        padding_element = j - i + 1

        # print("padding element number:",padding_element)
        if padding_element < minimal_padding:
            raise Exception("total length",len(length),"padding element", padding_element, "not enough element")
        padding_standard = length[i]
        # calculate padding size
        sub_padding_size = 0
        padded_length.extend([padding_standard]*(j-i+1))
        for k in range(i + 1, j + 1):
            sub_padding_size += padding_standard - length[k]
        # print("padding size:",sub_padding_size)
        total_padding_size += sub_padding_size
        i = j + 1

    return [total_padding_size, total_original_size, padded_length]


def padding_with_reference(length,reference_length):
    total_padding_size = 0
    total_original_size = sum(length)
    for l,ref_l in zip(length, reference_length):
        if l>ref_l:
            exit("unexpected condition")
        total_padding_size+=ref_l-l
    return total_padding_size, total_original_size, reference_length


def build_padding_map(word_count, minimal_padding):
    padding_segment = []
    i = 0
    while i < len(word_count):
        # calculate number of element to padding
        j = i + minimal_padding - 1
        if len(word_count) - j < minimal_padding:
            j = len(word_count) - 1
        else:
            while j + 1 < len(word_count):
                if len(word_count) - (j + 1) == minimal_padding:
                    break
                if word_count[j + 1][1] == word_count[j][1]:
                    j += 1
                else:
                    break
        padding_segment.append([ word_count[k][0] for k in range(i,j+1)])
        padding_element = j - i + 1
        # print("padding element number:",padding_element)
        if padding_element < minimal_padding:
            raise Exception("total length", len(word_count), "padding element", padding_element, "not enough element")
        i = j + 1

    return padding_segment


def to_padding(origin_length, minimal_padding):

    padded_length = []
    i = 0
    while i < len(origin_length):
        # calculate number of element to padding
        j = i + minimal_padding - 1
        if len(origin_length) - (j+1) < minimal_padding:
            j = len(origin_length) - 1
        else:
            while j + 1 < len(origin_length):
                if len(origin_length) - (j + 1) == minimal_padding:
                    break
                if origin_length[j + 1][0] == origin_length[j][0]:
                    j += 1
                else:
                    break
        padding_element = j - i + 1
        # print(i,j)
        # print("padding element number:",padding_element)
        if padding_element < minimal_padding:
            raise Exception("total length", len(origin_length), "padding element", padding_element, "not enough element","i",i,"j",j)
        padding_standard = origin_length[i][0]
        # calculate padding size
        sub_padding_size = 0
        padded_length.extend([padding_standard] * (j - i + 1))
        i = j + 1
    length_to_padding = []
    for total, origin in zip(padded_length, origin_length):
        if total<origin[0]:
            exit("total not larger than origin","total",total,"origin",origin[0])
        length_to_padding.append([total - origin[0],origin[1]])
    length_to_padding.sort(key=lambda x:x[1])
    length_to_padding = [l[0] for l in length_to_padding]
    return length_to_padding

def to_padding_with_segment(origin_length, minimal_padding):
    padded_length = []
    i = 0
    segments = []
    while i < len(origin_length):
        # calculate number of element to padding
        j = i + minimal_padding - 1
        if len(origin_length) - (j + 1) < minimal_padding:
            j = len(origin_length) - 1
        else:
            while j + 1 < len(origin_length):
                if len(origin_length) - (j + 1) == minimal_padding:
                    break
                if origin_length[j + 1][0] == origin_length[j][0]:
                    j += 1
                else:
                    break
        padding_element = j - i + 1
        # print(i,j)
        # print("padding element number:",padding_element)
        if padding_element < minimal_padding:
            raise Exception("total length", len(origin_length), "padding element", padding_element,
                            "not enough element", "i", i, "j", j)
        padding_standard = origin_length[i][0]
        # calculate padding size
        sub_padding_size = 0
        padded_length.extend([padding_standard] * (j - i + 1))
        segments.append([origin_length[k][1] for k in range(i,j+1)])
        i = j + 1
    length_to_padding = []
    for total, origin in zip(padded_length, origin_length):
        if total < origin[0]:
            exit("total not larger than origin", "total", total, "origin", origin[0])
        length_to_padding.append([total - origin[0], origin[1]])
    length_to_padding.sort(key=lambda x: x[1])
    length_to_padding = [l[0] for l in length_to_padding]
    return length_to_padding, segments