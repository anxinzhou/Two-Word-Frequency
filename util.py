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
        return np.save(content, path)

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
    w, _ = LA.eig(arr)
    return w
