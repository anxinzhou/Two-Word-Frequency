# def stemming_tokenizer(str_input):
#     words = re.sub(r"[^A-Za-z]", " ", str_input).lower().split()
#     return words

# def get_count_vector(cps, save=True, vector_path=vectorPath, dictionary_path=dictionaryPath):
#     if os.path.exists(vector_path):
#         print("dic and vector paths already saved in file, skip doing again")
#         x = np.load(vector_path)
#         if os.path.exists(dictionaryPath):
#             dic = load_target_files(dictionaryPath)
#             return x, dic
#         else:
#             raise Exception("no dictionary path")
#     vectorizer = CountVectorizer(stop_words=stop_words, tokenizer=stemming_tokenizer)
#     x = vectorizer.fit_transform(cps)
#     x = np.array(x.toarray)
#     dic = vectorizer.get_feature_names()
#     if save:
#         np.save(vector_path, x)
#         save_target_files(dic, dictionary_path)
#     return x, dic


# def filter_corpus(fs, save=True, file_path=filteredFile):
#     if os.path.exists(filteredFile):
#         print("filteter already exists, skip")
#         return load_target_files(filteredFile)
#     cps = []
#     skip_count = 0
#     for f in fs:
#         try:
#             vectorizer = CountVectorizer(stop_words=stop_words, tokenizer=stemming_tokenizer)
#             vectorizer.fit_transform([f])
#             cps.append(f)
#         except ValueError:
#             print("empty dictionary: " + f)
#             skip_count += 1
#     print("filter count:", skip_count)
#     print("total count:", len(cps))
#     if save:
#         print("write filtered content to " + file_path)
#         save_target_files(cps, file_path)
#     return cps


# def count_corpus2(cps, dic, save=True, vector_path=vectorPath):
#     # use sparse matrix
#     if os.path.exists(vector_path):
#         print("counter vector already exists, load from file")
#         return load_npz(vector_path)
#     print("counting corpus")
#     data = []
#     row = []
#     col = []
#     counter = np.zeros((len(cps), len(dic)), dtype=int)
#     print(counter.shape)
#     print("begin to count")
#     for i in range(len(cps)):
#         words = cps[i].split()
#         seen = set()
#         for w in words:
#             if w in dic and w not in seen:
#                 data.append(1)
#                 row.append(i)
#                 col.append(dic[w])
#                 seen.add(w)
#     counter = coo_matrix((data, (row, col)), shape=(len(cps), len(dic)), dtype=int)
#     print("counter shape", counter.shape)
#     if save:
#         save_npz(vector_path, counter)
#     return counter


# def build_one_word_vector_sparse(vector, save=True, one_word_path=oneWordPath):
#     # use sparse matrix
#     if os.path.exists(one_word_path):
#         print("one word already exists, load from file")
#         return np.load(one_word_path)
#     print("building one word vector, len of col", len(vector.col))
#     length = vector.shape[1]
#     one_word = np.zeros((length,), dtype=int)
#     col = vector.col
#     for c in col:
#         one_word[c] += 1
#     if save:
#         np.save(one_word_path, one_word)
#     return one_word


# def build_one_word_vector_dense(vector, save=True, one_word_path=oneWordPath):
#     # use dense matrix
#     if os.path.exists(one_word_path):
#         print("one word already exists, load from file")
#         return np.load(one_word_path)
#     print("building one word")
#     one_word = np.sum(vector, axis=0)
#     print("one word shape", one_word.shape)
#     if save:
#         np.save(one_word_path, one_word)
#     return one_word


# def build_two_word_vector_dense(vector, save=True, two_word_path=twoWordPath):
#     if os.path.exists(two_word_path):
#         print("two word already exists")
#         return load_npz(two_word_path)
#     print("building two word")
#     length = vector.shape[1]
#     data = []
#     row = []
#     col = []
#
#     print("calculate bitmap")
#     bit_map = [None] * length
#     for i in range(length):
#         bit_map[i] = int(''.join([str(vector[i, j]) for j in range(length)]), 2)
#         # print(bit_map[i])
#     print("calculate bitmap over")
#
#     for i in range(length - 1):
#         print(i)
#         for j in range(i + 1, length):
#             count = bit_map[i] & bit_map[j]
#             if count != 0:
#                 data.append(count)
#                 row.append(i)
#                 col.append(j)
#     two_word = coo_matrix((data, (row, col)), shape=(length, length), dtype=int)
#     if save:
#         print("save two word to file")
#         save_npz(two_word_path, two_word)
#     return two_word


# def upper_bound(arr, start, v):
#     lo, hi = start, len(arr)
#     while lo < hi:
#         mid = lo + (hi - lo) // 2
#         if arr[mid] <= v:
#             lo = mid + 1
#         else:
#             hi = mid
#     return lo


# def build_two_word_vector_sparse(vector, save=True, two_word_path=twoWordPath):
#     # use sparse matrix
#     if os.path.exists(two_word_path):
#         print("two word already exists")
#         return load_npz(two_word_path)
#     print("building two word vector")
#     length = vector.shape[1]
#     v_row = vector.row
#     v_col = vector.col
#
#     l_row = len(vector.row)
#     i = 0
#     dic = dict()
#     while i < l_row:
#         start = i
#         end = upper_bound(v_row, i, i)
#         for p in range(start, end - 1):
#             for q in range(p + 1, end):
#                 tp = (v_col[p], v_col[q])
#                 if v_col[p] < v_col[q]:
#                     tp = (v_col[p], v_col[q])
#                 else:
#                     tp = (v_col[q], v_col[p])
#                 if tp not in dic:
#                     dic[tp] = 0
#                 else:
#                     dic[tp] += 1
#         i = end
#     data = []
#     row = []
#     col = []
#     for k, v in dic:
#         data.append(v)
#         row.append(k[0])
#         row.append(k[1])
#     two_word = coo_matrix((data, (row, col)), shape=(length, length), dtype=int)
#     if save:
#         save_npz(two_word_path, two_word)
#     return two_word