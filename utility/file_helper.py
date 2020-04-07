import random
import os
import re
stopWords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
             'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such',
             'into',
             'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each',
             'the',
             'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
             'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both',
             'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and',
             'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over',
             'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where',
             'too',
             'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my',
             'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'nbsp'}


def files_from_dir(path, target_amount, dir_filter, file_filter):
    files = []

    def dfs_dir(target_path):
        if len(files) == target_amount:
            return
        if os.path.isdir(target_path) and not dir_filter(target_path):
            subs = os.listdir(target_path)
            for sub in subs:
                dfs_dir(os.path.join(target_path, sub))
                # if len(files) == target_amount:
                #     break
        elif os.path.isfile(target_path) and not file_filter(target_path):
            files.append(target_path)
        else:
            print("skip " + target_path)

    dfs_dir(path)
    # random.shuffle(files)
    keep_set = set(random.sample(range(len(files)), target_amount))
    files = [e for i, e in enumerate(files) if i in keep_set]
    print(files[:5])
    return files


def blog_get_corpus_from_dir(path, target_amount, **kwargs):
    def dir_filter(file):
        return file.startswith(".")

    def file_filter(file):
        return not file.endswith(".xml")

    def file_parser(file):
        with open(file, 'r') as f:
            c = []
            try:
                lines = f.readlines()
                for line in lines:
                    line = re.sub(r"[^\s]*@[^\s]*", " ", line)
                    line = re.sub(r"[^A-Za-z]", " ", line).lower()
                    # remove duplicate words
                    seen = set()
                    tmp = line.split()
                    line = []
                    for l in tmp:
                        if len(l) >= 2 and l not in stopWords and l not in seen:
                            seen.add(l)
                            line.append(l)
                    line = ' '.join(line)
                    if len(line) != 0:
                        c.append(line)
                        # print(line)
            except UnicodeDecodeError:
                print(file)
                return ''
            # c_lists = content.split()
            # content = ' '.join([c for c in c_lists if enDict.check(c)]) # check if is a word
        return ' '.join(c)

    fs = files_from_dir(path, target_amount + target_amount // 50, dir_filter, file_filter)
    print("geting corpus from files")
    cps = []
    skip_count = 0
    for i in range(len(fs)):
        content = file_parser(fs[i])
        if len(content) == 0:
            skip_count += 1
            continue
        cps.append(content)
    print("skip:", skip_count)
    print("total:", len(cps))
    return cps[:target_amount]


def enron_get_corpus_from_dir(path, target_amount, **kwargs):
    def dir_filter(file):
        return file.startswith(".")

    def file_filter(file):
        return not file.endswith(".")

    def file_parser(file):
        with open(file, 'r') as f:
            c = []
            try:
                lines = f.readlines()
                # print(lines)
                reach_head = False
                for line in lines:
                    if line.startswith('X-FileName'):
                        reach_head = True
                        continue
                    # skip mail header
                    if not reach_head:
                        continue
                    # skip mail forward and appended mail
                    if 'Forwarded by' in line:
                        continue
                    if 'Original Message' in line:
                        continue
                    if 'From:' in line:
                        continue
                    if 'To:' in line:
                        continue
                    if 'Cc:' in line:
                        continue
                    if 'Sent:' in line:
                        continue
                    if 'Subject:' in line:
                        continue
                    if 'cc:' in line:
                        continue
                    if 'subject:' in line:
                        continue
                    if 'Subject:' in line:
                        continue
                    if 'from:' in line:
                        continue
                    # line = line.replace('\n',' ')
                    line = re.sub(r"[^\s]*@[^\s]*", " ", line)
                    line = re.sub(r"[^A-Za-z]", " ", line).lower()
                    # print(line.split())

                    # remove duplicate words
                    seen = set()
                    tmp = line.split()
                    line = []
                    for l in tmp:
                        if len(l) >= 2 and l not in stopWords and l not in seen:
                            seen.add(l)
                            line.append(l)
                    line = ' '.join(line)
                    if len(line) != 0:
                        c.append(line)
                        # print(line)
            except UnicodeDecodeError:
                print(file)
                return ''
            # c_lists = content.split()
            # content = ' '.join([c for c in c_lists if enDict.check(c)]) # check if is a word
        return ' '.join(c)

    fs = files_from_dir(path, target_amount + target_amount // 50, dir_filter, file_filter)
    print("geting corpus from files")
    cps = []
    skip_count = 0
    for i in range(len(fs)):
        content = file_parser(fs[i])
        if len(content) == 0:
            skip_count += 1
            continue
        cps.append(content)
    print("skip:", skip_count)
    print("total:", len(cps))
    return cps[:target_amount]


def imdb_get_corpus_from_dir(path, target_amount, **kwargs):
    def dir_filter(file):
        return file.startswith(".")

    def file_filter(file):
        return not file.endswith(".txt")

    def file_parser(file):
        with open(file, 'r') as f:
            c = []
            try:
                lines = f.readlines()
                for line in lines:
                    line = re.sub(r"[^\s]*@[^\s]*", " ", line)
                    line = re.sub(r"[^A-Za-z]", " ", line).lower()
                    # remove duplicate words
                    seen = set()
                    tmp = line.split()
                    line = []
                    for l in tmp:
                        if len(l) >= 2 and l not in stopWords and l not in seen:
                            seen.add(l)
                            line.append(l)
                    line = ' '.join(line)
                    if len(line) != 0:
                        c.append(line)
                        # print(line)
            except UnicodeDecodeError:
                print(file)
                return ''
            # c_lists = content.split()
            # content = ' '.join([c for c in c_lists if enDict.check(c)]) # check if is a word
        return ' '.join(c)

    fs = files_from_dir(path, target_amount + target_amount // 50, dir_filter, file_filter)
    print("geting corpus from files")
    cps = []
    skip_count = 0
    for i in range(len(fs)):
        content = file_parser(fs[i])
        if len(content) == 0:
            skip_count += 1
            continue
        cps.append(content)
    print("skip:", skip_count)
    print("total:", len(cps))
    return cps[:target_amount]