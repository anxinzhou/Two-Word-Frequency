import os

def files_from_dir(path, target_amount, dir_filter, file_filter):
    files = []

    def dfs_dir(target_path):
        if len(files) == target_amount:
            return
        if os.path.isdir(target_path) and not dir_filter(target_path):
            subs = os.listdir(target_path)
            for sub in subs:
                dfs_dir(os.path.join(target_path, sub))
                if len(files) == target_amount:
                    break
        elif os.path.isfile(target_path) and not file_filter(target_path):
            files.append(target_path)
        else:
            print("skip " + target_path)

    dfs_dir(path)
    return files


# # imdb file size
# file_name = "imdb_file_size.txt"
# dir_name = "imdb"
# target_amount = 10000
# dir_filter = lambda x: x.startswith(".")
# file_filter = lambda x: not x.endswith(".txt")
# with open(file_name,"w") as f:
#     files = files_from_dir(dir_name, target_amount,dir_filter,file_filter)
#     print(len(files))
#     for file in files:
#         size = os.path.getsize(file)
#         f.write(str(size)+"\n")

# #enron file size
# file_name = "enron_file_size.txt"
# dir_name = "maildir"
# target_amount = 10000
# dir_filter = lambda x: x.startswith(".")
# file_filter = lambda x: not x.endswith(".")
# with open(file_name,"w") as f:
#     files = files_from_dir(dir_name, target_amount,dir_filter,file_filter)
#     print(len(files))
#     for file in files:
#         size = os.path.getsize(file)
#         f.write(str(size)+"\n")


#blog file size
file_name = "blog_file_size.txt"
dir_name = "pre-processed-blogs"
target_amount = 10000
dir_filter = lambda x: x.startswith(".")
file_filter = lambda x: not x.endswith(".xml")
with open(file_name,"w") as f:
    files = files_from_dir(dir_name, target_amount,dir_filter,file_filter)
    print(len(files))
    for file in files:
        size = os.path.getsize(file)
        f.write(str(size)+"\n")