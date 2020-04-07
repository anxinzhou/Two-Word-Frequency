file_option = ["imdb_result_length_random10000.txt", "imdb_result_length_top10000.txt", "enron_result_length_random10000.txt",
               "enron_result_length_top10000.txt","blog_result_length_random10000.txt","blog_result_length_top10000.txt"]

file_option2 = ["imdb_file_size.txt","enron_file_size.txt","blog_file_size.txt"]

for file_to_padding in file_option2:
    print(file_to_padding)
    file = open(file_to_padding)
    content = file.readlines()
    length = []
    for c in content:
        c = c.strip()
        length.append(eval(c))
    length.sort(reverse=True)
    padding_option = range(50,251,50)
    for minimal_padding in padding_option:
        i=0
        total_padding_size = 0
        total_original_size = sum(length)
        while i < len(length):
            # calculate number of element to padding
            j=i+minimal_padding-1
            if len(length)-j < minimal_padding:
                j=len(length) - 1
            else:
                while j+1<len(length):
                    if len(length)-(j+1) == minimal_padding:
                        break
                    if length[j+1] == length[j]:
                        j+=1
                    else:
                        break
            padding_element = j-i+1

            # print("padding element number:",padding_element)
            if padding_element<minimal_padding:
                raise Exception("padding element",padding_element, "not enough element")
            padding_standard = length[i]
            # calculate padding size
            sub_padding_size = 0
            for k in range(i+1,j+1):
                sub_padding_size += padding_standard - length[k]
            # print("padding size:",sub_padding_size)
            total_padding_size += sub_padding_size
            i=j+1

        print("minimal padding size", minimal_padding)
        print("total padding size:",total_padding_size)
        print("original size:",total_original_size)
        print("padding ratio:",total_padding_size/total_original_size)
    file.close()