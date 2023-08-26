import os
import pickle

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../DataShare/ApiSeq_and_Result/"


def remove_builtin(dir="output_connected.txt", dir1="output_processing", proj_name="test", folder="WordEmbedding", prefix=["", "bfs_"]):

    for pre in prefix:
        with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + pre + dir, "rt+") as fp:
            x = fp.read()

        with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + pre + dir1, "wt") as file:
            x = x.replace("<builtin>.", "")
            file.write(x)


def remove_short_seq2(proj_name="test", method="ft", dir="output_connected.txt", dir2="", min_len = 0, folder="WordEmbedding",
                      prefix=["", "bfs_"]):

    remove_dict_list = []
    for pre in prefix:
        remove_dict = {}
        j = 0

        with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + pre + dir, "rt") as fp:
            processing_lines = fp.readlines()
        seq_dup_dict = {}
        label_dup_dict = {}
        line_label = 0
        label_dup_list = []
        for line in processing_lines:
            line_label += 1
            if line in seq_dup_dict.keys():
                seq_dup_dict[line] += 1
                label_dup_dict[line].append(line_label)
                label_dup_list.append(line_label)
            else:
                seq_dup_dict[line] = 1
                label_dup_dict[line] = [line_label]
        with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/seq_dup_dict.pkl", "wb+") as file:
            pickle.dump(seq_dup_dict, file)

        with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + pre + dir2, "wt") as file:
            line_label= 0
            for line in processing_lines:
                line_label+= 1
                words = line.split()
                # print(line, len(words))
                if len(words) < min_len or line_label in label_dup_list:
                    j = j + 1
                    remove_dict[line_label] = j
                    # print(i,j)
                    continue
                file.write(line)
        remove_dict_list.append(remove_dict)

    remove_dict = remove_dict_list[0]
    # repair target dict
    with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + proj_name + "_" + method +
              "_target_num_dict.pkl", "rb") as file:
        target_dict = pickle.load(file)
    with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + proj_name + "_" + method +
              "_label_file_dict.pkl", "rb") as file:
        label_file_dict = pickle.load(file)
    operate_list = [0]*len(target_dict)
    for num in remove_dict.keys():
        cur = 0
        for item in target_dict.values():
            cur += 1
            if item < num:
                continue
            elif item == num:
                operate_list[cur-1] = -1
            else:
                operate_list[cur-1] = remove_dict[num]

    cur = 1
    count = 0
    # print(operate_list)
    # print(remove_dict)
    for i in operate_list:
        if i == -1:
            del target_dict[cur]
            del label_file_dict[cur]
            count += 1
        else:
            target_dict[cur] -= i

            if count != 0:
                target_dict[cur - count] = target_dict[cur]
                del target_dict[cur]
                label_file_dict[cur - count] = label_file_dict[cur]
                del label_file_dict[cur]
        cur += 1

    with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + proj_name + "_" +
              method + "_target_num_dict.pkl", "wb") as file:
        pickle.dump(target_dict, file)
    # print(len(target_dict))
    with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + proj_name + "_" +
              method + "_label_file_dict.pkl", "wb") as file2:
        pickle.dump(label_file_dict, file2)
    # print(len(label_file_dict))


def remove_short_seq(proj_name="test", method="ft", dir="output_connected.txt", min_len = 0, folder="WordEmbedding",
                     prefix=["", "bfs_"], proj_wait_to_be_merged=""):

    remove_dict_list = []
    for pre in prefix:
        try:
            with open(PROJECT_DIR + "Result/" + folder + "/" + proj_wait_to_be_merged + "/" + proj_wait_to_be_merged + "_" +
                      method + "_target_seq.txt", "r") as file2:
                wait_2_merge_line = file2.readlines()
        except:
            print("No benign set, set as empty.")
            wait_2_merge_line = []
        label_dup_dict = {}
        if prefix == [""]:
            for line in wait_2_merge_line:
                label_dup_dict[line.strip()] = -1
        else:
            if pre == "bfs_":
                for line in wait_2_merge_line:
                    line_list = line.strip().split(" ")
                    label_dup_dict[" ".join(line_list[:int(len(line_list)/2)])] = -1
            else:
                for line in wait_2_merge_line:
                    line_list = line.strip().split(" ")
                    label_dup_dict[" ".join(line_list[int(len(line_list)/2):])] = -1

        # print(label_dup_dict)

        remove_dict = {}
        j = 0
        with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + pre + dir, "rt") as fp:
            processing_lines = fp.readlines()
        seq_dup_dict = {}
        line_label = 0
        label_dup_list = []
        for line in processing_lines:
            line_label += 1
            line = line.strip()
            if line in label_dup_dict.keys():
                # print("@@" + line)
                seq_dup_dict[line_label] = -1
                if label_dup_dict[line] != -1:
                    seq_dup_dict[label_dup_dict[line]] += 1
                label_dup_list.append(line_label)
            else:
                seq_dup_dict[line_label] = 1
                label_dup_dict[line] = line_label
        # print(label_dup_dict)

        with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + pre + dir, "wt") as file:
            line_label = 0
            for line in processing_lines:
                line_label += 1
                words = line.split()
                # print(line, len(words))
                if len(words) < min_len or seq_dup_dict[line_label] == -1:
                    j = j + 1
                    remove_dict[line_label] = j
                    # print(i,j)
                    continue
                file.write(line)
        remove_dict_list.append(remove_dict)

    remove_dict = remove_dict_list[0]
    # repair target dict
    with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + proj_name + "_" + method +
              "_target_num_dict.pkl", "rb") as file:
        target_dict = pickle.load(file)
    with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + proj_name + "_" + method +
              "_label_file_dict.pkl", "rb") as file:
        label_file_dict = pickle.load(file)
    operate_list = [0]*(len(target_dict))
    seq_dup_dict_temp = {}
    for num in remove_dict.keys():
        cur = 0
        for key,item in target_dict.items():
            cur += 1
            if item < num:
                continue
            else:
                seq_dup_dict_temp[cur] = seq_dup_dict[item]  # 去除非setup文件
                if item == num:
                    operate_list[cur-1] = -1
                else:
                    operate_list[cur-1] = remove_dict[num]
    seq_dup_dict = seq_dup_dict_temp
    # print("operate_list")
    # print(operate_list)
    # print("seq_dup_dict")
    # print(seq_dup_dict)
    # print("remove_dict")
    # print(remove_dict)
    # print(len(operate_list))
    # print(len(seq_dup_dict))
    cur = 1
    for i in operate_list:
        if i == -1:
            del target_dict[cur]
            del label_file_dict[cur]
            del seq_dup_dict[cur]
        else:
            target_dict[cur] -= i
            if i != 0:
                target_dict[cur - i] = target_dict[cur]
                del target_dict[cur]
                if seq_dup_dict[cur] == -1:
                    del seq_dup_dict[cur]
                    print(cur)
                else:
                    seq_dup_dict[cur - i] = seq_dup_dict[cur]
                    del seq_dup_dict[cur]
                label_file_dict[cur - i] = label_file_dict[cur]
                del label_file_dict[cur]
        cur += 1

    with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + proj_name + "_" +
              method + "_target_num_dict.pkl", "wb") as file:
        pickle.dump(target_dict, file)
    # print(len(target_dict))
    with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + proj_name + "_" +
              method + "_label_file_dict.pkl", "wb") as file2:
        pickle.dump(label_file_dict, file2)
    # print(len(label_file_dict))
    # print(seq_dup_dict)
    with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/seq_dup_dict.pkl", "wb+") as file:
        pickle.dump(seq_dup_dict, file)


def remove_bad_api(dir="output_connected.txt", dir2="PreProcess/removed_api_list.txt", proj_name="test",
                   folder="WordEmbedding", prefix=["", "bfs_"]):

    for pre in prefix:
        with open(PROJECT_DIR + dir2, "r") as fp:
            rm_list = fp.readlines()

        with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + pre + dir, "r") as fp:
            lines = fp.readlines()

        result_list = []

        for line in lines:
            line_list = line.split()
            temp_list = []
            for j in line_list:
                sign = 0
                for item in rm_list:
                    if j == item.strip():
                        sign = 1
                if sign == 0:
                    temp_list.append(j)
            result_list.append(" ".join(temp_list))

        with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + pre + dir, "w") as file:
            for j in result_list:
                file.write(j + "\n")


def set_label(dir="output_connected.txt", proj_name="test", folder="WordEmbedding", prefix=["", "bfs_"]):

    for pre in prefix:
        output = ""
        # print(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + pre + dir)
        with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + pre + dir, "r") as fp:
            x = fp.readlines()
        fp.close()
        i = 0
        for line in x:
            if line.strip() == "":
                continue
            i += 1
            output += "__label__" + str(i) + " " + line
        # print(output)
        with open(PROJECT_DIR + "Result/" + folder + "/" + proj_name + "/" + pre + dir, "w") as file:
            file.write(output)


if __name__ == "__main__":
    dir = "output_connected.txt"
    proj_name = "test"
    folder = "test"
    remove_builtin(dir)
    remove_bad_api(dir, "PreProcess/removed_api_list.txt", proj_name, folder)
    remove_short_seq(dir, 3)
    set_label(dir)
