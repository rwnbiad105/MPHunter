from SequenceAnalyzer.Analysis.compute_distant import run2
from scipy.spatial.distance import cosine
import os
import numpy as np
from scipy import spatial

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prepare_4_cluster(proj_name="test", method="w2v", prefix=[""]):
    
    for pre in prefix:
        temp = list()
        with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + pre+
                  proj_name + "_" + method + "_target_vec.txt", "r") as fp:
            temp.append(fp.readlines())
    vec_lists = []
    for lines in temp:
        num = len(lines)
        vec_list = []
        for j in range(num):
            temp_list = []
            for w in lines[j].split():
                temp_list.append(float(w))
            vec_list.append(temp_list)
        vec_lists.append(vec_list)

    sum_vec_list = vec_lists[0]
    if len(vec_lists) > 1:
        for i in range(len(vec_lists) - 1):
            vec_list = vec_lists[i + 1]
            for j in range(len(vec_list)):
                vec = vec_list[j]
                for k in range(len(vec)):
                    sum_vec_list[j][k] += vec_list[j][k]
    target_vec_file = []
    for j in range(len(sum_vec_list)):
        temp_list = ""
        for k in range(len(sum_vec_list[0])):
            sum_vec_list[j][k] = sum_vec_list[j][k]/len(vec_lists)
            temp_list += str(sum_vec_list[j][k]) + " "
        target_vec_file.append(temp_list.strip())

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" +
              proj_name + "_" + method + "_target_vec.txt", "w") as fp:
        for line in target_vec_file:
            fp.write(line + "\n")

    list4matrix = []
    num = len(sum_vec_list)
    for i in range(num):
        content = []
        for j in range(num):
            content.append(run2(i + 1, j + 1, sum_vec_list))
            # content.append(1.1**(50*(2 - run2(i + 1, j + 1, lines)))-1)
        list4matrix.append(content)
            # print(i)

    return np.array(list4matrix)


if __name__ == "__main__":
    proj_name = "1226_no#_rm8_ep40_dim50"  # gai
    mode = "ft"

    m = prepare_4_cluster(proj_name, mode)
    print(len(m))