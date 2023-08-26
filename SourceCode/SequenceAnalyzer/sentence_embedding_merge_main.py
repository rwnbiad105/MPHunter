import pickle, os

from SequenceAnalyzer.Analysis.compute_distant import run3
from scipy.spatial.distance import cosine
import os
import numpy as np
from scipy import spatial

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def merge_distance_matrix(proj_a, proj_b, output_proj, method):
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + output_proj + "/" + output_proj
              + "_" + method + "_target_vec.txt", "r") as fp:
        lines = fp.readlines()
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_a + "/" + proj_a
              + "_" + method + "_distant_matrix.pkl", "rb") as fp_a:
        dict_a = pickle.load(fp_a)
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_b + "/" + proj_b
              + "_" + method + "_distant_matrix.pkl", "rb") as fp_b:
        dict_b = pickle.load(fp_b)
    
    num = len(lines)
    label_num_a = len(dict_a)
    label_num_b = len(dict_b)
    list4matrix = []
    for i in range(num):
        if i < label_num_a:
            content = list(dict_a[i])
            for j in range(label_num_a, num):
                line1 = lines[i]
                line2 = lines[j]
                content.append(run3(line1, line2))
        else:
            content = []
            for j in range(label_num_a):
                line1 = lines[i]
                line2 = lines[j]
                content.append(run3(line1, line2))
            content += list(dict_b[i - label_num_a])
        list4matrix.append(content)

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + output_proj + "/" + output_proj
              + "_" + method + "_distant_matrix.pkl", "wb+") as fp:
        pickle.dump(np.array(list4matrix), fp)

    # for i in list4matrix:
    #     print(i)


def merge_vec(proj_a, proj_b, output_proj, method):
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_a + "/" + proj_a +
              "_" + method + "_target_num_dict.pkl", "rb+") as target_a:
        target_dict_a = pickle.load(target_a)
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_b + "/" + proj_b +
              "_" + method + "_target_num_dict.pkl", "rb+") as target_b:
        target_dict_b = pickle.load(target_b)

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_a + "/" + proj_a +
              "_" + method + "_label_file_dict.pkl", "rb+") as target_a:
        label_dict_a = pickle.load(target_a)
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_b + "/" + proj_b +
              "_" + method + "_label_file_dict.pkl", "rb+") as target_b:
        label_dict_b = pickle.load(target_b)

    target_num = len(label_dict_a)
    label_num = target_dict_a[target_num]

    output_target_dict = target_dict_a
    for v, k in target_dict_b.items():
        output_target_dict[v + target_num] = k + label_num

    output_label_dict = label_dict_a
    for v, k in label_dict_b.items():
        output_label_dict[v + target_num] = k

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + output_proj + "/" + output_proj +
              "_" + method + "_target_num_dict.pkl", "wb+") as target_file:
        pickle.dump(output_target_dict, target_file)
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + output_proj + "/" + output_proj +
              "_" + method + "_label_file_dict.pkl", "wb+") as label_file:
        pickle.dump(output_label_dict, label_file)

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_a + "/" + proj_a +
              "_" + method + "_target_seq.txt", "r") as seq_file_a:
        target_seq_dict_a = seq_file_a.readlines()
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_b + "/" + proj_b +
              "_" + method + "_target_seq.txt", "r") as seq_file_b:
        target_seq_dict_b = seq_file_b.readlines()
    output_target_seq_lines = target_seq_dict_a + target_seq_dict_b
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + output_proj + "/" + output_proj +
              "_" + method + "_target_seq.txt", "w+") as seq_file:
        for line in output_target_seq_lines:
            seq_file.write(line)

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_a + "/" + proj_a +
              "_" + method + "_target_vec.txt", "r") as vec_file_a:
        target_vec_dict_a = vec_file_a.readlines()
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_b + "/" + proj_b +
              "_" + method + "_target_vec.txt", "r") as vec_file_b:
        target_vec_dict_b = vec_file_b.readlines()
    output_target_vec_lines = target_vec_dict_a + target_vec_dict_b
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + output_proj + "/" + output_proj +
              "_" + method + "_target_vec.txt", "w+") as vec_file:
        for line in output_target_vec_lines:
            vec_file.write(line)


def merge_main(proj_a, proj_b, output_proj, model_name, method):
    if not os.path.exists(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + output_proj):
        os.mkdir(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + output_proj)
    cmd = "cp " + PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + model_name + \
          "/Set_API_count.txt " + PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + \
          output_proj + "/Set_API_count.txt"
    os.system(cmd)

    merge_vec(proj_a, proj_b, output_proj, method)
    merge_distance_matrix(proj_a, proj_b, output_proj, method)


if __name__ == "__main__":
    proj_a = "0222_6000-pkg_new_updated_pypi_setup_c.x"
    proj_b = "0304_3440-pkg_new_updated_pypi_setup_c.x"
    output_proj = "0304_9440-pkg_new_updated_pypi_setup_c.x_merged"

    model_name = "0222_12000-pkg_for_corpus_c.x"
    method = "ft"
    merge_main(proj_a, proj_b, output_proj, model_name, method)