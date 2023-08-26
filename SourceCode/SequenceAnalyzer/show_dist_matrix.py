import pickle

from SequenceAnalyzer.Analysis.compute_distant import run2
from scipy.spatial.distance import cosine
import os
import numpy as np
from scipy import spatial

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_into_file(proj_name, method):
    print(PROJECT_DIR)
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_" + method + "_distant_matrix.pkl", "rb") as fp:
        dist_matrix = pickle.load(fp)
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_" + method + "_distant_matrix.txt", "w+") as op:
        for i in dist_matrix:
            for j in i:
                op.write(str(j) + " ")
            op.write("\n")


def just_print(proj_name, method):
    print(PROJECT_DIR)
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + proj_name +
              "_" + method + "_distant_matrix.pkl", "rb") as fp:
        # _target_num_dict.pkl
        # _label_file_dict.pkl
        # _distant_matrix.pkl
        dist_matrix = pickle.load(fp)

    for i in dist_matrix:
        temp = ""
        for j in i:
            temp += str(j) + ", "
        print(temp)

    # for i, j in dist_matrix.items():
    #     temp = ""
    #     temp += "(" + str(i) + "," + str(j) + ");"
    #     print(temp)


if __name__ == "__main__":
    proj_name = "final-1_0421_2_2_cfg_setup_c.x"  # gai
    mode = "ft"

    just_print(proj_name, mode)
