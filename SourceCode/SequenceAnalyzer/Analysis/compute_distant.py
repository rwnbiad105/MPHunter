import os
import numpy as np
from scipy import spatial

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(label1, label2, model_name, mode):
    list1 = []
    list2 = []

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + model_name + "_" + mode + "_model.txt", "r") as fp:
        lines = fp.readlines()
        for line in lines:
            if line.split()[0] == "__label__" + str(label1):
                for i in line.split()[1:]:
                    list1.append(float(i))
            if line.split()[0] == "__label__" + str(label2):
                for i in line.split()[1:]:
                    list2.append(float(i))

        result1 = np.array(list1)
        result2 = np.array(list2)

        print(list1)
        print(list2)
        cosangle = result1.dot(result2) / (np.linalg.norm(result1) * np.linalg.norm(result2))
        print(cosangle)


def run2(label1, label2, lines):
    list1 = lines[label1 - 1]
    list2 = lines[label2 - 1]

    result1 = np.array(list1)
    result2 = np.array(list2)
    # print(result1)
    # print(result2)
    # cosangle = result1.dot(result2) / (np.linalg.norm(result1) * np.linalg.norm(result2))
    cosangle = spatial.distance.cosine(result2, result1)
    # print(cosangle)
    return cosangle


def run3(line1, line2):

    list1 = []
    list2 = []
    for w in line1.split():
        list1.append(float(w))
    for w in line2.split():
        list2.append(float(w))
    result1 = np.array(list1)
    result2 = np.array(list2)
    # print(result1)
    # print(result2)
    # cosangle = result1.dot(result2) / (np.linalg.norm(result1) * np.linalg.norm(result2))
    cosangle = spatial.distance.cosine(result2, result1)
    # print(cosangle)
    return cosangle

if __name__ == "__main__":
    model_name = "0113_6000-pkg_new_updated_pypi_setup_c.x"
    method = "ft"
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + model_name + "/" + model_name + "_" + method + "_target_vec.txt", "r") as fp:
        lines = fp.readlines()
    print(run2(645, 1160, lines))