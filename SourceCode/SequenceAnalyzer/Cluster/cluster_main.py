import math

import hdbscan
import os
import pickle
import matplotlib as mpl
import numpy
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def remove_dup(proj_name, method):
    with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" +
              proj_name + "_" + method + "_target_vec.txt", "r") as file:
        target_vec_lines = file.readlines()
    with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" +
              proj_name + "_" + method + "_target_seq.txt", "r") as file:
        target_seq_lines = file.readlines()
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" +
              proj_name + "_" + method + "_target_num_dict.pkl", "rb+") as file:
        target_dict = pickle.load(file)

    print("seq distant start")
    with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_ft_distant_matrix.pkl",
              "rb+") as file:
        dist_matrix = pickle.load(file)
        dist_matrix = dist_matrix.tolist()
    seq_dup_dict = {}
    label_dup_dict = {}
    line_label = 0
    for line in target_seq_lines:
        line_label += 1
        if line in seq_dup_dict.keys():
            seq_dup_dict[line] += 1
            label_dup_dict[line].append(line_label)
        else:
            seq_dup_dict[line] = 1
            label_dup_dict[line] = [line_label]

    for dup_list in label_dup_dict.values():
        for dup_label in dup_list:
            target_vec_lines[dup_label - 1] = [404]
            dist_matrix[dup_label - 1] = [404]
    dist_matrix = list(filter(lambda l: l != [404], dist_matrix))
    target_vec_lines = list(filter(lambda l: l != [404], target_vec_lines))
    new_dist_matrix = []
    for line in dist_matrix:
        for dup_list in label_dup_dict.values():
            for dup_label in dup_list[1:]:
                line[dup_label - 1] = 404
        new_dist_matrix.append(list(filter(lambda e: e != 404, line)))

    with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" +
              proj_name + "_" + method + "_target_vec_dup.txt", "wb") as file:
        pickle.dump(target_vec_lines,file)
    with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" +
              proj_name + "_" + method + "_target_seq.txt_dup", "wb") as file:
        pickle.dump(target_seq_lines,file)
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" +
              proj_name + "_" + method + "_target_num_dict_dup.pkl", "wb") as file:
        pickle.dump(target_dict, file)
    return target_seq_lines, target_vec_lines, new_dist_matrix, label_dup_dict



def cluster_main_dbscan(method, proj_name):
    vec_dir = PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_" + method + "_target_vec.txt"
    all_vec_list = []
    print("start collecting vectors:")
    with open(vec_dir, "r") as vec_file:
        lines = vec_file.readlines()

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_" + method + "_target_num_dict.pkl", "rb+") as file:
        target_dict = pickle.load(file)
    # 读向量信息
    for line in lines[:]:
        temp = []
        for i in line.split(" ")[:]:    # for i in line.split(" ")[1:]:
            if i.strip() == "":
                continue
            temp.append(float(i))
        all_vec_list.append(temp)

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_" + method + "_distant_matrix.pkl", "rb") as matrix_file:
        matrix = pickle.load(matrix_file)

    projection = TSNE(n_components=3).fit_transform(all_vec_list)
    cluster = DBSCAN(min_samples=20, eps=0.1, metric='cosine')
    clusterer = cluster.fit(all_vec_list)
    # cluster = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=5, cluster_selection_epsilon=0.5, gen_min_span_tree=True, metric='precomputed')
    # clusterer = cluster.fit(matrix)
    mpl.rcParams['legend.fontsize'] = 20  # mpl模块载入的时候加载配置信息存储在rcParams变量中，rc_params_from_file()函数从文件加载配置信息
    font = {
        # 'color': 'b',
        'style': 'oblique',
        'size': 20,
        'weight': 'bold'
    }
    fig = plt.figure(figsize=(16, 12))  # 参数为图片大小
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')  # 坐标轴间比例一致

    color_palette = sns.color_palette('hls', 50)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.1, 0.1, 0.1)
                      for x in clusterer.labels_]
    # cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
    # ax.scatter(*projection.T, linewidth=0, c=cluster_member_colors, alpha=1)
    ax.scatter(*projection.T, linewidth=0, c=cluster_colors, alpha=1)
    # plt.show()

    j = 0
    result_bad_list = []
    result_noisy_list = []
    result_good_list = []

    # HDBSCAN output
    # for sat, sat2 in zip(cluster.labels_, clusterer.probabilities_):
    #     j = j + 1
    #     if sat == -1:
    #         result_noisy_list.append(target_dict[j])
    #     elif sat2 < 0.5:
    #         result_bad_list.append(target_dict[j])
    #     else:
    #         result_good_list.append(target_dict[j])

    # DBSCAN output
    for sat in clusterer.labels_:
        j = j + 1
        if sat < 0:
            result_noisy_list.append(target_dict[j])
        else:
            result_good_list.append(target_dict[j])
    # print(target_dict)
    # print(result_good_list)
    # print(result_noisy_list)
    # print(result_bad_list)
    # print(cluster.labels_)
    # # print(cluster.probabilities_)
    return result_noisy_list


def cluster_main_hdbscan(method, proj_name, min_cluster_size=2, min_samples=100):
    # vec_dir = PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "_" + method + "_model.txt"
    vec_dir = PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" +\
              proj_name + "_" + method + "_target_vec.txt"
    all_vec_list = []
    print("start collecting vectors:")
    with open(vec_dir, "r") as vec_file:
        lines = vec_file.readlines()

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" +
              proj_name + "_" + method + "_target_num_dict.pkl", "rb+") as file:
        target_dict = pickle.load(file)

    # 读向量信息
    # new_d = {v: k for k, v in target_dict.items()}
    # for line in lines[1:]:
    #     if int(line.split(" ")[0][9:]) not in new_c.keys():
    #         continue

    for line in lines[:]:
        temp = []
        for i in line.split(" ")[:]:    # for i in line.split(" ")[1:]:
            if i.strip() == "":
                continue
            temp.append(float(i))
        all_vec_list.append(temp)
    data = numpy.array(all_vec_list)

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" +
              proj_name + "_" + method + "_distant_matrix.pkl", "rb") as matrix_file:
        matrix = pickle.load(matrix_file)

    # vec_num = len(lines)
    # min_sampels = int(math.pow(vec_num/100, 3))
    # if min_sampels < 2:
    #     min_sampels = 2
    # if min_sampels > 100:
    #     min_sampels = 100
    # projection = TSNE(n_components=3).fit_transform(data)
    # # cluster = DBSCAN(min_samples=4, eps=0.3, metric='precomputed')
    # min_cluster_size = 2
    # print("min_sampels:" + str(min_sampels))
    # print("min_cluster_size:" + str(min_cluster_size))

    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                              gen_min_span_tree=True, metric='precomputed')
    clusterer = cluster.fit(matrix)
    # cluster = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, cluster_selection_epsilon=1, gen_min_span_tree=True,
    #                           metric='cosine')
    # clusterer = cluster.fit(all_vec_list)

    # #################################################to show image##################################################
    # mpl.rcParams['legend.fontsize'] = 20  # mpl模块载入的时候加载配置信息存储在rcParams变量中，rc_params_from_file()函数从文件加载配置信息
    # font = {
    #     # 'color': 'b',
    #     'style': 'oblique',
    #     'size': 20,
    #     'weight': 'bold'
    # }
    # fig = plt.figure(figsize=(16, 12))  # 参数为图片大小
    # ax = fig.add_subplot(projection='3d')
    # ax.set_aspect('auto')  # 坐标轴间比例一致
    #
    # color_palette = sns.color_palette('hls', 200)
    # cluster_colors = [color_palette[x] if x >= 0
    #                   else (0.1, 0.1, 0.1)
    #                   for x in clusterer.labels_]
    # cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
    # ax.scatter(*projection.T, linewidth=0, c=cluster_member_colors, alpha=1)
    # plt.show()
    # #################################################to show image#################################################

    j = 0
    # result_bad_list = []
    result_noisy_list = []
    result_good_list = []

    # HDBSCAN output
    for sat, sat2 in zip(cluster.labels_, clusterer.probabilities_):
        j = j + 1
        if sat == -1:
            result_noisy_list.append(target_dict[j])
        # elif sat2 < 0.5:
        #     result_bad_list.append(target_dict[j])
        else:
            result_good_list.append(target_dict[j])

    result_list = []
    result_list.append(result_good_list)
    # result_list.append(result_bad_list)
    result_list.append(result_noisy_list)
    result_list.append(cluster.labels_)
    result_list.append(cluster.probabilities_)
    # print(result_list)
    return result_list


def cluster_main_cc(method, proj_name, similarity_threshold):

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" +
              proj_name + "_" + method + "_target_num_dict.pkl", "rb+") as file:
        target_dict = pickle.load(file)

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" +
              proj_name + "_" + method + "_distant_matrix.pkl", "rb") as matrix_file:
        matrix = pickle.load(matrix_file)


    adjacency_mask = []
    for line in matrix:
        # print(line)
        masked_line = []
        for elem in line:
            if 1 - elem >= similarity_threshold:
                masked_line.append(1)
            else:
                masked_line.append(0)
        print(masked_line)
        adjacency_mask.append(masked_line)

    # adjacency_mask = numpy.array(adjacency_mask)
    graph = csr_matrix(adjacency_mask)
    print(graph)
    cluster, cluster_labels = connected_components(graph, connection='strong')

    print(cluster)
    print(cluster_labels)
    print(len(cluster_labels))
    j = 0
    result_noisy_list = []
    result_good_list = []

    # HDBSCAN output
    for sat in cluster_labels:
        j = j + 1
        if sat == -1:
            result_noisy_list.append(target_dict[j])
        else:
            result_good_list.append(target_dict[j])

    result_list = []
    result_list.append(result_good_list)
    result_list.append(result_noisy_list)
    result_list.append(cluster_labels)

    return result_list


if __name__ == "__main__":
    cluster_main_cc("ft", "final-1_0421_2_cfg_setup_c.x_merged", 0.)