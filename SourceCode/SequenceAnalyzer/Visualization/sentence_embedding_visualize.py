# -*- coding: utf-8 -*-
import os

import scipy.spatial.distance
from sklearn.manifold import TSNE
from gensim.models import FastText
from gensim.test.utils import datapath
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import hdbscan
import seaborn as sns
from scipy.spatial import Delaunay

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
random_set = 31


def visualize(model_name, method):
    vec_dir = PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/model_" + model_name + "_" + method + ".txt"
    all_vec_list = []
    print("start collecting vectors:")
    with open(vec_dir, "r") as vec_file:
        lines = vec_file.readlines()

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/PreProcess/target_num_dict.pkl", "rb+") as file:
        target_dict = pickle.load(file)
    # 读向量信息
    for line in lines[1:]:
        if int(line.split(" ")[0][9:]) not in target_dict.values():
            continue
        temp = []
        for i in line.split(" ")[1:]:
            if i.strip() == "":
                continue
            temp.append(float(i))
        all_vec_list.append(temp)
    print(len(all_vec_list))
    print("done")
    d3(all_vec_list)


def d2(all_vec_list, label_list):
    plt.clf()
    print("generate T-SNE:")
    pca = TSNE(n_components=2, random_state=2)

    print("begin process vectors")
    all_vector_TSNE_result = pca.fit_transform(all_vec_list)

    vec_num = len(all_vector_TSNE_result)

    plt.figure(1, figsize=(8, 4))
    print("start visualizing:")

    color_palette = sns.color_palette('hls', 70)
    cluster_colors = [color_palette[x] if x == 19
                    else (0.1, 0.1, 0.1)
                    for x in label_list]
    cluster_member_colors = [sns.desaturate(x, 1) for x in cluster_colors]
    # 画目标
    print(*all_vector_TSNE_result.T)
    print(cluster_member_colors)
    plt.scatter(all_vector_TSNE_result[:, 0], all_vector_TSNE_result[:, 1], c=cluster_member_colors)
    for x, y, label in zip(*all_vector_TSNE_result.T, label_list):
        plt.text(x, y, label)

    # plt.savefig(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + model_name + "_" + method + ".jpg")
    plt.show()
    plt.clf()


def d3(all_vec_list):
    mpl.rcParams['legend.fontsize'] = 20  # mpl模块载入的时候加载配置信息存储在rcParams变量中，rc_params_from_file()函数从文件加载配置信息
    font = {
        'color': 'b',
        'style': 'oblique',
        'size': 20,
        'weight': 'bold'
    }
    fig = plt.figure(figsize=(16, 12))  # 参数为图片大小
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')  # 坐标轴间比例一致

    print("generate T-SNE:")
    pca = TSNE(n_components=3, init='pca', learning_rate=20, random_state=random_set)
    print("done")
    print("begin process vectors")
    all_vector_TSNE_result = pca.fit_transform(all_vec_list)
    print("done")
    vec_num = len(all_vector_TSNE_result)
    print(all_vector_TSNE_result)
    # for i in all_vector_TSNE_result:
    ax.scatter([d[0] for d in all_vector_TSNE_result], [d[1] for d in all_vector_TSNE_result], [d[2] for d in all_vector_TSNE_result], c='r', marker='o')  # 这里marker的尺寸和z的大小成正比

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title("Scatter plot", alpha=0.6, color="b", size=25, weight='bold', backgroundcolor="y")  # 子图的title
    ax.legend(loc="upper left")  # legend的位置左上

    plt.show()


def cluster_main_hdbscan(method, model_name, proj_name, all_vec_list):
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_" + method + "_target_num_dict.pkl", "rb+") as file:
        target_dict = pickle.load(file)

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_" + method + "_distant_matrix.pkl", "rb") as matrix_file:
        matrix = pickle.load(matrix_file)

    projection = TSNE(n_components=3).fit_transform(all_vec_list)
    # cluster = DBSCAN(min_samples=4, eps=0.3, metric='precomputed')
    cluster = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, cluster_selection_epsilon=1, gen_min_span_tree=True, metric='precomputed')
    clusterer = cluster.fit(matrix)
    # cluster = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, cluster_selection_epsilon=1, gen_min_span_tree=True,
    #                           metric='cosine')
    # clusterer = cluster.fit(all_vec_list)
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

    color_palette = sns.color_palette('hls', 200)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.1, 0.1, 0.1)
                      for x in clusterer.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
    ax.scatter(*projection.T, linewidth=0, c=cluster_member_colors, alpha=1)
    plt.show()

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
    print(result_list)
    return result_list


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)

    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle

    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # print('circum_r', circum_r)

        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges

def arcsin_and_arccos(pt1, pt2):
    import math
    delta_x = pt2[0] - pt1[0]
    delta_y = pt2[1] - pt1[1]
    sin = delta_y/math.sqrt(delta_x**2 + delta_y**2)
    cos = delta_x/math.sqrt(delta_x**2 + delta_y**2)
    # return math.asin(sin), math.acos(cos)
    if sin>=0 and cos>=0:
        return math.asin(sin), math.acos(cos)
    elif sin>=0 and cos<0:
        return math.pi-math.asin(sin), math.acos(cos)
    elif sin<0 and cos<0:
        return math.pi-math.asin(sin), 2*math.pi-math.acos(cos)
    elif sin<0 and cos>=0:
        return 2*math.pi+math.asin(sin), 2*math.pi-math.acos(cos)

def top10_d2_visual_with_lable(all_vec_list, label_list, top_vec_list, top_label_list, malicious_vec_list):
    top_n = len(top_label_list)
    print("generate T-SNE:")
    pca = TSNE(n_components=2, random_state=97)

    all_cluster_vec_list = []
    all_cluster_label = []
    for vec, label in zip(all_vec_list, label_list):
        if label != -1:
            all_cluster_vec_list.append(vec)
            all_cluster_label.append(label)

    print("begin process vectors")
    all_vec_TSNE_result = pca.fit_transform(np.array(malicious_vec_list + all_cluster_vec_list + top_vec_list))
    # top_vector_TSNE_result = pca.fit_transform(np.array(top_vec_list))
    # vec_num = len(cluster_vec_TSNE_result)
    top_vector_TSNE_result = all_vec_TSNE_result[-1*top_n:]
    cluster_vec_TSNE_result = all_vec_TSNE_result[len(malicious_vec_list):-1*top_n]
    malicious_vec_TSNE_result = all_vec_TSNE_result[:len(malicious_vec_list)]
    # plt.figure(1, figsize=(8, 4))
    fig, ax = plt.subplots(figsize=(8, 8))
    print("start visualizing:")

    # ==========================================================================
    # get TSNE result for every cluster
    TSNE_result_cluster_dict = {}

    for label in range(max(all_cluster_label) + 1):
        TSNE_result_cluster_dict[label] = []
    for result, label in zip(cluster_vec_TSNE_result, all_cluster_label):
        TSNE_result_cluster_dict[label].append(result)

    color_palette = sns.color_palette('hls', top_n)
    cluster_colors = []
    for x in all_cluster_label:
        cluster_colors = cluster_colors + [color_palette[x]]
    # for x in range(10):
    #     cluster_colors = cluster_colors + [(0.1, 0.1, 0.1)]
    cluster_member_colors = [sns.desaturate(x, 1) for x in cluster_colors]
    # 画目标
    # print(*cluster_vec_TSNE_result.T)
    # print(cluster_member_colors)
    plt.scatter(malicious_vec_TSNE_result[:, 0], malicious_vec_TSNE_result[:, 1], c="red", marker="v", s=20)
    plt.scatter(cluster_vec_TSNE_result[:, 0], cluster_vec_TSNE_result[:, 1], c="blue", s=40, marker="+")
    plt.scatter(top_vector_TSNE_result[:, 0], top_vector_TSNE_result[:, 1], c="black")

    # =================================================================
    # draw edges
    from matplotlib.patches import Ellipse, Circle

    for k, v in TSNE_result_cluster_dict.items():
        if len(v) >= 4:
            points = np.array([tuple(i) for i in v])
            edges = alpha_shape(points, alpha=42, only_outer=True)
            for i, j in edges:
                # print(points[[i, j], 0], points[[i, j], 1])
                # plt.plot(points[[i, j], 0], points[[i, j], 1], color=color_palette[k])
                pass
        else:
            centroid = [0, 0]
            for i in v:
                centroid[0] += i[0]/len(v)
                centroid[1] += i[1]/len(v)
            import math
            r = math.dist(centroid, v[0])
            # print(r)

            a1, a2 = arcsin_and_arccos([1, 0], [v[0][0]-v[1][0], v[0][1]-v[1][1]])
            if r <= 100000:
                c = Ellipse(xy=tuple(centroid), width=2*r, height=1.2*r, alpha=1, angle=180 * a2/math.pi,  color=color_palette[k], fill=False)  # 圆
                ax.add_artist(c)
                pass

    # for x, y, label in zip(*cluster_vec_TSNE_result.T, np.array(all_cluster_label)):
    #     plt.text(x, y, label)
    # 标角标
    for x, y, label in zip(*top_vector_TSNE_result.T, np.array(top_label_list)):
        plt.text(x, y, label)

    # Z = []
    # for i in range(len(cluster_vec_TSNE_result[:, 0])):
    #     temp = []
    #     for j in range(len(cluster_vec_TSNE_result[:, 0])):
    #         if i != j:
    #             temp.append(-1)
    #         else:
    #             temp.append(all_cluster_label[i])
    #     Z.append(temp)

    # c = plt.contour(cluster_vec_TSNE_result[:, 0], cluster_vec_TSNE_result[:, 1], Z, levels=[i for i in range(-1, 8)], colors='black', linewidth=.05)
    # plt.savefig(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + model_name + "_" + method + ".jpg")

    plt.show()
    plt.clf()
def top10_d2_visual(all_vec_list, label_list, top_vec_list, top_label_list, malicious_vec_list):
    top_n = len(top_label_list)
    print("generate T-SNE:")
    pca = TSNE(n_components=2, random_state=7)

    all_cluster_vec_list = []
    all_cluster_label = []
    for vec, label in zip(all_vec_list, label_list):
        if label != -1:
            all_cluster_vec_list.append(vec)
            all_cluster_label.append(label)

    print("begin process vectors")
    all_vec_TSNE_result = pca.fit_transform(np.array(malicious_vec_list + all_cluster_vec_list + top_vec_list))
    # top_vector_TSNE_result = pca.fit_transform(np.array(top_vec_list))
    # vec_num = len(cluster_vec_TSNE_result)
    top_vector_TSNE_result = all_vec_TSNE_result[-1*top_n:]
    cluster_vec_TSNE_result = all_vec_TSNE_result[len(malicious_vec_list):-1*top_n]
    malicious_vec_TSNE_result = all_vec_TSNE_result[:len(malicious_vec_list)]
    # plt.figure(1, figsize=(8, 4))
    fig, ax = plt.subplots(figsize=(8, 8))
    print("start visualizing:")

    # ==========================================================================
    # get TSNE result for every cluster
    TSNE_result_cluster_dict = {}

    for label in range(max(all_cluster_label) + 1):
        TSNE_result_cluster_dict[label] = []
    for result, label in zip(cluster_vec_TSNE_result, all_cluster_label):
        TSNE_result_cluster_dict[label].append(result)

    color_palette = sns.color_palette('hls', top_n)
    cluster_colors = []
    for x in all_cluster_label:
        cluster_colors = cluster_colors + [color_palette[x]]
    # for x in range(10):
    #     cluster_colors = cluster_colors + [(0.1, 0.1, 0.1)]
    cluster_member_colors = [sns.desaturate(x, 1) for x in cluster_colors]
    # 画目标
    # print(*cluster_vec_TSNE_result.T)
    # print(cluster_member_colors)
    plt.scatter(malicious_vec_TSNE_result[:, 0], malicious_vec_TSNE_result[:, 1], c="red", marker="v", s=20)
    plt.scatter(cluster_vec_TSNE_result[:, 0], cluster_vec_TSNE_result[:, 1], c="blue", s=40, marker="+")
    plt.scatter(top_vector_TSNE_result[:, 0], top_vector_TSNE_result[:, 1], c="black")

    # =================================================================
    # draw edges
    from matplotlib.patches import Ellipse, Circle

    for k, v in TSNE_result_cluster_dict.items():
        if len(v) >= 4:
            points = np.array([tuple(i) for i in v])
            edges = alpha_shape(points, alpha=42, only_outer=True)
            for i, j in edges:
                # print(points[[i, j], 0], points[[i, j], 1])
                # plt.plot(points[[i, j], 0], points[[i, j], 1], color=color_palette[k])
                pass
        else:
            centroid = [0, 0]
            for i in v:
                centroid[0] += i[0]/len(v)
                centroid[1] += i[1]/len(v)
            import math
            r = math.dist(centroid, v[0])
            # print(r)

            a1, a2 = arcsin_and_arccos([1, 0], [v[0][0]-v[1][0], v[0][1]-v[1][1]])
            if r <= 100000:
                # c = Ellipse(xy=tuple(centroid), width=2*r, height=1.2*r, alpha=1, angle=180 * a2/math.pi,  color=color_palette[k], fill=False)  # 圆
                # ax.add_artist(c)
                pass

    # for x, y, label in zip(*cluster_vec_TSNE_result.T, np.array(all_cluster_label)):
    #     plt.text(x, y, label)
    # 标角标
    # for x, y, label in zip(*top_vector_TSNE_result.T, np.array(top_label_list)):
    #     plt.text(x, y, label)

    # Z = []
    # for i in range(len(cluster_vec_TSNE_result[:, 0])):
    #     temp = []
    #     for j in range(len(cluster_vec_TSNE_result[:, 0])):
    #         if i != j:
    #             temp.append(-1)
    #         else:
    #             temp.append(all_cluster_label[i])
    #     Z.append(temp)

    # c = plt.contour(cluster_vec_TSNE_result[:, 0], cluster_vec_TSNE_result[:, 1], Z, levels=[i for i in range(-1, 8)], colors='black', linewidth=.05)
    # plt.savefig(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + model_name + "_" + method + ".jpg")

    plt.show()
    plt.clf()


if __name__ == "__main__":
    visualize("test", "ft")