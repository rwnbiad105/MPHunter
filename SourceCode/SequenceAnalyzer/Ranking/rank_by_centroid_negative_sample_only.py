import os
import math
import numpy as np
from scipy import spatial
import heapq
import pickle

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/DataShare/ApiSeq_and_Result/"


class ranker():
    def __init__(self):
        self.vec_dict = {}
        self.clust_heart_dict = {}
        self.clust_num_dict = {}
        self.rank_dict = {}
        self.dist_dict = {}
        self.biggest_clust_dist_dict = {}
        self.var_dict = {}
        self.weight_dict = {}
        self.label_list = []
        self.negative_vec_dict = {}
        self.seq_dist_dict = {}

    def get_heart(self, proj_name, negative_proj_name, negative_label_list):
        # 读取所有的negative_target句向量
        with open(PROJECT_DIR + "Result/SentenceEmbedding/" + negative_proj_name + "/" + negative_proj_name + "_ft_target_vec.txt",
                  "r") as file:
            target_vec_lines = file.readlines()
        i = 0

        for line in target_vec_lines:
            i = i + 1
            vec_list = []
            line = line.split(" ")
            for num in line:
                if num.strip() == "":
                    continue
                vec_list.append(float(num))
            self.negative_vec_dict[i] = vec_list
        # 读取所有的target句向量
        with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_ft_target_vec.txt",
                  "r") as file:
            target_vec_lines = file.readlines()
        i = 0

        for line in target_vec_lines:
            i = i + 1
            vec_list = []
            line = line.split(" ")
            for num in line:
                if num.strip() == "":
                    continue
                vec_list.append(float(num))
            self.vec_dict[i] = vec_list

        # 求质心， 求簇内向量和
        for label, vec in zip(negative_label_list, self.negative_vec_dict.values()):
            if label == -1:
                continue
            if label in self.clust_num_dict.keys():
                self.clust_num_dict[label] += 1
            else:
                self.clust_num_dict[label] = 1

            if label in self.clust_heart_dict.keys():
                current_vec = self.clust_heart_dict[label]
            else:
                self.clust_heart_dict[label] = vec
                continue

            new_vec = []
            for a, b in zip(current_vec, vec):
                new_vec.append(a + b)
            self.clust_heart_dict[label] = new_vec
        print("clust_num_dict:")
        print(self.clust_num_dict.items())

        # 处以簇内的向量个数，求平均
        for label, num in self.clust_num_dict.items():
            current_vec = self.clust_heart_dict[label]
            new_vec = []
            for i in current_vec:
                new_vec.append(i / num)
            self.clust_heart_dict[label] = new_vec
            self.label_list.append(label)

        # 算每个向量与每个质心的距离列表
        for label_num, vec in self.vec_dict.items():
            dist_list = []
            for label, heart_vec in sorted(self.clust_heart_dict.items()):
                cos_dist = spatial.distance.cosine(vec, heart_vec)
                dist_list.append(cos_dist)
            self.dist_dict[label_num] = dist_list

        # for k, v in self.clust_heart_dict.items():
        #     print(k, v)
        return self.clust_heart_dict

    def ranking_biggest_top_weight(self):
        max_n_clust = heapq.nlargest(len(self.clust_num_dict), self.clust_num_dict.items(), key=lambda d: d[1])
        print("top n biggest clust:")
        print(max_n_clust)
        biggest_list = []
        for i in max_n_clust:
            biggest_list.append(int(i[0]))

        # 算每个向量与每个质心的距离列表
        for label_num, vec in self.vec_dict.items():
            dist_list = []
            for label, heart_vec in self.clust_heart_dict.items():
                ##############
                # if label == 11:
                #     continue
                ##############
                if label in biggest_list:
                    cos_dist = spatial.distance.cosine(vec, heart_vec)
                    dist_list.append(cos_dist)
            self.biggest_clust_dist_dict[label_num] = dist_list

        # 算向量的rank得分， 分高的向量是我们想要的
        for label_num, dist_list in self.biggest_clust_dist_dict.items():
            score_list = heapq.nsmallest(1, dist_list)
            score = 0
            # print(dist_list)
            # print(score_list)
            # print(self.weight_dict)
            for i in score_list:
                score += i
            self.rank_dict[label_num] = score

        print(self.rank_dict)
        return self.rank_dict

    def ranking_biggest_weight(self):
        max_n_clust = heapq.nlargest(5, self.clust_num_dict.items(), key=lambda d: d[1])
        print(max_n_clust)
        biggest_list = []
        for i in max_n_clust:
            biggest_list.append(int(i[0]))

        # 算每个向量与每个质心的距离列表
        for label_num, vec in self.vec_dict.items():
            dist_list = []
            for label, heart_vec in self.clust_heart_dict.items():
                if label in biggest_list:
                    cos_dist = spatial.distance.cosine(vec, heart_vec)
                    dist_list.append(cos_dist)
            self.biggest_clust_dist_dict[label_num] = dist_list

        # 算每个簇的权重, 簇越大权重越低
        whole = len(self.vec_dict)
        for label, num in self.clust_num_dict.items():
            if label in biggest_list:
                self.weight_dict[label] = math.log(whole / num, 10)

        # 算向量的rank得分， 分高的向量是我们想要的
        print(self.weight_dict)
        for label_num, dist_list in self.biggest_clust_dist_dict.items():
            # score_list = heapq.nsmallest(3, dist_list)
            score = 0
            print(dist_list)
            for i in dist_list:
                score += i * self.weight_dict[biggest_list[dist_list.index(i)]]
            self.rank_dict[label_num] = score

        print(self.rank_dict)
        return self.rank_dict

    def ranking_by_seq(self):

        # 算每个target向量与每个negative target向量的距离列表
        for label_num, vec in self.vec_dict.items():
            dist_list = []
            for label, negative_vec in self.negative_vec_dict.items():
                cos_dist = spatial.distance.cosine(vec, negative_vec)
                dist_list.append(cos_dist)
            self.seq_dist_dict[label_num] = dist_list

        # 算向量的rank得分， 分低的向量是我们想要的
        print(self.weight_dict)
        for label_num, dist_list in self.seq_dist_dict.items():
            score_list = heapq.nsmallest(1, dist_list)
            score = score_list[0]
            self.rank_dict[label_num] = score

        print(self.rank_dict)
        return self.rank_dict