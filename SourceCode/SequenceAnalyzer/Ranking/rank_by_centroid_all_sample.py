import os
import math
import numpy as np
from scipy import spatial
import heapq
import pickle
import copy

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
        self.negative_seq_dist_dict = {}
        self.seq_dist_dict = {}
        self.weight_dict_2 = {}

    def get_heart(self, proj_name, negative_proj_name, label_list):
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
        for label, vec in zip(label_list, self.vec_dict.values()):
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

    def ranking_biggest_top_centroid(self):
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

    def ranking_by_p_centroid_n_seq(self):
        # 算每个target向量与每个negative target向量的距离列表
        for label_num, vec in self.vec_dict.items():
            dist_list = []
            for label, negative_vec in self.negative_vec_dict.items():
                cos_dist = spatial.distance.cosine(vec, negative_vec)
                dist_list.append(cos_dist)
            self.seq_dist_dict[label_num] = dist_list

        # 算每个向量与每个质心的距离列表
        max_n_clust = heapq.nlargest(len(self.clust_heart_dict.keys()), self.clust_num_dict.items(), key=lambda d: d[1])
        print(max_n_clust)
        biggest_list = []
        for i in max_n_clust:
            biggest_list.append(int(i[0]))

        for label_num, vec in self.vec_dict.items():
            dist_list = []
            for label, heart_vec in self.clust_heart_dict.items():
                if label in biggest_list:
                    cos_dist = spatial.distance.cosine(vec, heart_vec)
                    dist_list.append(cos_dist)
            self.dist_dict[label_num] = dist_list

        # 算每个簇的权重, 簇越大权重越低
        # 那是以前的做法
        # 现在算每个簇的权重, 簇越大权重越高
        whole = len(self.vec_dict)
        for label, num in self.clust_num_dict.items():
            self.weight_dict[label] = math.log(whole / num, math.e)  # old one
            # self.weight_dict[label] = math.log(num, math.e)

        # 算向量的rank得分， 分高的向量是我们想要的
        for label_num, dist_list in self.dist_dict.items():
            score_list = heapq.nsmallest(61, dist_list)
            score = 0
            # print(dist_list)
            # print(score_list)
            # print(self.weight_dict)
            for i in score_list:
                score += i * self.weight_dict[biggest_list[dist_list.index(i)]]
            self.rank_dict[label_num] = score

        # 对字典中的值进行归一化处理
        max_value = max(self.rank_dict.values())
        min_value = min(self.rank_dict.values())
        for k, v in self.rank_dict.items():
            self.rank_dict[k] = (v - min_value)/(max_value + min_value)

        # 算向量的rank得分， 分低的向量是我们想要的
        print(self.weight_dict)
        for label_num, dist_list in self.seq_dist_dict.items():
            score_list = heapq.nsmallest(1, dist_list)
            score = score_list[0]
            self.rank_dict[label_num] += (2 - score)/2

        print(self.rank_dict)
        return self.rank_dict, self.seq_dist_dict

    def ranking_by_p_seq_n_seq(self, proj_name, label_list):
        # 算每个target向量与每个negative target向量的距离列表
        for label_num, vec in self.vec_dict.items():
            dist_list = []
            for label, negative_vec in self.negative_vec_dict.items():
                cos_dist = spatial.distance.cosine(vec, negative_vec)
                dist_list.append(cos_dist)
            self.seq_dist_dict[label_num] = dist_list

        with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_ft_target_vec.txt", "r") as file:
            target_vec_lines = file.readlines()
        i = 0
        with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_ft_target_seq.txt", "r") as file:
            target_seq_lines = file.readlines()
        i = 0

        seq_too_long_list = []
        count = 1
        for j in target_seq_lines:
            if len(j.split(" ")) > 30:
                seq_too_long_list.append(count)
            count += 1

        # 读取所有的target句向量
        for line in target_vec_lines:
            i = i + 1
            vec_list = []
            line = line.split(" ")
            for num in line:
                if num.strip() == "":
                    continue
                vec_list.append(float(num))
            self.vec_dict[i] = vec_list

        # 制作数量dict
        for label, vec in zip(label_list, self.vec_dict.values()):
            if label in self.clust_num_dict.keys():
                self.clust_num_dict[label] += 1
            else:
                self.clust_num_dict[label] = 1

        # 算每个簇的权重, 簇越大权重越高
        whole = len(self.vec_dict)
        for label, num in self.clust_num_dict.items():
            # self.weight_dict[label] = math.log(whole/num, 10)
            self.weight_dict[label] = math.log(num, 10)
        max_value = max(self.weight_dict.values())
        min_value = min(self.weight_dict.values())
        for k,v in self.weight_dict.items():
            self.weight_dict[k] = (v - min_value) / (max_value + min_value)
        self.weight_dict[-1] = 1

        whole = len(self.vec_dict)
        for label, num in self.clust_num_dict.items():
            self.weight_dict_2[label] = math.log(whole/num, math.e)
        max_value = max(self.weight_dict_2.values())
        min_value = min(self.weight_dict_2.values())
        for k,v in self.weight_dict_2.items():
            self.weight_dict_2[k] = (v - min_value) / (max_value + min_value)
        self.weight_dict_2[-1] = 1

        print(self.weight_dict)
        print(self.weight_dict_2)

        # 算每个target向量与向量的距离列表
        # if os.path.exists(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_rank_dist_dict.pkl"):
        #     with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_rank_dist_dict.pkl",
        #               "rb+") as file:
        #         self.dist_dict = pickle.load(file)
        # else:
        print("seq distant start")
        # for label_num, vec in self.vec_dict.items():
        #     dist_list = []
        #     for label_num2, vec2 in self.vec_dict.items():
        #         cos_dist = spatial.distance.cosine(vec, vec2)
        #         # euclid_dist = spatial.distance.euclidean(vec, vec2)
        #         dist_list.append(cos_dist)
        #     self.dist_dict[label_num] = dist_list
        with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_ft_distant_matrix.pkl",
                "rb+") as file:
            dist_matrix = pickle.load(file)
        dist_matrix = dist_matrix.tolist()
        count_temp = 0
        for i in dist_matrix:
            count_temp += 1
            self.dist_dict[count_temp] = i
        print("seq distant end")

        # 算向量的rank得分， 分高的向量是我们想要的
        # print(self.weight_dict)
        for label_num, dist_list in self.dist_dict.items():
            # score_list = heapq.nsmallest(3, dist_list)
            score = 0
            # print(dist_list)
            for i in dist_list:
                score += i * self.weight_dict[label_list[dist_list.index(i)]]

            # result_list = []
            # for i in dist_list:
            #     if i == 0:
            #         continue
            #     else:
            #         result_list.append(i)
            # score = 1 - min(result_list)
            if label_num in seq_too_long_list:
                score = score*0.01
            score = score * self.weight_dict_2[label_list[label_num - 1]]
            self.rank_dict[label_num] = score

        # 对字典中的值进行归一化处理
        max_value = max(self.rank_dict.values())
        min_value = min(self.rank_dict.values())
        for k, v in self.rank_dict.items():
            self.rank_dict[k] = (v - min_value)/(max_value + min_value)

        # 算向量的rank得分， 分低的向量是我们想要的
        print(self.weight_dict)
        for label_num, dist_list in self.seq_dist_dict.items():
            score_list = heapq.nsmallest(1, dist_list)
            score = score_list[0]
            self.rank_dict[label_num] += (2 - score)/2

        print(self.rank_dict)
        return self.rank_dict, self.seq_dist_dict


    def ranking_by_p_seq_one_n_seq(self, proj_name, label_list, merge_len):
        # ========================================================================================================= #
        # 找过长的序列
        with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" +
                  proj_name + "_ft_target_seq.txt", "r") as file:
            target_seq_lines = file.readlines()
        # seq_too_long_list = []
        # count = 1
        # for j in target_seq_lines:
        #     if len(j.split(" ")) > 30000:
        #         seq_too_long_list.append(count)
        #     count += 1


        # ========================================================================================================= #
        # 读取所有的target句向量
        dup_dict = {}
        with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" +
                  proj_name + "_ft_target_vec.txt", "r") as file:
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

        # ========================================================================================================= #
        # 制作数量dict
        for label in label_list:
            if label in self.clust_num_dict.keys():
                self.clust_num_dict[label] += 1
            else:
                self.clust_num_dict[label] = 1
        value_list_list = []
        for i in range(len(self.clust_num_dict)-1):
            value_list_list.append([2])

        # ========================================================================================================= #
        # 算每个target向量与向量的距离列表
        print("seq distant start")
        with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_ft_distant_matrix.pkl",
                "rb+") as file:
            dist_matrix = pickle.load(file)
        dist_matrix = dist_matrix.tolist()
        count_temp = 0
        for i in dist_matrix:
            count_temp += 1
            self.dist_dict[count_temp] = i
        print("seq distant end")

        # 算每个簇的权重, 簇越大权重越高
        # whole = len(self.vec_dict)
        # for label, num in self.clust_num_dict.items():
        #     # self.weight_dict[label] = math.log(whole/num, 10)
        #     self.weight_dict[label] = math.log(num, math.e)
        # max_value = max(self.weight_dict.values())
        # min_value = min(self.weight_dict.values())
        # for k,v in self.weight_dict.items():
        #     self.weight_dict[k] = (v - min_value) / (max_value + min_value)
        # self.weight_dict[-1] = 1
        #
        # for label, num in self.clust_num_dict.items():
        #     self.weight_dict_2[label] = math.log(whole/num, math.e)
        # max_value = max(self.weight_dict_2.values())
        # min_value = min(self.weight_dict_2.values())
        # for k,v in self.weight_dict_2.items():
        #     self.weight_dict_2[k] = (v - min_value) / (max_value + min_value)
        # self.weight_dict_2[-1] = 1
        #
        # print(self.weight_dict)
        # print(self.weight_dict_2)

        # ========================================================================================================= #
        # 算向量的rank得分， 分高的向量是我们想要的
        for label_num, dist_list in self.dist_dict.items():
            # label_num 是行号，从1开始
            # score_list = heapq.nsmallest(3, dist_list)
            if label_num < merge_len :#or label_list[label_num - 1] != -1:
                continue

            temp = [2]*(len(self.clust_num_dict) - 1)
            temp2 = [-1]*(len(self.clust_num_dict) - 1)
            current_value_list_list = copy.deepcopy(value_list_list)
            count = 0
            dist_dict = {}
            for label, dist in zip(label_list, dist_list):
                count += 1
                if -1 == label:
                    continue
                if dist == 0:
                    continue
                dist_dict[count] = dist
                current_value_list_list[label].append(dist)
                if dist < temp[label]:    # find min in cluster
                    temp[label] = dist
                    temp2[label] = count
            temp3 = []
            for value_list in current_value_list_list:
                sum = 0
                for i in heapq.nsmallest(1, value_list):   # 每个簇里top n
                    sum += i
                temp3.append(sum)
            if temp == [2] * (len(self.clust_num_dict)-1):    # 0-vec
                score = 0
            else:
                # min_score = min(temp)
                # min_score = min(temp3)
                min_score = 0
                for i in heapq.nsmallest(3, temp3):    # top x簇
                    min_score += i/3
                if label_num == 1676:
                    print(temp)
                    print(temp2)
                    print(sorted(dist_dict.items(), key=lambda s:s[1]))
                    # min_score = 0.4028944108263043
                score = min_score #* self.weight_dict[label_list[dist_list.index(min_score)]]

            self.rank_dict[label_num] = score

        # ========================================================================================================= #
        # 对字典中的值进行归一化处理
        # max_value = max(self.rank_dict.values())
        # min_value = min(self.rank_dict.values())
        # for k, v in self.rank_dict.items():
        #     self.rank_dict[k] = (v - min_value)/(max_value + min_value)

        # ========================================================================================================= #
        # 算每个target向量与每个negative target向量的距离列表
        for label_num, vec in self.vec_dict.items():
            if label_num < merge_len:
                continue
            dist_list = []
            if vec.count(0) == len(vec):  # 0-vec
                dist_list = [2]*len(vec)
            else:
                for label, negative_vec in self.negative_vec_dict.items():
                    cos_dist = spatial.distance.cosine(vec, negative_vec)
                    dist_list.append(cos_dist)
            self.seq_dist_dict[label_num] = dist_list

        # ========================================================================================================= #
        # 算向量的rank得分， 分高的向量是我们想要的
        from scipy.special import expit
        positive_dict = copy.deepcopy(self.rank_dict)
        for label_num, dist_list in self.seq_dist_dict.items():
            if label_num < merge_len:
                continue
            # if label_list[label_num - 1] != -1:
            #     self.rank_dict[label_num] = 0
            else:
                score_list = heapq.nsmallest(1, dist_list)   # negative
                score = score_list[0]
                line_len = len(target_seq_lines[label_num - 1].split(" "))/2
                Wb = 0.5 - (1 - expit(line_len - 1))/5
                Wm = 1- Wb
                # print(line_len)
                self.rank_dict[label_num] = Wb * self.rank_dict[label_num]/2 + Wm * (2 - score)/2

        # print(self.rank_dict)
        return self.rank_dict, self.seq_dist_dict, positive_dict
    #
    # def remove_dup(self, proj_name, label_list, merge_len):
    #     with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" +
    #               proj_name + "_ft_target_vec.txt", "r") as file:
    #         target_vec_lines = file.readlines()
    #     with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" +
    #               proj_name + "_ft_target_seq.txt", "r") as file:
    #         target_seq_lines = file.readlines()
    #
    #     print("seq distant start")
    #     with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_ft_distant_matrix.pkl",
    #               "rb+") as file:
    #         dist_matrix = pickle.load(file)
    #         dist_matrix = dist_matrix.tolist()
    #     seq_dup_dict = {}
    #     label_dup_dict = {}
    #     line_label = 0
    #     for line in target_seq_lines:
    #         line_label += 1
    #         if line in seq_dup_dict.keys():
    #             seq_dup_dict[line] += 1
    #             label_dup_dict[line].append(line_label)
    #         else:
    #             seq_dup_dict[line] = 1
    #             label_dup_dict[line] = [line_label]
    #
    #     for dup_list in label_dup_dict.values():
    #         for dup_label in dup_list:
    #             target_vec_lines[dup_label - 1] = [404]
    #             dist_matrix[dup_label - 1] = [404]
    #     dist_matrix = list(filter(lambda l: l != [404], dist_matrix))
    #     target_vec_lines = list(filter(lambda l: l != [404], target_vec_lines))
    #     new_dist_matrix = []
    #     for line in dist_matrix:
    #         for dup_list in label_dup_dict.values():
    #             for dup_label in dup_list[1:]:
    #                 line[dup_label - 1] = 404
    #         new_dist_matrix.append(list(filter(lambda e: e != 404, line)))
    #     return target_seq_lines, target_vec_lines, new_dist_matrix, label_dup_dict
    #
    # def ranking_by_p_seq_one_n_seq_new(self, proj_name, label_list, merge_len):
    #     target_seq_lines, target_vec_lines, dist_matrix, label_dup_dict = self.remove_dup(proj_name, label_list, merge_len)
    #     # ========================================================================================================= #
    #     # 找过长的序列
    #     # with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" +
    #     #           proj_name + "_ft_target_seq.txt", "r") as file:
    #     #     target_seq_lines = file.readlines()
    #     seq_too_long_list = []
    #     count = 1
    #     for j in target_seq_lines:
    #         if len(j.split(" ")) > 300:
    #             seq_too_long_list.append(count)
    #         count += 1
    #
    #     # ========================================================================================================= #
    #     # 读取所有的target句向量
    #     # with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" +
    #     #           proj_name + "_ft_target_vec.txt", "r") as file:
    #     #     target_vec_lines = file.readlines()
    #     i = 0
    #     for line in target_vec_lines:
    #         i = i + 1
    #         vec_list = []
    #         line = line.split(" ")
    #         for num in line:
    #             if num.strip() == "":
    #                 continue
    #             vec_list.append(float(num))
    #         self.vec_dict[i] = vec_list
    #
    #     # ========================================================================================================= #
    #     # 制作数量dict
    #     for label in label_list:
    #         if label in self.clust_num_dict.keys():
    #             self.clust_num_dict[label] += 1
    #         else:
    #             self.clust_num_dict[label] = 1
    #     value_list_list = []
    #     for i in range(len(self.clust_num_dict)-1):
    #         value_list_list.append([2])
    #
    #     # ========================================================================================================= #
    #     # 算每个target向量与向量的距离列表
    #     print("seq distant start")
    #     # with open(PROJECT_DIR + "Result/SentenceEmbedding/" + proj_name + "/" + proj_name + "_ft_distant_matrix.pkl",
    #     #         "rb+") as file:
    #     #     dist_matrix = pickle.load(file)
    #     # dist_matrix = dist_matrix.tolist()
    #     count_temp = 0
    #     for i in dist_matrix:
    #         count_temp += 1
    #         self.dist_dict[count_temp] = i
    #     print("seq distant end")
    #
    #     # 算每个簇的权重, 簇越大权重越高
    #     whole = len(self.vec_dict)
    #     for label, num in self.clust_num_dict.items():
    #         # self.weight_dict[label] = math.log(whole/num, 10)
    #         self.weight_dict[label] = math.log(num, math.e)
    #     max_value = max(self.weight_dict.values())
    #     min_value = min(self.weight_dict.values())
    #     for k,v in self.weight_dict.items():
    #         self.weight_dict[k] = (v - min_value) / (max_value + min_value)
    #     self.weight_dict[-1] = 1
    #
    #     for label, num in self.clust_num_dict.items():
    #         self.weight_dict_2[label] = math.log(whole/num, math.e)
    #     max_value = max(self.weight_dict_2.values())
    #     min_value = min(self.weight_dict_2.values())
    #     for k,v in self.weight_dict_2.items():
    #         self.weight_dict_2[k] = (v - min_value) / (max_value + min_value)
    #     self.weight_dict_2[-1] = 1
    #
    #     print(self.weight_dict)
    #     print(self.weight_dict_2)
    #
    #     # ========================================================================================================= #
    #     # 算向量的rank得分， 分高的向量是我们想要的
    #     import copy
    #     for label_num, dist_list in self.dist_dict.items():
    #         # label_num 是行号，从1开始
    #         # score_list = heapq.nsmallest(3, dist_list)
    #         if label_num < merge_len :#or label_list[label_num - 1] != -1:
    #             continue
    #
    #         temp = [2]*(len(self.clust_num_dict) - 1)
    #         temp2 = [-1]*(len(self.clust_num_dict) - 1)
    #         current_value_list_list = copy.deepcopy(value_list_list)
    #         count = 0
    #         dist_dict = {}
    #         for label, dist in zip(label_list, dist_list):
    #             count += 1
    #             if -1 == label:
    #                 continue
    #             if dist == 0:
    #                 continue
    #             dist_dict[count] = dist
    #             current_value_list_list[label].append(dist)
    #             if dist < temp[label]:    # find min in cluster
    #                 temp[label] = dist
    #                 temp2[label] = count
    #         temp3 = []
    #         for value_list in current_value_list_list:
    #             sum = 0
    #             for i in heapq.nsmallest(1, value_list):
    #                 sum += i
    #             temp3.append(sum)
    #         if temp == [2] * len(self.clust_num_dict):    # 0-vec
    #             score = 0
    #         else:
    #             # min_score = min(temp)
    #             min_score = min(temp3)
    #             if label_num == 1744:
    #                 print(temp)
    #                 print(temp2)
    #                 print(sorted(dist_dict.items(), key=lambda s:s[1]))
    #                 # min_score = 0.4028944108263043
    #             score = min_score #* self.weight_dict[label_list[dist_list.index(min_score)]]
    #
    #
    #
    #         self.rank_dict[label_num] = score
    #
    #     # ========================================================================================================= #
    #     # 对字典中的值进行归一化处理
    #     # max_value = max(self.rank_dict.values())
    #     # min_value = min(self.rank_dict.values())
    #     # for k, v in self.rank_dict.items():
    #     #     self.rank_dict[k] = (v - min_value)/(max_value + min_value)
    #
    #     # ========================================================================================================= #
    #     # 算每个target向量与每个negative target向量的距离列表
    #     for label_num, vec in self.vec_dict.items():
    #         if label_num < merge_len:
    #             continue
    #         dist_list = []
    #         if vec.count(0) == len(vec):  # 0-vec
    #             dist_list = [2]*len(vec)
    #         else:
    #             for label, negative_vec in self.negative_vec_dict.items():
    #                 cos_dist = spatial.distance.cosine(vec, negative_vec)
    #                 dist_list.append(cos_dist)
    #         self.seq_dist_dict[label_num] = dist_list
    #
    #     # ========================================================================================================= #
    #     # 算向量的rank得分， 分高的向量是我们想要的
    #     import copy
    #     positive_dict = copy.deepcopy(self.rank_dict)
    #     for label_num, dist_list in self.seq_dist_dict.items():
    #         if label_num < merge_len:
    #             continue
    #         # if label_list[label_num - 1] != -1:
    #         #     self.rank_dict[label_num] = 0
    #         else:
    #             score_list = heapq.nsmallest(1, dist_list)
    #             score = score_list[0]
    #             self.rank_dict[label_num] = self.rank_dict[label_num]*0.5 + 0.5 * (2 - score)/2
    #
    #     # print(self.rank_dict)
    #     return self.rank_dict, self.seq_dist_dict, positive_dict, label_dup_dict