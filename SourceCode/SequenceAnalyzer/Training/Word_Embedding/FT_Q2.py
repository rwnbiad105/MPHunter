from gensim.models import FastText
from gensim.test.utils import datapath
import sys
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
import numpy as np
import os
import pickle

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/DataShare/"


def ft_Q2(seed_num=15, model_name="test"):
    with open(PROJECT_DIR + "ApiSeq_and_Result/PreProcess/api_dict.pkl", "rb") as dict_input:
        s_api_dict = pickle.load(dict_input)  # 简化后的 api的directory

    seeds_api_list = []
    with open(PROJECT_DIR + "ApiSeq_and_Result/Result/WordEmbedding/ft.Q1.txt", "r") as seeds_api_file:
        for i in range(seed_num):
            seeds_api_list.append(seeds_api_file.readline().split()[0])
    # 读ft1的结果
    set_APIs_dir = PROJECT_DIR + "ApiSeq_and_Result/PreProcess/Set_API_count.txt"  # gai in this case is 1 to n
    with open(set_APIs_dir, "r") as set_APIs_file:
        set_APIs_list = set_APIs_file.read().split('\n')  # 待推理api

    # 采用新的比较算法？聚类？
    fname = datapath(PROJECT_DIR + "model/" + model_name + '-DDG-fasttext.model')
    print("get file name:complete")
    model = FastText.load(fname)
    print("loading:complete")
    # print(sys.argv[1])

    i = 0
    for seed_api in seeds_api_list:
        seed_api_sim = s_api_dict[seed_api.strip()]
        i = i + 1
        if not os.path.exists(PROJECT_DIR + "ApiSeq_and_Result/FTQ2Result"):
            os.mkdir(PROJECT_DIR + "ApiSeq_and_Result/FTQ2Result")

        results = {}
        for set_api in set_APIs_list:
            set_api_sim = s_api_dict[set_api.strip()]

            seed_api_sim_v = model.wv[seed_api_sim]
            set_api_sim_v = model.wv[set_api_sim]

            d2 = 0
            try:
                d2 = np.dot(seed_api_sim_v, set_api_sim_v)/(np.linalg.norm(seed_api_sim_v) * np.linalg.norm(set_api_sim_v))
            except:
                print((np.linalg.norm(seed_api_sim_v) * np.linalg.norm(set_api_sim_v)))
                print(seed_api)
                print(set_api)
                print(set_api_sim_v)
                print("/0 ERROR!")
            results[set_api.strip()] = d2

        with open(PROJECT_DIR + "ApiSeq_and_Result/FTQ2Result/ft.Q2." + str(i) + ".txt", "w+") as output:
            sorted_results = sorted(results.items(), key=lambda d: d[1], reverse=True)
            print(sorted_results)
            for sr in sorted_results:
                # print(sr[0], sr[1], records[sr[0]])
                output.write(sr[0] + ' ' + str(sr[1]))
                output.write('\n')

if __name__ == "__main__":
    seeds_num = 15
    model_name = "1213"
    ft_Q2(seeds_num, model_name)
