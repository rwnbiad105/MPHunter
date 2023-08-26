from gensim.models import FastText
from gensim.test.utils import datapath
import sys
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
import numpy as np
import os
import pickle

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + "/DataShare/"


def ft_Q1(model_name="test", proj_name="test", mode="ft", abstract="test",
          model_dir="/home/liang/Mount/workspace/DataShare2/model/"):
    with open(PROJECT_DIR + "ApiSeq_and_Result/Result/WordEmbedding/" + proj_name + "/api_dict.pkl", "rb") as dict_input:
        s_api_dict = pickle.load(dict_input)  # 简化后的 api的directory

    sink_seed_dir = PROJECT_DIR + "ApiSeq_and_Result/PreProcess/sink_seeds.txt"  # gai in this case is 1 to n
    sink_seed_file = open(sink_seed_dir, "r")
    sink_seeds_list = sink_seed_file.read().split('\n')  # 第一阶段的种子api文件

    set_APIs_dir = PROJECT_DIR + "ApiSeq_and_Result/Result/WordEmbedding/" + proj_name + "/Set_API_count.txt"  # gai in this case is 1 to n
    set_APIs_file = open(set_APIs_dir, "r")
    set_APIs_list = set_APIs_file.read().split('\n')  # 第一阶段待推理api

    output = open(PROJECT_DIR + "ApiSeq_and_Result/Result/WordEmbedding/" + proj_name + "/" + model_name + "." + abstract + "." + mode + ".txt", "w+")

    model_file_dir = datapath(model_dir + "WordEmbedding/" + proj_name + "/" + model_name + "-" + mode + '.model')
    model = FastText.load(model_file_dir)
    print("model loading: complete")

    # print(sink_seed_v_list)
    # sink_seed_v_list = np.array(sink_seed_v_list)
    # sink_seed_v = np.mean(sink_seed_v_list, axis=0)   # 多个向量求平均

    results = {}

    for split_content in set_APIs_list:
        if split_content == "":
            continue
        split_content = split_content.split()[0]
        if split_content.strip() == "" or split_content == "":
            continue
        for sink_seed in sink_seeds_list:
            # 待测api与种子api 两两匹配
            if sink_seed.strip() == "":
                continue
            if sink_seed.strip() in s_api_dict.keys():
                sink_seed_v = model.wv[s_api_dict[sink_seed.strip()]]  # dict change
            else:
                sink_seed_v = model.wv[sink_seed.strip()]  # dict change
            s_split_content = s_api_dict[split_content.strip()]
            try:
                sink_like_api_v = model.wv[s_split_content]    # 待测api转换成词向量
            except:
                print("ERROR 1")
                print(len(split_content))
                print(s_split_content)
            try:
                d1 = np.dot(sink_like_api_v, sink_seed_v)/(np.linalg.norm(sink_like_api_v) * np.linalg.norm(sink_seed_v))
            except:
                print((np.linalg.norm(sink_like_api_v) * np.linalg.norm(sink_seed_v)))
                print(split_content)
                #print(sink_seed_v)
                print("/0 ERROR!")
            if split_content not in results.keys():   # 录入结果
                results[split_content] = (sink_seed.strip(), d1)
            elif results[split_content][1] <= d1:
                results[split_content] = (sink_seed.strip(), d1)
            else:
                pass

    sorted_results = sorted(results.items(), key=lambda d:d[1][1] , reverse=True)
    #print(sorted_results)

    for sr in sorted_results:
        # print(sr[0], sr[1], records[sr[0]])
        output.write(sr[0] + ' ' + str(sr[1]))
        output.write('\n')

    output.close()
    set_APIs_file.close()
    sink_seed_file.close()


if __name__ == "__main__":
    ft_Q1("1215", "w2v", "codegen")
