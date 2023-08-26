from SequenceAnalyzer.Training.Word_Embedding.FT_Q1 import ft_Q1
from SequenceAnalyzer.Cluster.cluster_main import cluster_main_hdbscan, cluster_main_dbscan
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

from gensim.models import FastText
from gensim.test.utils import datapath
import numpy as np
import os
import pickle
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 尚未完成

def ft_train(vector_size=300, window=5, min_count=1, epochs=5, sg=1, workers=12, proj_name="test",
              model_name="test", mode="w2v", dir2="output_simplified.txt",
              model_dir="/home/liang/Mount/workspace/DataShare2/model/",
              API_set_Path=""
             ):
    if not os.path.exists(model_dir + "WordEmbedding"):
        os.mkdir(model_dir + "WordEmbedding")
    if not os.path.exists(PROJECT_DIR + "ApiSeq_and_Result/Result/WordEmbedding"):
        os.mkdir(PROJECT_DIR + "ApiSeq_and_Result/Result/WordEmbedding")
    corpus_file = datapath(PROJECT_DIR + 'ApiSeq_and_Result/Result/WordEmbedding/' + proj_name + "/" + dir2)
    #corpus_file = datapath('/home/osroot/api_sequences.txt')

    if mode == "ft":
        model = FastText(vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, sg=sg, workers=workers)
    else:
        print("Mode ERROR")
        return

    model.build_vocab(corpus_file=corpus_file)

    total_words = model.corpus_total_words
    print("\nstart training....")
    model.train(corpus_file=corpus_file,total_words=total_words,epochs=5)
    print("done")
    print("saving the model...")
    if not os.path.exists(model_dir + "WordEmbedding/" + proj_name):
        os.mkdir(model_dir + "WordEmbedding/" + proj_name)
    model.save(model_dir + "WordEmbedding/" + proj_name + "/" + model_name + "-" + mode + '.model')
    print("done\n")
    # print(model.wv['kfree'])
    cmd = "cp " + PROJECT_DIR + "ApiSeq_and_Result/PreProcess/api_dict.pkl " + PROJECT_DIR + "ApiSeq_and_Result/Result/WordEmbedding/" + proj_name + "/api_dict.pkl"
    os.system(cmd)


def ft_embedding_and_calculate_sim(model_name="test", proj_name="test", mode="ft", abstract="codegen",
          model_dir="/home/liang/Mount/workspace/DataShare2/model/"):
    with open(PROJECT_DIR + "ApiSeq_and_Result/Result/WordEmbedding/" + proj_name + "/api_dict.pkl", "rb") as dict_input:
        s_api_dict = pickle.load(dict_input)  # 简化后的 api的directory

    sink_seed_dir = PROJECT_DIR + "ApiSeq_and_Result/PreProcess/sink_seeds.txt"  # gai in this case is 1 to n
    with open(sink_seed_dir, "r") as sink_seed_file:
        sink_seeds_list = sink_seed_file.read().split('\n')  # 第一阶段的种子api文件

    set_APIs_dir = PROJECT_DIR + "ApiSeq_and_Result/Result/WordEmbedding/" + proj_name + "/Set_API_count.txt"
    # gai in this case is 1 to n
    with open(set_APIs_dir, "r") as set_APIs_file:
        set_APIs_list = set_APIs_file.read().split('\n')  # 第一阶段待推理api

    with open(PROJECT_DIR + "ApiSeq_and_Result/Result/WordEmbedding/" + proj_name + "/" + model_name + "." +
              abstract + "." + mode + ".txt", "w+") as output:

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
            if split_content.strip() == "":
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


if __name__ == "__main__":
    vector_size = 300
    window = 5
    min_count = 1
    epochs = 5
    sg = 1
    workers = 12
    model_name = "test"
    mode = "ft"
    ft_train(vector_size, window, min_count, epochs, sg, workers, model_name, mode)

    model_dir = "/home/liang/Desktop/workspace/type01/DataShare/model/"
    model_name = "0127_12000-pkg_c.x"   # gai
    proj_name = "0127_12000-pkg_c.x"
    abstract = "codegen"   # e.g. "codegen" or "obf"

    # method = "w2v"
    # ft_Q1(model_name, method, abstract)

    method = "ft"
    ft_embedding_and_calculate_sim(model_name, proj_name, method, abstract, model_dir)
    # seed_num = 10
    # ft_Q2(seed_num, model_name)
