from SequenceAnalyzer.Training.Word_Embedding.FT_Q1 import ft_Q1
from SequenceAnalyzer.Cluster.cluster_main import cluster_main_hdbscan, cluster_main_dbscan
import os, pickle
import numpy as np


# 两阶段类比推理中的第一阶段罢了
if __name__ == "__main__":
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = "/home/liang/Desktop/workspace/type01/DataShare/model/"
    model_name = "0127_12000-pkg_c.x"   # gai
    proj_name = "0127_12000-pkg_c.x"
    abstract = "codegen"   # e.g. "codegen" or "obf"

    # method = "w2v"
    # ft_Q1(model_name, method, abstract)

    method = "ft"
    ft_Q1(model_name, proj_name, method, abstract, model_dir)
    # seed_num = 10
    # ft_Q2(seed_num, model_name)
