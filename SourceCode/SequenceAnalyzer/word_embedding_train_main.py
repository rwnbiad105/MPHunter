import pickle, os
from PreProcess.Remove_Builtin import remove_builtin, remove_short_seq, remove_bad_api, set_label
from SequenceAnalyzer.PreProcess.Set_API import set_API
from SequenceAnalyzer.PreProcess.SimplifyCallName import simplify_call_name
from SequenceAnalyzer.Training.Word_Embedding.Train_model import train
from SequenceAnalyzer.Analysis.get_dist_matrix import prepare_4_cluster

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 词嵌入模型训练主程序

if __name__ == "__main__":

    method = "ft"
    vector_size = 200
    window = 5
    min_count = 1
    epochs = 50
    sg = 1   # skip gram(1) or CBOW(0)
    workers = 12
    model_dir = "/home/liang/Desktop/workspace/type01/DataShare/model/"
    model_name = "0330_12000-pkg_c.x"   #"0102_13000pkg_ft_d200_e50_rm3"   # gai
    proj_name = "0330_12000-pkg_c.x"
    bad_api_dir = "PreProcess/removed_api_list.txt"
    dir = "output_connected_0330_12000_c.x.txt"
    dir1 = "output_processing_0330_12000_c.x.txt"
    dir2 = "output_simplified_0330_12000_c.x.txt"
    folder = "WordEmbedding"
    prefix_len = 2

    remove_builtin(dir,dir1, proj_name, folder)
    remove_bad_api(dir1, bad_api_dir, proj_name, folder)
    remove_short_seq(proj_name, method, dir1, 3, folder)
    set_label(dir1, proj_name, folder)

    set_API(dir1, proj_name, folder)
    simplify_call_name(dir1, dir2, proj_name, prefix_len, folder)

    # train(method, model_name, epoch, vector_size, proj_name, dir2)
    train(vector_size, window, min_count, epochs, sg, workers, proj_name, model_name, method, dir2, model_dir)

    # method = "w2v"
    # train(vector_size, window, min_count, epochs, sg, workers, model_name, method)
