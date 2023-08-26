import pickle, os
from SequenceAnalyzer.PreProcess.Remove_Builtin import remove_builtin, remove_short_seq, remove_bad_api, set_label
from SequenceAnalyzer.PreProcess.Set_API import set_API
from SequenceAnalyzer.PreProcess.SimplifyCallName import simplify_call_name
from SequenceAnalyzer.Training.Sentence_Embedding.train_model import get_vec, train_only
from SequenceAnalyzer.Analysis.get_dist_matrix import prepare_4_cluster

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def concatenate_dfs_bfs_output(dir1, dir3, folder, proj_name):
    prefix = ["", "bfs_"]
    all_lines = ""
    for pre in prefix:
        with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/" + folder + "/" + proj_name + "/" + pre + dir1, "r") as fp:
            all_lines += fp.read()

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/" + folder + "/" + proj_name + "/" + dir3, "w+") as fp:
        fp.write(all_lines)
    # print(PROJECT_DIR + "/Datashare/ApiSeq_and_Result/Result/" + folder + "/" + proj_name + "/" + dir2)


def train_corpus(method, vector_size, min_count, model_dir, model_name, proj_name, bad_api_dir,
                            dir0, dir1, dir2, prefix_len, folder, epoch, graph_mode):
    if graph_mode == "cg":
        prefix = [""]
    if graph_mode == "cfg+cg":
        prefix = ["", "bfs_"]

    # 上下连接bfs dfs
    # ===============================================================
    # dir3 = "output_corpus.txt"
    # remove_builtin(dir0, dir1, proj_name, folder)
    # remove_bad_api(dir1, bad_api_dir, proj_name, folder)
    # remove_short_seq(proj_name, method, dir1, min_count, folder)
    #
    # concatenate_dfs_bfs_output(dir1, dir3, folder, proj_name)
    # set_label(dir3, proj_name, folder, [""])
    # set_API(dir3, proj_name, folder, [""])
    # simplify_call_name(dir3, dir2, proj_name, prefix_len, folder, [""])
    # ===============================================================

    # 前后连接bfs dfs
    # ===============================================================
    remove_builtin(dir0, dir1, proj_name, folder, prefix)
    remove_bad_api(dir1, bad_api_dir, proj_name, folder, prefix)
    remove_short_seq(proj_name, method, dir1, min_count, folder, prefix)
    set_label(dir1, proj_name, folder, prefix)
    set_API(dir1, proj_name, folder, prefix)
    simplify_call_name(dir1, dir2, proj_name, prefix_len, folder, prefix)

    lines_list = []
    for pre in prefix:
        with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/" + folder + "/" + proj_name + "/" + pre + dir2, "r") as fp:
            lines_list.append(fp.readlines())
    output_list = []
    if len(lines_list) == 2:
        for i in range(len(lines_list[0])):
            output_list.append(lines_list[0][i].strip() + " " + " ".join(lines_list[1][i].strip().split(" ")[1:]))
        with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/" + folder + "/" + proj_name + "/concatenated_" + dir2, "w+") as fp:
            for i in output_list:
                fp.write(i.strip() + "\n")

        dir3 = "concatenated_" + dir2
    else:
        dir3 = dir2
    # ===============================================================

    train_only(method, model_name, epoch, vector_size, proj_name, dir3, model_dir)


if __name__ == "__main__":
    method = "ft"
    vector_size = 300
    window = 5
    min_count = 3
    epoch = 200
    workers = 12
    model_dir = "/home/liang/Desktop/workspace/type01/DataShare/model/"
    model_name = "0222_12000-pkg_for_corpus_c.x" #"0102_13000pkg_ft_d200_e50_rm3"   # gai
    proj_name = "0222_12000-pkg_for_corpus_c.x"
    bad_api_dir = "PreProcess/removed_api_list.txt"
    dir0 = "output_connected_0222_12000_c.x.txt"
    dir1 = "output_processing_0222_12000_c.x.txt"
    dir2 = "output_simplified_2_0222_12000_c.x.txt"
    prefix_len = 2
    folder = "SentenceEmbedding"

    train_corpus(method, vector_size, min_count, model_dir, model_name, proj_name, bad_api_dir,
                            dir0, dir1, dir2, prefix_len, folder, epoch)


