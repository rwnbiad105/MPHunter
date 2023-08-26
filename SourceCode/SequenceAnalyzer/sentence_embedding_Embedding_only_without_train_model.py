import pickle, os
from .PreProcess.Remove_Builtin import remove_builtin, remove_short_seq, remove_short_seq2, remove_bad_api, set_label
from SequenceAnalyzer.PreProcess.Set_API import set_API
from SequenceAnalyzer.PreProcess.SimplifyCallName import simplify_call_name
from SequenceAnalyzer.Training.Sentence_Embedding.train_model import get_vec, train
from SequenceAnalyzer.Analysis.get_dist_matrix import prepare_4_cluster

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================================================
# 本程序会只分析setup.py, 并且不会进行模型训练，更不会覆盖原来的模型
# 需要其他模型的支持
# =====================================================================================


def sentence_embedding_only(method="ft", vector_size=300, min_count=3,
                            model_dir="/home/liang/Desktop/workspace/type01/DataShare/model/",
                            model_name="0222_12000-pkg_for_corpus_c.x",
                            proj_name="0228_1600-pkg_new_updated_pypi_setup_c.x",
                            bad_api_dir="PreProcess/removed_api_list.txt",
                            dir0="output_connected_0228_1600_c.x.txt",
                            dir1="output_processing_0228_1600_c.x.txt",
                            dir2="output_simplified_2_0228_1600_c.x.txt",
                            prefix_len=2,
                            folder="SentenceEmbedding",
                            graph_mode="cg",
                            proj_wait_to_be_merged=""):
    if graph_mode == "cg":
        prefix = ["cg_"]
    elif graph_mode == "cfg+cg":
        prefix = ["", "bfs_"]
    remove_builtin(dir0, dir1, proj_name, folder, prefix)
    remove_bad_api(dir1, bad_api_dir, proj_name, folder, prefix)

    # set_API(dir, proj_name, folder)
    # 0331 fork

    cmd = "cp " + PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + model_name + "/Set_API_count.txt" \
          " " + PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/Set_API_count.txt"
    os.system(cmd)
    lines_list = []
    for pre in prefix:
        with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/" + folder + "/" + proj_name + "/" + pre + dir1, "r") as fp:
            lines_list.append(fp.readlines())
    output_list = []
    if graph_mode == "cfg+cg":
        for i in range(len(lines_list[0])):
            output_list.append(lines_list[0][i].strip() + " " + " ".join(lines_list[1][i].strip().split(" ")[:]))
    elif graph_mode == "cg":
        for i in range(len(lines_list[0])):
            output_list.append(lines_list[0][i])
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/" + folder + "/" + proj_name + "/concatenated_" + dir1, "w+") as fp:
        for i in output_list:
            fp.write(i.strip() + "\n")
    simplify_call_name("concatenated_" + dir1, "concatenated_" + dir2, proj_name, prefix_len, folder, [""])
    remove_short_seq(proj_name, method, "concatenated_" + dir2, min_count, folder, [""], proj_wait_to_be_merged)

    # remove_short_seq2(proj_name, method, dir1, dir2, min_count, folder, prefix)
    # cmd = "cp " + PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/" + folder + "/" + proj_name + "/" + dir2 +\
    #       " " + PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/" + folder + "/" + proj_name + "/dfs_" + dir2
    # os.system(cmd)
    # pre_analysis only code begin:

    set_label("concatenated_" + dir2, proj_name, folder, [""])
    # ++++++++++++++++++ "concatenated_" can be replaced by pre ++++++++++++++++++++
    seq_dir = PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/concatenated_" + dir2
    target_seq_list = []
    print("start collecting vectors:")
    with open(seq_dir, "r") as seq_file:
        lines = seq_file.readlines()

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" +
              proj_name + "_" + method + "_target_num_dict.pkl", "rb+") as file:
        target_dict = pickle.load(file)
    print(target_dict)
    # 读向量信息
    for line in lines:
        # print(line)
        if line.strip() == "":
            continue
        if int(line.split(" ")[0][9:]) not in target_dict.values():
            continue
        temp = []
        for i in line.split(" ")[1:]:
            if i.strip() == "":
                continue
            temp.append(i)
        target_seq_list.append(" ".join(temp).strip("\n"))

    get_vec(method, target_seq_list, model_name, proj_name, model_dir)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    prefix = [""]    # 0407
    matrix = prepare_4_cluster(proj_name, method, prefix)
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" +
              proj_name + "_" + method + "_distant_matrix.pkl", "wb+") as matrix_file:
        pickle.dump(matrix, matrix_file)
    print(matrix)
    print(len(matrix))


if __name__ == "__main__":
    method = "ft"
    vector_size = 300
    window = 5
    min_count = 3
    epoch = 200
    workers = 12
    model_dir = "/home/liang/Desktop/workspace/type01/DataShare/model/"
    model_name = "0222_12000-pkg_for_corpus_c.x"
    proj_name = "0304_3440-pkg_new_updated_pypi_setup_c.x"  # "0213_200-pkg_negative_c.x"
    bad_api_dir = "PreProcess/removed_api_list.txt"
    dir0 = "output_connected_0304_3440_c.x.txt"
    dir1 = "output_processing_0304_3440_c.x.txt"
    dir2 = "output_simplified_0304_3440_c.x.txt"

    prefix_len = 2
    folder = "SentenceEmbedding"

    sentence_embedding_only(method, vector_size, min_count, model_dir, model_name, proj_name, bad_api_dir,
                            dir0, dir1, dir2, prefix_len, folder)

