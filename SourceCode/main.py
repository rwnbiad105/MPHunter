from SequenceGenerator.get_seq_main import get_seq_main
from SequenceGenerator.basic_preprocess_main import Preprocessor
from SequenceAnalyzer.sentence_embedding_Embedding_only_without_train_model import sentence_embedding_only
from SequenceAnalyzer.sentence_embedding_merge_main import merge_main
from SequenceAnalyzer.sentence_embedding_analyse_main_calculate_centroid import \
    sentence_embedding_analyse_main_calculate_centroid
from SequenceAnalyzer.sentence_embedding_train_model_only_without_Embedding import train_corpus
import time, os, configparser

start = time.time_ns()
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_corpus_model(datashare_path, pkg_name, folder, proj_name, method, vector_size, min_count, model_dir,
                       model_name, bad_api_dir, dir0, dir1, dir2, prefix_len, epoch, WE_model_name, WE_proj_name,
                       file_search_mode, graph_mode, task_num):
    get_seq_main(datashare_path, pkg_name, folder, proj_name, model_dir, WE_model_name, WE_proj_name, method,
                 file_search_mode, graph_mode, task_num)
    pre = Preprocessor()
    pre.seq_preprocess_mul_process(0, dir0, proj_name, folder, method, [""], graph_mode, task_num)
    train_corpus(method, vector_size, min_count, model_dir, model_name, proj_name, bad_api_dir,
                 dir0, dir1, dir2, prefix_len, folder, epoch, graph_mode)


def detec_malware(datashare_path, pkg_name, folder, proj_name, method, vector_size, min_count, model_dir, model_name,
                  bad_api_dir, dir0, dir1, dir2, prefix_len, proj_wait_to_be_merged, proj_merged, negative_proj_name,
                  rank_mode, merge_len, WE_model_name, WE_proj_name, file_search_mode, graph_mode, task_num,
                  min_cluster_size, min_samples):
    get_seq_main(datashare_path, pkg_name, folder, proj_name, model_dir, WE_model_name, WE_proj_name, method,
                 file_search_mode, graph_mode, task_num)
    pre = Preprocessor()
    pre.seq_preprocess_mul_process(0, dir0, proj_name, folder, method, ["setup.py"], graph_mode, task_num)
    sentence_embedding_only(method, vector_size, min_count, model_dir, model_name, proj_name, bad_api_dir,
                            dir0, dir1, dir2, prefix_len, folder, graph_mode, proj_wait_to_be_merged)
    merge_main(proj_wait_to_be_merged, proj_name, proj_merged, model_name, method)
    sentence_embedding_analyse_main_calculate_centroid(proj_merged, negative_proj_name, rank_mode,
                                                                      method, merge_len, min_cluster_size, min_samples)


def train_WE_and_train_corpus_model(datashare_path, pkg_name, folder, proj_name, method, vector_size,
                                    min_count, model_dir, model_name, bad_api_dir, dir0, dir1, dir2,
                                    prefix_len, epoch, WE_model_name, WE_proj_name, prefix, file_search_mode,
                                    graph_mode):

    get_seq_main(datashare_path, pkg_name, folder, proj_name, model_dir, WE_model_name, WE_proj_name, method,
                 file_search_mode, graph_mode)
    # get_seq_main_cmd(datashare_path, pkg_name, folder, proj_name, model_dir, WE_model_name, WE_proj_name, method)
    # pre = Preprocessor()
    # pre.seq_preprocess(0, dir0, proj_name, folder, method, [], False, "output_cg.txt")
    # train_corpus(method, vector_size, min_count, model_dir, model_name, proj_name, bad_api_dir,
    #              dir0, dir1, dir2, prefix_len, folder, epoch, prefix)
    # get_seq_main_cmd(datashare_path, pkg_name, folder, proj_name, model_dir, WE_model_name, WE_proj_name, method)
    # pre = Preprocessor()
    # pre.seq_preprocess(0, dir0, proj_name, folder, method, [])
    # train_corpus(method, vector_size, min_count, model_dir, model_name, proj_name, bad_api_dir,
    #              dir0, dir1, dir2, prefix_len, folder, epoch, prefix)


def main():
    cf = configparser.ConfigParser()
    cf.read("./config.ini")
    
    datashare_path = os.path.join(PROJECT_DIR, "../Packages")
    # datashare_path = "/home/liang/Mount/workspace/DataShare2/Packages"
    model_dir = PROJECT_DIR + "/DataShare/model/"
    # model_dir = "/home/liang/Desktop/workspace/type01/DataShare/model/"
    folder = "SentenceEmbedding"
    method = "ft"
    bad_api_dir = "PreProcess/removed_api_list.txt"

    vector_size = cf.getint("basic", "vector_size")
    window = cf.getint("basic", "window")
    min_count = cf.getint("basic", "min_count")
    epoch = cf.getint("basic", "epoch")
    workers = cf.getint("basic", "workers")
    prefix_len = cf.getint("basic", "prefix_len")

    WE_model_name = cf.get("basic", "WE_model_name")
    WE_proj_name = cf.get("basic", "WE_proj_name")
    model_name = cf.get("basic", "model_name")
    # "0412-8000-pkg_for_corpus_c.x"  # "0412-10000-pkg_for_corpus_cg_inline_c.x"#"0412-8000-pkg_for_corpus_c.x"
    # "0332-10000-pkg_for_corpus_c.x"   # "0222_12000-pkg_for_corpus_c.x"  "0415-10000-pkg_for_corpus_c.x"

    action = cf.get("basic", "action")

    # ======================for train======================================= #
    if action == "train":
        pkg_name = cf.get("train", "pkg_name")    # "Unpackaged_Packages_0226_0103_corpus_10000"
        # "0214_12000-pkg_for_corpus_c.x"  or   "0214_6000-pkg_new_updated_pypi_setup_c.x"
        proj_name = model_name  # "0213_200-pkg_negative_c.x"
        dir0 = cf.get("train", "dir0")
        dir1 = cf.get("train", "dir1")
        dir2 = cf.get("train", "dir2")
        file_search_mode = cf.get("train", "file_search_mode")     # "all_files"  or "setup_only"
        graph_mode = cf.get("train", "graph_mode")    # "cfg+cg"   or  "cg"
        task_num = cf.getint("train", "task_num")

        with open("./DataShare/ApiSeq_and_Result/PreProcess/proj_config.txt", "w+") as config_file:
            for i in [datashare_path, pkg_name, folder, proj_name, method, vector_size, min_count, model_dir,
                      model_name,bad_api_dir, dir0, dir1, dir2, prefix_len, WE_model_name, WE_proj_name,
                      file_search_mode, graph_mode, task_num]:
                config_file.write(str(i) + "\n")

        train_corpus_model(datashare_path, pkg_name, folder, proj_name, method, vector_size, min_count, model_dir,
                       model_name, bad_api_dir, dir0, dir1, dir2, prefix_len, epoch, WE_model_name, WE_proj_name,
                       file_search_mode, graph_mode, task_num)

    # ========================for detect=====================================#
    elif action == "detect":
        pkg_name = cf.get("detect", "pkg_name")#"Unpackaged_Packages_all_malicious_set_pypi_tar"#"Unpackaged_Packages_0404_benign_pypi_tar"#"Unpackaged_Packages_recall_rate_malicious_set"#"Unpackaged_Packages_recall_rate_background_set"#"Packages_0407_recall_rate_test_malicious"#"Unpackaged_Packages_0103_6000_pypi_tar"    #"Unpackaged_Packages_0414_m_pypi_tar"  #"Unpackaged_Packages_pypi_all_malicious_set_tar" "Unpackaged_Packages_0103_6000_pypi_tar"   #
        # "0214_12000-pkg_for_corpus_c.x"  or   "0214_6000-pkg_new_updated_pypi_setup_c.x" or "0214_200-pkg_negative_c.x"
        proj_name = cf.get("detect", "proj_name") # "0417_2_cfg_setup_c.x"   #"0414_retest_0414_m_setup_c.x"  # "0213_200-pkg_negative_c.x"
        dir0 = cf.get("detect", "dir0")
        dir1 = cf.get("detect", "dir1")
        dir2 = cf.get("detect", "dir2")
        proj_merged = cf.get("detect", "proj_merged")
        file_search_mode = cf.get("detect", "file_search_mode")     # "all_files"  or "setup_only"
        graph_mode = cf.get("detect", "graph_mode")    # "cfg+cg"   or  "cg"
        min_cluster_size = cf.getint("detect", "min_cluster_size")
        min_samples = cf.getint("detect", "min_samples")
        task_num = cf.getint("detect", "task_num")
        proj_wait_to_be_merged = cf.get("detect", "proj_wait_to_be_merged")#"Unpackaged_Packages_0404_benign_pypi_tar" #"final-3_0505_benign_cfg_setup_c.x"#"final-1_0417_benign_cfg_setup_c.x"   #"final-1_0417_rrt_cfg_setup_c.x_benign4000"#"final-1_0417_benign_cfg_setup_c.x"  #"0414_benign_6000-pkg_dfs_setup_c.x_rm3"   #"0415_10000-pkg_benign_c.x"   #"0404_benign_10000-pkg_setup_c.x"   # "0222_6000-pkg_new_updated_pypi_setup_c.x"
        negative_proj_name = cf.get("detect", "negative_proj_name")#"final-3_0505_malicious_100_cfg_setup_c.x"   #"final-1_0417_rrt_cfg_setup_c.x_malicious" #"final-1_0417_malicious_cfg_setup_c.x"  #"0414_all_malicious_setup_c.x"  # "0415_10000-pkg_negative_c.x"     #"0404_all_malicious_setup_c.x"     # "0304_900-pkg_malicious_setup_c.x"  # "0226_800-pkg_all_malicious_c.x"

        rank_mode = cf.get("detect", "rank_mode")  # "positive" or "negative" or "p&n"
        merge_len = 1672 + 1  #5847 + 1 # 2839 + 1   #1306 + 1  # 2585 + 1  # 5847 + 1

        with open("./DataShare/ApiSeq_and_Result/PreProcess/proj_config.txt", "w+") as config_file:
            for i in [datashare_path, pkg_name, folder, proj_name, method, vector_size, min_count, model_dir, model_name,
                      bad_api_dir, dir0, dir1, dir2, prefix_len, proj_wait_to_be_merged, negative_proj_name,
                      rank_mode, merge_len, WE_model_name, WE_proj_name, file_search_mode, graph_mode, task_num]:
                config_file.write(str(i) + "\n")

        detec_malware(datashare_path, pkg_name, folder, proj_name, method, vector_size, min_count, model_dir, model_name,
                      bad_api_dir, dir0, dir1, dir2, prefix_len, proj_wait_to_be_merged, proj_merged, negative_proj_name,
                      rank_mode, merge_len, WE_model_name, WE_proj_name, file_search_mode, graph_mode, task_num,
                      min_cluster_size, min_samples)
    
    
if __name__ == "__main__":
    main()


end = time.time_ns()
print('Running time: %s Seconds' % str(int(end - start)/1000000000))