import os, pickle

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def train(method="ft", model_name="test", epoch=40, dim=300,
          proj_name="test", simplified_dir="output_simplified.txt",
          model_dir="/home/liang/Mount/workspace/DataShare2/model/"):

    if not os.path.exists(model_dir + "SentenceEmbedding"):
        os.mkdir(model_dir + "SentenceEmbedding")
    if not os.path.exists(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding"):
        os.mkdir(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding")

    if method == "ft":
        cmd = "cd " + PROJECT_DIR + "/SequenceAnalyzer/Training/fastText-PV\n" \
                                    "./fasttext PVDM -input ../../../DataShare/ApiSeq_and_Result/" \
                                    "Result/SentenceEmbedding/" + proj_name + "/" + simplified_dir + " " \
                                    " -output " + model_dir + "SentenceEmbedding/" + model_name + "_" + \
                                    method + "_model -epoch " + str(epoch) + " -dim " + str(dim)
        print(cmd)
        os.system(cmd)
        cmd = "cp " + model_dir + "SentenceEmbedding/" + model_name + "_" + \
              method + "_model.labels.vec " + PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/" \
                "SentenceEmbedding/" + proj_name + "/" + model_name + "_" + method + "_model.txt"
        os.system(cmd)

        # 建立target output
        seq_dir = PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/" \
                                "SentenceEmbedding/" + proj_name + "/" + simplified_dir
        target_seq_list = []
        print("start collecting vectors:")
        with open(seq_dir, "r") as seq_file:
            lines = seq_file.readlines()

        with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + 
                  "/" + proj_name + "_" + method + "_target_num_dict.pkl", "rb+") as file:
            target_dict = pickle.load(file)
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
            target_seq_list.append(" ".join(temp))

        get_vec(method, target_seq_list, model_name, proj_name, model_dir)
        # print(target_seq_list)


def train_only(method="ft", model_name="test", epoch=40, dim=300, proj_name="test", 
               simplified_dir="output_simplified.txt",
                model_dir="/home/liang/Mount/workspace/DataShare2/model/"):

    if not os.path.exists(model_dir + "SentenceEmbedding"):
        os.mkdir(model_dir + "SentenceEmbedding")
    if not os.path.exists(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding"):
        os.mkdir(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding")

    if method == "ft":
        cmd = "cd " + PROJECT_DIR + "/SequenceAnalyzer/Training/fastText-PV\n" \
                                    "./fasttext PVDM -input ../../../DataShare/ApiSeq_and_Result/Result/" \
                                    "SentenceEmbedding/" + proj_name + "/" + simplified_dir + " " \
                                    " -output " + model_dir + "SentenceEmbedding/" + model_name + "_" + \
                                    method + "_model -epoch " + str(epoch) + " -dim " + str(dim)
        print(cmd)
        os.system(cmd)
        cmd = "cp " + model_dir + "SentenceEmbedding/" + model_name + "_" + method + \
              "_model.labels.vec " + PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" +\
              proj_name + "/" + model_name + "_" + method + "_model.txt"
        os.system(cmd)


def get_vec(method="ft", target_seq_list=[], model_name="test", proj_name="test",
          model_dir="/home/liang/Mount/workspace/DataShare2/model/", prefix=""):

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + prefix +
              proj_name + "_" + method + "_target_seq.txt", "w+") as file:
        for line in target_seq_list:
            file.write(line.strip() + "\n")

    list = []
    for line in target_seq_list:
        print(line)
        with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/temp.txt", "w+") as file:
            file.write(line + "\n")

        cmd = "cd " + PROJECT_DIR + "/SequenceAnalyzer/Training/fastText-PV\n" \
              "./fasttext predictPVDM " + model_dir + "SentenceEmbedding/" + model_name + "_" + method +\
              "_model.bin " + PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/temp.txt" \
              " ../../../DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + proj_name +\
              "_" + method + "_target_vec"
        # print(cmd)
        os.system(cmd)
        with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" +
                  proj_name + "_" + method + "_target_vec.vec", "r") as fp:
            line_temp = fp.readline()
            list.append(line_temp)
            # print(line_temp)
    # print(list)
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + prefix + proj_name +
              "_" + method + "_target_vec.txt", "w+") as file:
        for line in list:
            file.write(line)
    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/Result/SentenceEmbedding/" + proj_name + "/" + prefix + proj_name +
              "_" + method + "_target_vec.pkl", "w+") as file:
        for line in list:
            file.write(line)


if __name__ == "__main__":
    train(method="ft", epoch=40, model_name="test")



