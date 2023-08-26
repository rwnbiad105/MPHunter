import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../DataShare/"


def set_API(dir="output_connected.txt", proj_name="test", folder="WordEmbedding", prefix=["", "bfs_"]):
    dict1 = {}
    for pre in prefix:
        with open(PROJECT_DIR + "ApiSeq_and_Result/Result/" + folder + "/" + proj_name + "/" + pre + dir, "r") as file:

            text = file.readlines()
            words = " ".join(text)
            word_list = words.split()
            # 按照分隔符“ ”对整个文本进行切割

            for every_world in word_list:
                if every_world.strip() != "" or every_world != "":
                    if every_world[:6] == "<call>" or every_world[:9] == "__label__":
                        continue
                    if every_world in dict1:
                        dict1[every_world] += 1
                    else:
                        dict1[every_world] = 1
    with open(PROJECT_DIR + "ApiSeq_and_Result/Result/" + folder + "/" + proj_name + "/" + "Set_API_count.txt", "w+") as fp:
        dict1 = sorted(dict1.items(), key=lambda d:d[1], reverse=True)
        for item in dict1:
            fp.write(item[0] + " " + str(item[1]) + "\n")


if __name__ == "__main__":
    set_API()