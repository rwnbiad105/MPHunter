import os
import pickle

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../DataShare/"


def simplify_call_name(dir="output_connected.txt", dir2="output_simplified.txt", proj_name="test", prefix_len=2, folder="WordEmbedding", prefix=["", "bfs_"]):
    for pre in prefix:
        original_api_list = set()
        simplified_api_list = set()
        simplified_api_dict = {}

        # build dict
        with open(PROJECT_DIR + "ApiSeq_and_Result/Result/" + folder + "/" + proj_name + "/" + "Set_API_count.txt", "r") as api_set_file:
            api_list = api_set_file.readlines()
            for api in api_list:
                if api == "":
                    continue
                if api.strip() == "":
                    continue
                api = api.split()[0]
                original_api_list.add(api)
                end_api = ""
                if api[:6] == "<call>":
                    end_api = api.split(".")[-1] + ""
                elif len(api.split(".")) < prefix_len:
                    end_api = api
                else:
                    for i in range(prefix_len):
                        index = 0 - prefix_len + i
                        end_api += api.split(".")[index]
                        if index != -1:
                            end_api += "."
                if end_api.strip() not in simplified_api_list:
                    simplified_api_list.add(end_api.strip())
                    simplified_api_dict[api.strip()] = end_api.strip()
                else:
                    n = 0
                    while 1:
                        n += 1
                        if end_api.strip() + str(n) not in simplified_api_list:
                            simplified_api_list.add(end_api.strip() + str(n))
                            simplified_api_dict[api.strip()] = end_api.strip() + str(n)
                            break

        # replace func name
        with open(PROJECT_DIR + "ApiSeq_and_Result/Result/" + folder + "/" + proj_name + "/" + pre + dir2, "w+") as s_output_file:
            with open(PROJECT_DIR + "ApiSeq_and_Result/Result/" + folder + "/" + proj_name + "/" + pre + dir, "r") as o_output_file:
                for line in o_output_file.read().split("\n"):
                    for api in line.split(" "):
                        if api in simplified_api_dict.keys():
                            s_output_file.write(simplified_api_dict[api] + " ")
                        else:
                            if len(api.split(".")) < prefix_len:
                                end_api = api
                            else:
                                end_api = ""
                                for i in range(prefix_len):
                                    index = 0 - prefix_len + i
                                    end_api += api.split(".")[index]
                                    if index != -1:
                                        end_api += "."
                            s_output_file.write(end_api + " ")
                    s_output_file.write("\n")

    # bump dict to file
    with open(PROJECT_DIR + "ApiSeq_and_Result/PreProcess/api_dict.pkl", "wb") as dict_output:
        pickle.dump(simplified_api_dict, dict_output)


if __name__ == "__main__":
    # you need to change the path to DataShare folder
    simplify_call_name()