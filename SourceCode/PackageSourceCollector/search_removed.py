import os, requests
from concurrent.futures import ThreadPoolExecutor
from PackageSourceCollector.Unpackage import unpackage_main

Filedict = {}


def get_source_list(dir, removed_list):
    global Filelist
    if os.path.isfile(dir):
        #print(dir)
        if dir.endswith("tar.gz"):
            tar_name = "-".join(dir.split("/")[-1].split("-")[:-1])
            print(tar_name)
            name1 = "-".join(tar_name.split("_"))
            name2 = "_".join(tar_name.split("-"))
            if name1 in removed_list:
                Filedict[name1] = dir
            if name2 in removed_list:
                Filedict[name2] = dir
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_source_list(newDir, removed_list)


def search_removed(datashare_path, packages_path, output_path, removed_list_path):
    output_path = os.path.join(datashare_path, output_path)
    with open(removed_list_path, "r") as file:
        lines = file.readlines()

    removed_list = []
    for i in lines:
        removed_list.append(i.strip())

    get_source_list(os.path.join(datashare_path, packages_path), removed_list)

    print(len(Filedict))
    import shutil
    for i in Filedict.values():
        tarname = i.split("/")[-1]
        shutil.copy(i, os.path.join(output_path, tarname))

    unsuccesslist = []
    for i in removed_list:
        if i not in Filedict.keys():
            unsuccesslist.append(i)
    # print(unsuccesslist)
    print(len(unsuccesslist))


if __name__ == "__main__":
    datashare_path = "/home/liang/Mount/workspace/DataShare2/Packages/"
    packages_path = "Packages_0103_6000_pypi_tar"
    output_path = "14-removed"
    removed_list_path = "./removed.txt"
    # packages_path = "Unpackaged_test"

    # search_removed(datashare_path, packages_path, output_path, removed_list_path)
    output_path2 = os.path.join(datashare_path, output_path)
    unpackage_main(datashare_path, output_path, output_path2)



