import os
import filetype
# 遍历文件夹及其子文件夹中的文件，并存储在一个列表中

# 输入文件夹路径、空文件列表[]

# 返回 文件列表Filelist,包含文件名（完整路径）
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_file_list(dir, Filelist):

    if os.path.isfile(dir):

        Filelist.append(dir)
        # # 若只是要返回文件文，使用这个
        # Filelist.append(os.path.basename(dir))

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            # if s == "xxx":
            # continue
            newDir = os.path.join(dir, s)
            get_file_list(newDir, Filelist)

    return Filelist


def unpackage(dir, list, datashare_path):
    for e in list:
        print(e)
        pathlist = e.split("/")
        kind = filetype.guess(e)
        if kind is None:
            print('Cannot guess file type!')
            continue
        else:
            extension = kind.extension
            zip_pkg_name = ".".join(pathlist[-1].split(".")[:-1])
            tar_pkg_name = ".".join(pathlist[-1].split(".")[:-2])
            if extension == "zip":
                cmd = "unzip " + e + " -d " + datashare_path + "/Unpackaged_" + dir.split("/")[-1] + "/" + zip_pkg_name
            elif extension == "gz":
                if not os.path.exists(datashare_path + "/Unpackaged_" + dir.split("/")[-1] + "/" + tar_pkg_name):
                    os.mkdir(datashare_path + "/Unpackaged_" + dir.split("/")[-1] + "/" + tar_pkg_name)
                cmd = "tar -C " + datashare_path + "/Unpackaged_" + dir.split("/")[-1] + "/" + tar_pkg_name + " -zxf " + e
            else:
                cmd = ""
            print(cmd)
            os.system(cmd)
            # cmd = "rm " + e
            # print(cmd)
            # os.system(cmd)


def unpackage_main(datashare_path, packages_path, dir):

    if not os.path.exists(datashare_path + "/Unpackaged_"+packages_path.split("/")[-1]):
        os.mkdir(datashare_path + "/Unpackaged_"+packages_path.split("/")[-1])

    list = get_file_list(dir, [])

    print(len(list))
    unpackage(packages_path, list, datashare_path)


if __name__ == '__main__':
    datashare_path = "/home/liang/Mount/workspace/DataShare2/Packages/"
    packages_path = "Packages_0306_pypi_1720_tar"
    dir = datashare_path + "/" + packages_path + "/"
    unpackage_main(datashare_path, packages_path, dir)

