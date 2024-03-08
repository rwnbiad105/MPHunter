import os, requests
from concurrent.futures import ThreadPoolExecutor

removed_list = []
Filelist = []
def get_source_list(dir):
    global Filelist
    if os.path.isfile(dir):
        #print(dir)
        pathlist = dir.split("/")
        if pathlist[-1] == "PKG-INFO":
            Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_source_list(newDir)


def get_json(name):
    global removed_list
    url = "https://pypi.org/pypi/" + name + "/json"
    try:
        r = requests.get(url, timeout=10)
        if r.json() == {"message": "Not Found"}:
            removed_list.append(name)
            print(name + " is removed")
        else:
            print(1)
    except:
        removed_list.append(name)
    # print(r.json())



def find_remove(datashare_path, packages_path):
    path = os.path.join(datashare_path, packages_path)
    target_list = os.listdir(path)

    get_source_list(path)
    pkginfo_list = Filelist
    # print(pkginfo_list)
    name_list = []

    for i in pkginfo_list:
        with open(i, "r") as file:
            lines = file.readlines()
        for line in lines:
            if line.startswith("Name:"):
                # print(line.strip("Name:"))

                if "-".join(line.split("_")).strip("Name:").strip() in name_list:
                    # print("-".join(line.split("_")).strip("Name:"))
                    break
                elif "_".join(line.split("-")).strip("Name:").strip() in name_list:
                    # print("-".join(line.split("_")).strip("Name:"))
                    break
                else:
                    name_list.append(line.strip("Name:").strip())
                break

    name_set = set(name_list)
    # print(name_set)
    print(len(name_set))

    with ThreadPoolExecutor(max_workers=16) as t:
        for name in name_set:
            t.submit(get_json, name)

    print(removed_list)


if __name__ == "__main__":
    datashare_path = "/home/liang/Mount/workspace/DataShare2/Packages/"
    packages_path = "Unpackaged_Packages_0103_6000_pypi_tar"
    # packages_path = "Unpackaged_test"
    find_remove(datashare_path, packages_path)


