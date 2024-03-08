import json
import random
import os, re, subprocess
import pickle
import requests
import datetime
import dateutil.parser
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED, as_completed
import threading
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# ################# for threadpool #########################################
import queue
from concurrent.futures import ThreadPoolExecutor

class BoundThreadPoolExecutor(ThreadPoolExecutor):

    def __init__(self, *args, **kwargs):
        super(BoundThreadPoolExecutor, self).__init__(*args, **kwargs)
        self._work_queue = queue.Queue(14)
#############################################################################

def download_all(index, output, output_dir):
    file = open(index, "r")
    for line in file.readlines():
        reguline = line.replace("/", "")
        os.system("pip download --pre --no-deps --no-binary=:all: --default-timeout=100 --no-cache-dir --use-deprecated=legacy-resolver -d " + output_dir + output + reguline)


def download_part(index, output, num, output_dir):
    with open(index, "r") as file:
        lines = file.readlines()
    try:
        with open(CURRENT_DIR + "/../DataShare/Packages/" + output + "/index.pkl", "rb+") as index_f:
            lidex_set = pickle.load(index_f)
    except:
        lidex_set = set()
    if len(lidex_set) >= len(lines) + num:
        print("PYPI don't have enough packages, try a smaller download amount.")

    while num > 0:
        sample = random.sample(lines, num)
        for line in sample:
            reguline = line.replace("/", "")
            if reguline.strip() in lidex_set:
                continue
            cmd = "pip download -d " + output_dir + output + "/" + reguline.strip() + "/ --pre --no-deps " \
                   "--default-timeout=100 --use-deprecated=legacy-resolver --no-cache-dir --no-build-isolation " \
                   "--no-binary=:all: " + reguline.strip() + " 2>>" + output_dir + output + "/pip_output.txt"
            # cmd = "pip download -d ../DataShare/Packages/" + output + "/" + reguline.strip() + "/ --no-deps --no-binary=:all: --python-version 3 " + reguline
            print(reguline.strip() + " downloading.....................................")
            print(cmd)
            os.system(cmd)
            lidex_set.add(reguline.strip())
            num -= 1
            with open(CURRENT_DIR + "/../DataShare/Packages/" + output + "/downloaded.txt", "a+") as index_f:
                index_f.write(reguline)

    with open(CURRENT_DIR + "/../DataShare/Packages/" + output + "/index.pkl", "wb+") as index_set:
        pickle.dump(lidex_set, index_set)


def download_part_by_txt(target_list_dir, output, output_dir):
    def download_single(reguline, output, output_dir):
        if reguline.split("-")[:4] == ['botocore','a','la','carte'] or reguline.split("-")[:3] == ['tencentcloud','sdk','python']:
            return ""
        cmd = "pip download -d " + output_dir + output + "/ --pre --no-deps --no-build-isolation --no-binary=:all: " \
                "--no-cache-dir --use-deprecated=legacy-resolver " + reguline.strip() + " 2>>" + output_dir + output + "/pip_output.txt"
        print(reguline.strip() + " downloading.....................................")
        print(cmd)
        os.system(cmd)
        return reguline

    if not os.path.exists(output_dir + output):
        os.mkdir(output_dir + output)
    with open(output_dir + output + "/pip_output.txt", "w+") as fp:
        fp.write("start\n")
    with open(target_list_dir, "r") as file:
        lines = file.readlines()
    num = len(lines)

    print(num)
    with ThreadPoolExecutor(max_workers=14) as pool:
        # all_task = [pool.submit(download_single, [line, output]) for line in lines]
        # wait(all_task, return_when=ALL_COMPLETED)
        obj_list = []
        for line in lines:
            obj = pool.submit(lambda cpx:download_single(*cpx), (line, output, output_dir))
            obj_list.append(obj)
        for future in as_completed(obj_list):
            print("done:" + future.result())


def get_benign_pkg_name(num, lines, benign_list):

    line_num = random.randint(1, num)
    pkg = lines[line_num].split("/")[-2]
    if pkg in benign_list:
        return
    url = "https://pypi.org/pypi/" + pkg + "/json"
    # url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/" + pkg + "/json"
    print(url)
    try:
        r = requests.get(url, timeout=60)
    except:
        print("request fail:" + pkg)
        return
    data = r.json()
    if "releases" in data.keys():
        releases_version = next(iter(data["releases"]))
        upload_time = data["releases"][releases_version]
        if upload_time:
            upload_time = upload_time[0]
            if "upload_time" in upload_time.keys():
                upload_time = upload_time["upload_time"]
                insertion_date = dateutil.parser.parse(upload_time)
                now_time = datetime.datetime.utcnow().isoformat()
                diffretiation = dateutil.parser.parse(now_time) - insertion_date
                if diffretiation.days > 90:
                    print(pkg)
                    return pkg


def get_random_pkg_name(num, lines, total_num):
    temp = []
    visited = []
    i = 0
    while i < total_num:
        line_num = random.randint(1, num)
        if line_num not in visited:
            pkg = lines[line_num].split("/")[-2]
            temp.append(pkg)
            i = i + 1
    return temp


def download_part_by_index(index_list, output, output_dir, download_num, date):
    def download_single(reguline, output, output_dir):
        if reguline.split("-")[:4] == ['botocore','a','la','carte'] or reguline.split("-")[:3] == ['tencentcloud','sdk','python']:
            return ""
        cmd = "pip download -d " + output_dir + output + "/ --pre --no-deps --no-build-isolation --no-binary=:all: " \
                "--no-cache-dir --use-deprecated=legacy-resolver " + reguline.strip() + " 2>>" + output_dir + output + "/pip_output.txt"
        print(reguline.strip() + " downloading.....................................")
        print(cmd)
        os.system(cmd)
        return reguline

    if not os.path.exists(output_dir + output):
        os.mkdir(output_dir + output)
    with open(output_dir + output + "/pip_output.txt", "w+") as fp:
        fp.write("start\n")
    with open(index_list, "r") as file:
        lines = file.readlines()
    num = len(lines)
    count = 0
    benign_list = get_random_pkg_name(num, lines, download_num)
    if not os.path.exists(CURRENT_DIR + "/../DataShare/json/" + date + "/"):
        os.mkdir(CURRENT_DIR + "/../DataShare/json/" + date + "/")
    with open(CURRENT_DIR + "/../DataShare/json/" + date + "/name.txt", "w+") as file:
        for i in benign_list:
            file.write(i + "\n")

    print(num)
    with ThreadPoolExecutor(max_workers=14) as pool:
        # all_task = [pool.submit(download_single, [line, output]) for line in lines]
        # wait(all_task, return_when=ALL_COMPLETED)
        obj_list = []
        for line in benign_list:
            obj = pool.submit(lambda cpx:download_single(*cpx), (line, output, output_dir))
            obj_list.append(obj)
        for future in as_completed(obj_list):
            print("done:" + future.result())


def download_benign_part_by_index(index_list, output, output_dir, download_num, date):
    def download_single(reguline, output, output_dir):
        if reguline.split("-")[:4] == ['botocore','a','la','carte'] or reguline.split("-")[:3] == ['tencentcloud','sdk','python']:
            return ""
        cmd = "pip download -d " + output_dir + output + "/ --pre --no-deps --no-build-isolation --no-binary=:all: " \
                "--no-cache-dir --use-deprecated=legacy-resolver " + reguline.strip() + " 2>>" + output_dir + output + "/pip_output.txt"
        print(reguline.strip() + " downloading.....................................")
        print(cmd)
        os.system(cmd)
        return reguline

    if not os.path.exists(output_dir + output):
        os.mkdir(output_dir + output)
    with open(output_dir + output + "/pip_output.txt", "w+") as fp:
        fp.write("start\n")
    with open(index_list, "r") as file:
        lines = file.readlines()
    num = len(lines)
    count = 0
    benign_list = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        while count <= download_num:
            ret = pool.submit(lambda cpx:get_benign_pkg_name(*cpx), (num, lines, benign_list))
            if ret.result():
                benign_list.append(ret)
                count = len(benign_list)
    if not os.path.exists(CURRENT_DIR + "/../DataShare/json/" + date + "/"):
        os.mkdir(CURRENT_DIR + "/../DataShare/json/" + date + "/")
    with open(CURRENT_DIR + "/../DataShare/json/" + date + "/name.txt", "w+") as file:
        for i in as_completed(benign_list):
            file.write(i.result() + "\n")

    print(num)
    with ThreadPoolExecutor(max_workers=14) as pool:
        # all_task = [pool.submit(download_single, [line, output]) for line in lines]
        # wait(all_task, return_when=ALL_COMPLETED)
        obj_list = []
        for line in as_completed(benign_list):
            obj = pool.submit(lambda cpx:download_single(*cpx), (line.result(), output, output_dir))
            obj_list.append(obj)
        for future in as_completed(obj_list):
            print("done:" + future.result())



def download_failure_one(output, output_dir):
    with open(output_dir + output + "/pip_output.txt", "r") as file:
        lines = file.readlines()
    packages = []
    needs_pip_install = []
    for line in lines:
        m = re.match("[ ]*╰─> [0-9a-zA-Z_. -]*", line)
        if m:
            pkg = m.group().replace("╰─> ", "")
            if pkg.strip() != "":
                packages.append(pkg)
        m2 = re.match("[ ]*ERROR: No matching distribution found for [0-9a-zA-Z_. -]*", line)
        if m2:
            pkg2 = m2.group().replace("ERROR: No matching distribution found for ", "")
            if pkg2.strip() != "":
                packages.append(pkg2)
        m3 = re.match("[ ]*ModuleNotFoundError: No module named \'[0-9a-zA-Z_. -]*\'", line)
        if m3:
            pkg3 = m3.group().replace("\'", "")
            pkg3 = pkg3.replace("ModuleNotFoundError: No module named ", "")
            if pkg3.strip() != "":
                needs_pip_install.append(pkg3)

    print(needs_pip_install)
    for i in needs_pip_install:
        try:
            cmd = "pip install " + i
            print(cmd)
            subprocess.run(cmd, shell=True, timeout=60)
        except:
            pass

    print(packages)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for pkg in packages:
            pool.submit(lambda cpx: download_by_curl(*cpx), (pkg, output_dir, output))



def download_by_curl(pkg, output_dir, output):
    url = "https://pypi.org/pypi/" + pkg + "/json"
    # url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/" + pkg + "/json"
    print(url)
    try:
        r = requests.get(url)
    except:
        print(pkg)
        return
    data = r.json()
    if "urls" in data.keys():
        num = len(data["urls"])
        tar = ""
        whl = ""
        for i in range(num):
            m = re.match("[0-9a-zA-Z._/:-]*.tar.gz", data["urls"][i]["url"])
            if m:
                tar = m.group()
            m2 = re.match("[0-9a-zA-Z._/:-]*.whl", data["urls"][i]["url"])
            if m2:
                whl = m2.group()
        if tar != "":
            cmd = "curl -m 120 -o " + os.path.join(output_dir, output, tar.split("/")[-1]) + " " + tar
            os.system(cmd)
        # elif whl != "":
        #     cmd = "curl -m 120 -o " + os.path.join(output_dir, output, whl.split("/")[-1]) + " " + whl
        #     os.system(cmd)
    else:
        if pkg.strip() != "":
            print(pkg + " not found urls")


def download_json(date, is_html=True):
    if is_html:
        with open(CURRENT_DIR + "/../DataShare/html/" + date + "/name.txt", "rt") as file:
            lines = file.read()
        lines = lines.replace("      <span class=\"package-snippet__name\">", "")
        lines = lines.replace("</span>", "")
        with open(CURRENT_DIR + "/../DataShare/html/" + date + "/name.txt", "wt") as file:
            file.write(lines)
        with open(CURRENT_DIR + "/../DataShare/html/" + date + "/name.txt", "r") as file:
            lines = file.readlines()
    else:
        with open(CURRENT_DIR + "/../DataShare/json/" + date + "/name.txt", "r") as file:
            lines = file.readlines()

    if not os.path.exists(CURRENT_DIR + "/../DataShare/json/" + date + "/"):
        os.mkdir(CURRENT_DIR + "/../DataShare/json/" + date + "/")
    for line in lines:
        pkg = line.strip()
        # url = "https://pypi.org/pypi/" + pkg + "/json"
        url = "https://mirrors.tuna.tsinghua.edu.cn/pypi/" + pkg + "/json"
        print(url)
        try:
            r = requests.get(url, timeout=60)
        except:
            print(pkg)
            continue
        try:
            data = r.json()
            save = json.dumps(data)
            with open(CURRENT_DIR + "/../DataShare/json/" + date + "/" + pkg + ".json", "w+") as file:
                file.write(save)
        except:
            continue


if __name__ == "__main__":
    # index = "./AllPackagesIndex.txt"
    target_list = "./malwares_at_qhy.txt"
    output_dir = "/home/liang/Mount/workspace/DataShare2/Packages/"
    output = "Packages_0308_6000_flooding_malware_tar"
    # num = 11000
    # download_part(index, output, num)
    download_part_by_txt(target_list, output, output_dir)
