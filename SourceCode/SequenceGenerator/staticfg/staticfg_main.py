import pickle

from .builder import CFGBuilder
from gensim.models import FastText
from gensim.test.utils import datapath
from scipy.spatial.distance import cosine
import numpy as np
import os


def dfs_bb(bb, visited, out):
    def sort_method(elem):
        return elem.target.score

    if bb.id not in visited:
        # print(bb.id)
        for i in bb.func_calls:
            if i:
                out += i
        visited.append(bb.id)
        # bb.exits.sort(key=sort_method)  # 按威胁程度排序
        for exit_ in bb.exits:
            if exit_.target in visited:
                continue
            out, visited_ = dfs_bb(exit_.target, visited, out)

    return out, visited


def bfs_bb(bb, visited, bfs_out):
    def sort_method(elem):
        return elem.target.score

    to_visit = [bb]
    while to_visit:
        block = to_visit.pop(0)
        if block not in visited:
            for i in block.func_calls:
                if i:
                    bfs_out += i
            visited.add(block)
            # block.exits.sort(key=sort_method)  # 按威胁程度排序
            for exit_ in block.exits:
                if exit_.target in visited or exit_.target in to_visit:
                    continue
                to_visit.append(exit_.target)

    return bfs_out, visited


def dfs_bb_calculate_score(bb, visited, out, model, sensitive_func_v_list):
    if bb.id not in visited:
        bb_min = 2
        for i in bb.func_calls:
            min = 2
            if i:
                func_v = model.wv[i.strip()]
                for j in sensitive_func_v_list:
                    score = get_angle(j, func_v)
                    if score < min:
                        min = score
            if min < bb_min:
                bb_min = min
        bb.score = bb_min
        # print(bb.id)
        # print(bb_min)
        visited.append(bb.id)
        for exit_ in bb.exits:
            if exit_.target in visited:
                continue
            out, visited_ = dfs_bb_calculate_score(exit_.target, visited, out, model, sensitive_func_v_list)
    return out, visited


def preprocess(model, cfg, sensitive_func_v_list):
    visited = []
    out = ""
    dfs_bb_calculate_score(cfg.entryblock, visited, out, model, sensitive_func_v_list)
    for subcfg in cfg.functioncfgs.values():
        dfs_bb_calculate_score(subcfg.entryblock, visited, out, model, sensitive_func_v_list)
    del visited
    del out


def get_angle(a, b):
    # a = np.array(a)
    # b = np.array(b)
    # cos_ = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    # sin_ = np.linalg.norm(np.cross(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    # angle = np.arctan2(sin_, cos_)
    angle = cosine(a, b)
    return angle


def staticfg_main(pyfile, call_dict, model, sensitive_func_v_list):
    dfs_out = ""
    bfs_out = ""
    for file in pyfile:
        dfs_out += "\n"
        bfs_out += "\n"
        dfs_out += file + "\n\n"
        bfs_out += file + "\n\n"
        cfg = CFGBuilder(file, call_dict).build_from_file('setup.py')
        if cfg.func_name:
            dfs_out += cfg.func_name
            bfs_out += cfg.func_name
        preprocess(model, cfg, sensitive_func_v_list)

        # dfs
        visited = []
        dfs_out, visited = dfs_bb(cfg.entryblock, visited, dfs_out)
        for subcfg in cfg.functioncfgs.values():
            dfs_out += "\n"
            if subcfg.func_name:
                dfs_out += subcfg.func_name
            dfs_out, visited = dfs_bb(subcfg.entryblock, visited, dfs_out)
        dfs_out += "\n"

        # bfs
        visited = set()
        bfs_out, visited = bfs_bb(cfg.entryblock, visited, bfs_out)
        for subcfg in cfg.functioncfgs.values():
            bfs_out += "\n"
            if subcfg.func_name:
                bfs_out += subcfg.func_name
            bfs_out, visited = bfs_bb(subcfg.entryblock, visited, bfs_out)
        bfs_out += "\n"

    return dfs_out, bfs_out


def uncanonical_staticfg_main(pyfile, call_dict): # 0505 消融实验
    dfs_out = ""
    bfs_out = ""
    for file in pyfile:
        dfs_out += "\n"
        bfs_out += "\n"
        dfs_out += file + "\n\n"
        bfs_out += file + "\n\n"
        cfg = CFGBuilder(file, call_dict).build_from_file('setup.py')
        if cfg.func_name:
            dfs_out += cfg.func_name
            bfs_out += cfg.func_name

        # dfs
        visited = []
        dfs_out, visited = dfs_bb(cfg.entryblock, visited, dfs_out)
        for subcfg in cfg.functioncfgs.values():
            dfs_out += "\n"
            if subcfg.func_name:
                dfs_out += subcfg.func_name
            dfs_out, visited = dfs_bb(subcfg.entryblock, visited, dfs_out)
        dfs_out += "\n"

        # bfs
        visited = set()
        bfs_out, visited = bfs_bb(cfg.entryblock, visited, bfs_out)
        for subcfg in cfg.functioncfgs.values():
            bfs_out += "\n"
            if subcfg.func_name:
                bfs_out += subcfg.func_name
            bfs_out, visited = bfs_bb(subcfg.entryblock, visited, bfs_out)
        bfs_out += "\n"

    return dfs_out, bfs_out

if __name__ == "__main__":
    model_dir = "/home/liang/Desktop/workspace/type01/DataShare/model/"
    WE_model_name = "0330_12000-pkg_c.x"   # gai
    WE_proj_name = "0330_12000-pkg_c.x"
    WE_abstract = "codegen"   # e.g. "codegen" or "obf"

    # method = "w2v"
    # ft_Q1(model_name, method, abstract)

    mode = "ft"
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    WE_model_file_dir = datapath(model_dir + "WordEmbedding/" + WE_proj_name + "/" + WE_model_name + "-" + mode + '.model')
    # print(WE_model_file_dir)
    model = FastText.load(WE_model_file_dir)
    print("model loading: complete")

    with open(PROJECT_DIR + "/DataShare/ApiSeq_and_Result/PreProcess/base_vec.pkl", "rb") as sensitive_func_file:
        base_vec = pickle.load(sensitive_func_file)

    # compute baseline
    # print("get base_vec: start")
    # word_vectors = model.wv
    # sum_vec = [0] * len(word_vectors[0])
    # print(len(word_vectors))
    # print(len(word_vectors[0]))
    # count = 0
    # for vec in word_vectors:
    #     count += 1
    #     for i in range(len(vec)):
    #         sum_vec[i] += vec[i]
    #     if count >= 200:
    #         break
    # base_vec = []
    # for i in range(len(vec)):
    #     base_vec.append(sum_vec[i]/200)
    # print("get base_vec: done")

    # '/home/liang/Desktop/workspace/type01/DataShare/MISC/always_updates-120.3/setup.py'   '/home/liang/Mount/workspace/DataShare2/Packages/Unpackaged_Packages_0329_2_pypi_tar/rcsb.utils.insilico3d-0.36/rcsb.utils.insilico3d-0.36/setup.py'
    pyfile = ['/home/liang/Mount/workspace/DataShare2/Packages/Unpackaged_Packages_0103_pypi_6000_tar/mypy-boto3-1.26.42/mypy-boto3-1.26.42/setup.py']
    call_dict = {'/home/liang/Mount/workspace/DataShare2/Packages/Unpackaged_Packages_0103_pypi_6000_tar/mypy-boto3-1.26.42/mypy-boto3-1.26.42/setup.py_8_32_8': 'os.path.abspath ', '/home/liang/Mount/workspace/DataShare2/Packages/Unpackaged_Packages_0103_pypi_6000_tar/mypy-boto3-1.26.42/mypy-boto3-1.26.42/setup.py_8_24_8': 'os.path.dirname ', '/home/liang/Mount/workspace/DataShare2/Packages/Unpackaged_Packages_0103_pypi_6000_tar/mypy-boto3-1.26.42/mypy-boto3-1.26.42/setup.py_8_19_8': '<builtin>.open ', '/home/liang/Mount/workspace/DataShare2/Packages/Unpackaged_Packages_0103_pypi_6000_tar/mypy-boto3-1.26.42/mypy-boto3-1.26.42/setup.py_8_19_8_0': 'read ', '/home/liang/Mount/workspace/DataShare2/Packages/Unpackaged_Packages_0103_pypi_6000_tar/mypy-boto3-1.26.42/mypy-boto3-1.26.42/setup.py_11_0_59': 'setuptools.setup '}
    dfs_out, bfs_out = staticfg_main(pyfile, call_dict, model, [base_vec])
    print(dfs_out)
    print(bfs_out)