import os
import numpy as np

from gensim.models import FastText
from gensim.test.utils import datapath
import sys
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def preprocess_union(project_name="test"):
    with open(PROJECT_DIR + '/Datashare/' + project_name + '/model_name_FT_Q1.txt', 'r') as model_name_file:
        model_name = model_name_file.readline()
        model_name_file.close()
    with open(PROJECT_DIR + '/Datashare/' + project_name + '/seed.txt', 'r') as seed_file:
    	seedlist = seed_file.readlines()
    	kmalloc_seed = seedlist[0].strip()
    	kfree_seed = seedlist[1].strip()
    	seed_file.close()
    fname = datapath(PROJECT_DIR + "/Datashare/"+project_name+"/model/"+model_name)
    print("get file name:complete")
    model = FastText.load(fname)
    print("loading:complete")
    
    input = open(PROJECT_DIR + "/Datashare/"+project_name+"/Union.txt", "r")
    #有可能需要审计
    # output_free = open(PROJECT_DIR + "/Datashare/"+project_name+"/ExtraFreeWaitSelect.txt", "w")
    # output_alloc = open(PROJECT_DIR + "/Datashare/"+project_name+"/AllocWaitSelect.txt", "w")
    output = open(PROJECT_DIR + "/Datashare/"+project_name+"/AllocFreeWaitSelect.txt", "w")
    content = input.read().split('\n')

    result_list = []
    list_element = []
    kmalloc_v = []
    kfree_v = []
    if len(kmalloc_seed.split(',')) == 1:
        kmalloc_v.append(model.wv[kmalloc_seed])
        kfree_v.append(model.wv[kfree_seed])
    else:
        kmalloc_seed_l = kmalloc_seed.split(',')
        kfree_seed_l = kfree_seed.split(',')
        for i in range(0, len(kmalloc_seed_l)):
            kmalloc_v.append(model.wv[kmalloc_seed_l[i]])
            kfree_v.append(model.wv[kfree_seed_l[i]])

    kmalloc_v = np.array(kmalloc_v)
    kfree_v = np.array(kfree_v)
    
    avg_malloc_v = np.mean(kmalloc_v, axis=0) # (kmalloc_v + vmalloc_v)/2
    avg_free_v = np.mean(kfree_v, axis=0) # (kfree_v + vfree_v)/2

    avg_offset_v = avg_free_v - avg_malloc_v
    
    for lines in content:
        line = lines.split(' ')
        if len(line) < 3:
            break
        alloclike = line[0]
        #print("      {{\"" + freelike + "\"}, &MallocChecker::checkExtraAlloc},")
        #output_alloc.write(freelike + "\n")
        freelike = line[1]
        #print("      {{\"" + freelike + "\"}, &MallocChecker::checkExtraFree},")
        #output_free.write(freelike + "\n")
        similarity = line[2]
        alloc_like_v = model.wv[alloclike]
        free_like_v = model.wv[freelike]
        al_fl = alloc_like_v - free_like_v
        fl_al = free_like_v - alloc_like_v
        # kmalloc - kfree v.s. split[0] - split[1]
        d1 = np.dot(alloc_like_v, (free_like_v + avg_offset_v))/(np.linalg.norm(alloc_like_v) * np.linalg.norm(free_like_v + avg_offset_v))
        # kmalloc - kfree v.s. split[1] - split[0]
        d2 = np.dot(free_like_v, (alloc_like_v + avg_offset_v))/(np.linalg.norm(free_like_v) * np.linalg.norm(alloc_like_v + avg_offset_v))
        if d1 > d2:
            sim = d1
        elif d2 >= d1:
            sim = d2
        sim_str = "%f"%sim
        output.write(alloclike + " " + freelike + " " + similarity + " " + sim_str + "\n")
    input.close()
    # output_free.close()
    # output_alloc.close()
    output.close()

    cmd = "sort -u " + PROJECT_DIR + "/Datashare/" + project_name + "/AllocFreeWaitSelect.txt > " + PROJECT_DIR + "/Datashare/" + project_name + "/AllocFreeWaitSelectSort.txt"
    # cmd = "sort -u " + PROJECT_DIR + "/Datashare/"+project_name+"/ExtraFreeWaitSelect.txt > " + PROJECT_DIR + "/Datashare/"+project_name+"/ExtraFreeWaitSelectSort.txt"
    # cmd2 = "sort -u " + PROJECT_DIR + "/Datashare/"+project_name+"/AllocWaitSelect.txt > " + PROJECT_DIR + "/Datashare/"+project_name+"/AllocWaitSelectSort.txt"
    os.system(cmd)
    # os.system(cmd2)


if __name__ == "__main__":
    preprocess_union()
