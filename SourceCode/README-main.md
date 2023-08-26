
# Malicious Packages Hunter

## Abstruct
***
As the most popular Python software repository, PyPI has become an indispensable part of the Python ecosystem. Regrettably, the open nature of PyPI exposes end-users to substantial security risks stemming from malicious packages. Consequently, the timely and effective identification of malware within the vast number of newly-uploaded PyPI packages has emerged as a pressing concern. Existing detection methods are dependent on difficult-to-obtain explicit knowledge, such as taint sources, sinks, and malicious code patterns, rendering them susceptible to overlooking emergent malicious packages.

MPHunter is proposed as a static tool to detect PyPI malicious packages _**without requiring any explicit prior knowledge.**_ MPHunter utilizes clustering techniques to group the installation scripts of PyPI packages, identifying outliers and ranking them according to their outlierness and the distance between them and known malicious instances, thus highlighting potential evil packages.


## How it works
***
After ana-lyzing some malicious samples, we have found that there are
two exploitable facts.

First, malicious packages are significantly rarer than normal
ones, akin to a needle in a haystack. Although numerous
malicious packages have emerged on PyPI, the vast majority
of PyPI packages remain benign.

Second, the functionality of malicious installation scripts
substantially differs from those of benign ones, with the latter
frequently forming clusters. The setup.py script is dedicated
to configuring and managing the construction, publication,
and installation of packages. Consequently, the activities of
benign setup.py scripts tend to be similar. However, malicious
installation scripts execute dangerous operations that are un-
desirable for a normal setup.py and seldom present in it, such
as accessing sensitive information and installing backdoors.

Based on these observations, we introduce a novel method
called MPHunter (Malicious Packages Hunter) for detect-
ing malicious PyPI packages. In contrast to existing studies,
MPHunter directly leverages readily available benign packages
and known malicious samples, avoding the tedious and error-
prone manual analysis needed to extract explicit detection
knowledge.

![img.png](img.png "Fig.1 pipline of MPHunter")

The upper figur illustrates the pipeline of MPHunter. MPHunter
consists of the following four steps. First, an API encoding
model (termed APIEM) is built in advance by employing
word embedding (❶). This model
is used to assist in extracting canonical API call sequences
from the target scripts as training samples for CCEM. Second,
the CFGs and CGs of the PyPI package scripts are explored
using two traversal strategies, depth first search (DFS) and
breadth first search (BFS), under the guidance of APIEM. The
canonical API sequences are extracted to build CCEM (❷). Subsequently, we use CCEM to embed
the setup.py of newly-uploaded PyPI packages and benign
ones collected in advance. The obtained embedded vectors
are clustered with HDBSCAN [24] to identify outliers, i.e.,
malicious installation script candidates (❸). In the final step, candidates are ranked by measuring
the distances between them and benign samples, as well as
known malicious samples (❹). This
enables us to select the most likely suspects for auditing to
detect newly-uploaded malicious packages. Candidates distant
from known benign samples but close to known malicious
ones will be considered for manual analysis.

## Deploy
***

### # For GitHub User

Before deploying MPHunter, please make sure you have Pyhton3 (Python 3.10 is recommended).

You can use ``git clone`` command to copy our project to local.
``cd`` to the root directory of MPHunter and setting the environment with ``pip install -r requirements.txt``.

### # For Docker User
Coming soon.


## How to use
***
MPHunter use config.ini to config. Usually, we first train the FastText-PVDM document embedding model (CCEM) to embedding the code representations into vectors later in the detecting process.
However, if you want to used your own model, you need to set the path to your model and skip the training process. Then you can detect malicious packages. 


### basic
The [basic] part of configuration set up the most important config for MPHunter. It is mandatory. 

| para name | default                         | meaning                                                                                                                                                                                         |
| --------- |---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| vector_size | 300                             | vector_size is used to set the vector size of document vectors embedded from our code reprisentations.                                                                                          |
| window | 5                               | The size of window in FastText-PVDM embedding process.                                                                                                                                          |
| min_count | 1                               | The Minimum API numbers of code representation.                                                                                                                                                 |
|epoch | 200                             | The training epoch of FastText-PVDM.                                                                                                                                                            |  
|workers | 16                              | The number of workers of FastText-PVDM training process.                                                                                                                                        |
|prefix_len | 2                               | For an API in python scripts, they can represented as "a.b.c.x". This para control the max length of API representations. e.g., when prefix_len = 3, a API will be expressed as "b.c.x" format. |
|bad_api_dir | PreProcess/removed_api_list.txt | Some API are commenly used in python project, which can greatly affect the result of code clutering. Appanding the bad APIs in defualt list or setting the path to you own list.   |
|WE_model_name | -                               | The APIEM model name.|
|WE_proj_name | -                               | The APIEM model training data file name in DataShare. Usually, we set the same name as WE_model_name.|
|model_name | -                               | The CCEM model name.|
|action | train                           | "train" for model training process;"detect" for malicious detection. |

### training
The [train] part of configuration is needed when action is "train".

| para name | default    | meaning                                                                                           | 
| --------- |------------|---------------------------------------------------------------------------------------------------|
|proj_name | model_name | The CCEM model training data file name in DataShare. Usually, we set the same name as model_name. |
|dir0 | -          | Inital code representation data file name.                                                        | 
|dir1 | -          | Processing code representation data file name.                                                    | 
|dir2 | -          | Final code representation data file name.                                                         | 
|file_search_mode | all_files  | all_files  or setup_only                                                                          |
|graph_mode | cfg+cg     | Setting "cfg+cg" for canonical code representation and "cg" for uncanonical code representaion.   |
|task_num | 16         | The number of workers when generating code representations.                                       | 

### detection
The [detect] part of configuration is needed when action is "detect"

| para name | default | meaning                                                                       |
| --------- |----|-------------------------------------------------------------------------------|
|pkg_name | Unpackaged_Packages_0609_1_pypi_tar | Path to folder of pkgs waiting to be tested.                                  | 
|proj_name | final-4_0609_1_cfg_setup_c.x | The name of data file in DataShare.                                           | 
|dir0 | -          | Inital code representation data file name.                                    | 
|dir1 | -          | Processing code representation data file name.                                | 
|dir2 | -          | Final code representation data file name.                                     | 
|proj_merged | final-4_0609_1_cfg_setup_c.x_merged | Merged (with benign set) file name.                                           |
|file_search_mode | setup_only | all_files  or setup_only                                                      |
|graph_mode | cfg+cg | cfg+cg   or  cg                                                               |
|rank_mode | p&n | positive or negative or p&n  : The way of taking rank(p-only, n-only and p&n) |
|min_cluster_size | 2 | HDBSCAN's min_cluster_size                                                    | 
|min_samples | 20 | HDBSCAN's min_samples                                                         | 
|task_num | 16 | worker number.                                                                | 
|proj_wait_to_be_merged | final-4_0417_benign_cfg_setup_c.x | name of benign set proj                                                       | 
|negative_proj_name | final-4_0417_malicious_cfg_setup_c.x | name of negative set proj                                                     | 


