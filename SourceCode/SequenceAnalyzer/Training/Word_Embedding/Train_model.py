from gensim.models import FastText
from gensim.models import Word2Vec
from gensim.test.utils import datapath
import os
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + "/DataShare/"


def train(vector_size=300, window=5, min_count=1, epochs=5, sg=1, workers=12, proj_name="test",
          model_name="test", mode="w2v", dir2="output_simplified.txt",
          model_dir="/home/liang/Mount/workspace/DataShare2/model/"):
    if not os.path.exists(model_dir + "WordEmbedding"):
        os.mkdir(model_dir + "WordEmbedding")
    if not os.path.exists(PROJECT_DIR + "ApiSeq_and_Result/Result/WordEmbedding"):
        os.mkdir(PROJECT_DIR + "ApiSeq_and_Result/Result/WordEmbedding")
    corpus_file = datapath(PROJECT_DIR + 'ApiSeq_and_Result/Result/WordEmbedding/' + proj_name + "/" + dir2)
    #corpus_file = datapath('/home/osroot/api_sequences.txt')

    if mode == "ft":
        model = FastText(vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, sg=sg, workers=workers)
    elif mode == "w2v":
        model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, sg=sg, workers=workers)
    else:
        print("Mode ERROR")
        return

    model.build_vocab(corpus_file=corpus_file)

    total_words = model.corpus_total_words
    print("\nstart training....")
    model.train(corpus_file=corpus_file,total_words=total_words,epochs=5)
    print("done")
    print("saving the model...")
    if not os.path.exists(model_dir + "WordEmbedding/" + proj_name):
        os.mkdir(model_dir + "WordEmbedding/" + proj_name)
    model.save(model_dir + "WordEmbedding/" + proj_name + "/" + model_name + "-" + mode + '.model')
    print("done\n")
    # print(model.wv['kfree'])
    cmd = "cp " + PROJECT_DIR + "ApiSeq_and_Result/PreProcess/api_dict.pkl " + PROJECT_DIR + "ApiSeq_and_Result/Result/WordEmbedding/" + proj_name + "/api_dict.pkl"
    os.system(cmd)


if __name__ == "__main__":
    vector_size = 300
    window = 5
    min_count = 1
    epochs = 5
    sg = 1
    workers = 12
    model_name = "test"
    mode = "w2v"
    train(vector_size, window, min_count, epochs, sg, workers, model_name, mode)
