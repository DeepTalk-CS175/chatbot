#-*- coding: utf-8 -*-
from gensim import models
from gensim.similarities import WmdSimilarity
import time

class wmd_model(object):

    def __init__(self, word2vec_path):
        start = time.time()
        self.word2vec = models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        self.word2vec.init_sims(replace=True)
        print ("word2vec model loaded.")
        print("took time: ",time.time()-start)


    def top_K_similar(self, query, corpus, K=10):
        wmd_inst = WmdSimilarity(corpus, self.word2vec,num_best=K, normalize_w2v_and_replace=False)
        scores = wmd_inst[query]
        return scores
    
    def most_similar_words(self, word, K=5):
        return self.word2vec.most_similar(word, topn=K)
        