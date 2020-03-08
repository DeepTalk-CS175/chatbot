from lib.wmd_model import *
from lib.process_corpus import *
import re
import time
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import scipy

from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

import redis
r = redis.Redis(host='localhost', port=6379, db=0)

start = time.time()
# get dataset ready
utterances, next_utterance, previous_utterances, all_acts, all_emotions = \
    process_data(f1='./res/train/dialogues_train.txt', \
        f2='./res/train/dialogues_act_train.txt', \
        f3='./res/train/dialogues_emotion_train.txt')

# tokenize all utterances as well to speed up queries a bit
utterances_tokenized = []
for u in utterances:
    tokens = word_tokenize(u)
    new_tokens = [token for token in tokens if re.match('^[A-Za-z0-9-]+$', token) and token not in stopwords]
    utterances_tokenized.append(new_tokens)

# Run word mover's distance model
wmd_model = wmd_model(word2vec_path="./res/GoogleNews-vectors-negative300.bin")
end = time.time()
loaded_time = end - start
print('===============successfully loaded models================')
print('in {} secs'.format(loaded_time))


def process_query(query, model, contexts):
    # retreive context from redis based on context_id
    if model == 'bm25-wmd':
        return bm25_wmd_model(query)
    elif model == 'w2v-wmd':
        return w2v_wmd_model(query)
    elif model == 'w2v-wmd-bert':
        return w2v_wmd_bert(query)
    else:
        return "DAM models will be up in future"


'''BM25 Model'''
def bm25_wmd_model(query):
    '''train bm25 model and get scores'''
    bm25 = BM25Okapi(utterances_tokenized)
    tokenized_query = query_tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    scores_w_indexes = [(i, s) for i, s in enumerate(bm25_scores)]
    bm25_sorted = sorted(scores_w_indexes, key=lambda x: -x[1])[0:1000]

    '''train WMD and get scores'''
    most_similar = []
    for index, score in bm25_sorted:
        dist = wmd_model.word2vec.wmdistance(query, utterances[index])
        most_similar.append((index, dist))

    best_utterance = sorted(most_similar, key=(lambda x: x[1]))[0]
    resp = utterances[next_utterance[best_utterance[0]]]

    return resp


'''w2v+wmd model'''
def w2v_wmd_model(query):
    utterance_indexes = get_utterance_indexes(query)
    most_similar = []
    for i in utterance_indexes:
        dist = wmd_model.word2vec.wmdistance(query, utterances[int(i)])
        most_similar.append((int(i), dist))
    
    best_utterance = sorted(most_similar, key=(lambda x: x[1]))[0]
    resp = utterances[next_utterance[best_utterance[0]]]

    return resp


'''w2v+wmd+bert model'''
def w2v_wmd_bert(query):
    utterance_indexes = get_utterance_indexes(query)
    most_similar = []
    for i in utterance_indexes:
        dist = wmd_model.word2vec.wmdistance(query, utterances[int(i)])
        most_similar.append((i, dist))
    
    # cut down the corpus size to 10
    sorted_lst = sorted(most_similar, key=(lambda x: x[1]))[0:10]

    # get all utterance_indexes of the new corpus (['111', '222'])
    top_utterance_indexes = [i for (i, dist) in sorted_lst]

    # each query has an index, utterance, and score; index corresponds to top_utterance_indexes
    top_10_queries = bert_model(top_utterance_indexes, query)

    # index, query, score for the top utterance (if i == 1, top_utterance_indexes[1] = '222')
    i, _, _ = top_10_queries[0]
    resp = utterances[next_utterance[int(top_utterance_indexes[i])]]

    return resp


'''BERT helper function'''
def bert_model(utterance_indexes, query):
    embedder = SentenceTransformer('bert-base-nli-mean-tokens')

    corpus = [utterances[int(i)] for i in utterance_indexes]
    corpus_embeddings = embedder.encode(corpus)

    # Query sentences:
    queries = [query]
    query_embeddings = embedder.encode(queries)

    # Find the closest 10 sentences of the corpus for each query sentence based on cosine similarity
    closest_n = 10
    top_10_queries = []
    
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        # idx is the utterance index in this 10-sentence corpus (e.g. 7, 0 ,2)
        for idx, distance in results[0:closest_n]:
#             print(idx, corpus[idx].strip(), "(Score: %.4f)" % (1-distance))
            top_10_queries.append((idx, corpus[idx].strip(), 1-distance))

    return top_10_queries


def get_utterance_indexes(query):
    tokens = word_tokenize(query)
    utterance_indexes = set()

    for token in tokens:
        token = token.lower()
        if re.match('^[A-Za-z0-9-]+$', token) and token not in stopwords:
            token_in_utterances = r.lrange('u-'+token, 0, -1)
            token_neighbors = r.lrange('n-'+token, 0, -1)
            
            if token_in_utterances != None:
                for u_index in token_in_utterances:
                    utterance_indexes.add(u_index.decode('utf-8'))
            
            # cut down to 5 nearest neighbors to speed up (no difference from 10 NN)
            if token_neighbors != None:
                index = 0
                for neighbor in token_neighbors:
                    if index == 5:
                        break
                    else:
                        neighbor_in_utterances = r.lrange('u-'+str(neighbor), 0, -1)
                        if neighbor_in_utterances != None:
                            for u_index in neighbor_in_utterances:
                                utterance_indexes.add(u_index.decode('utf-8'))
                        index += 1 

    return utterance_indexes


'''utilities function'''
def query_tokenize(query):
    tokens = word_tokenize(query)
    new_tokens = [token for token in tokens if re.match('^[A-Za-z0-9-]+$', token) and token not in stopwords]
    return new_tokens