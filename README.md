# CS 175 IR-based Chatbot
A Conversational chatbot based on information retreival techniques. It was trained through DailyDialog dataset.

Note: GoogleNews-vectors-negative300.bin is ignored as the file size is too large.

## How to Run this Chatbot in Your Local Machine
### Prerequisites
* Flask
* DailyDialog dataset
* Google News pre-trained word2vec model
* Packages needed: 
    * Gensim
    * pyemd
    * nltk
    * rank_bm25
    * sentence-transformers
* Redis sever

### Steps
1. Install all needed datasets/packages
2. Activate cs175-env: `source cs175-env/bin/activate`
3. Start server: `python3 app.py`

## Techniques We Used
1. BM25 + WMD
2. W2V (NN) + WMD
3. W2V (NN) + WMD + BERT
4. W2V (NN) + WMD + DAM 

WMD = Word Mover's Distance <br/>
W2V = word embeddings from Google News Corpus <br/>
BERT = Bidirectional Encoder Representations from Transformers <br/>
NN = nearest neighbors (relevant words) <br/>
DAM = Deep Attention Matching Network <br/>