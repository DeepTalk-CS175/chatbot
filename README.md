# CS 175 IR-based Chatbot
A Conversational chatbot based on information retreival techniques. It was trained through DailyDialog dataset.

## How to Run this Chatbot in Your Local Machine
### Prerequisites
* Flask
* DailyDialog dataset
* Google News pre-trained word2vec model (not included in this repo as the file size is too large)
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
We first tried single-turn models (does not care about context history). Then, we moved one step forward by which the bot will return a response based on previous utterances (aka. contexts). We picked Deep Attention Matching Network for the multi-turn conversation.

Single-turn models:
1. BM25 + WMD
2. W2V (NN) + WMD
3. W2V (NN) + WMD + BERT

Multi-turn models:
* W2V (NN) + WMD + DAM

Notes: <br/>
WMD = Word Mover's Distance <br/>
W2V = word embeddings from Google News Corpus <br/>
BERT = Bidirectional Encoder Representations from Transformers <br/>
NN = nearest neighbors (relevant words) <br/>
DAM = Deep Attention Matching Network <br/>

## Screenshots

<img src='https://i.imgur.com/RFidJt1.png' title='chatbot-demo' width='' alt='Chatbot Demo' />