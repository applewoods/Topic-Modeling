from itertools import count
from operator import index
import os
from pickle import FALSE
import pandas as pd
import numpy as np
from datetime import datetime

# Preprocessing
import gensim
from gensim import corpora
from gensim.models import TfidfModel, CoherenceModel
from gensim.test.utils import common_corpus

# LDA Library
import tomotopy as tp

# Visualization
import matplotlib.pyplot as plt

class Tomotopy_LDA_Models:
    def __init__(self, tokenized_doc, path='./'):
        self.path = path
        self.tokenized_doc = tokenized_doc
        self.dictionary = corpora.Dictionary(tokenized_doc)

        # Bag of Words corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in tokenized_doc]

        # TF-IDF corpus
        tfidf = TfidfModel(self.corpus)
        self.corpus_tfidf = tfidf[self.corpus]

        # etc
        self.today = datetime.now().strftime('%Y%m%d')
        self.save_file_name = np.int64(self.today + '001')

    def LDA(self, num_topics= 10, min_cf= 10, tw= 'IDF'):
        # num_topic : 토픽의 갯수
        # min_df : 
        # min_cf : 전체 말뭉치에서 N회 미만 등장한 단어들 제거
        # tw : tp.TermWeight의 항목 중 하나를 선택하여 입력(tp.TermWeight.ONE / tp.TermWeight.IDF / tp.TermWeight.PMI)
        # IDF(정보량) : 해당 사건이 발생한 확률의 역수에 로그를 취한 값
        # PMI(점별 상호정보량) : 문헌이 출현할 확률과 단어가 출현할 확률의 

        termweight = ''
        if tw == 'ONE':
            termweight = tp.TermWeight.ONE
        elif tw == 'IDF':
            termweight = tp.TermWeight.IDF
        elif tw == 'PMI':
            termweight = tp.TermWeight.PMI
        
        self.LDA_model = tp.LDAModel(k= num_topics, min_cf= min_cf, tw= termweight, rm_top= 0, alpha= 0.1, eta= 0.01, seed= 2022)

        for token in self.tokenized_doc:
            self.LDA_model.add_doc(token)

    def train_model(self, max= 500, term= 100):
        self.LDA_model.train(0)
        self.show_model_detail()

        print('Training...', flush= True)
        for i in range(0, max, term):
            self.LDA_model.train(term)
            print('Iteration: {}\t Log-likelihood: {}'. format(i, self.LDA_model.ll_per_word))

    def show_model_detail(self):
        print('Model Info')
        print('=' * 30)
        print('Num Docs:', len(self.LDA_model.docs))
        print('Vocab Size:', self.LDA_model.num_vocabs)
        print('Num Words:', self.LDA_model.num_words)

        print()
        print('Remove Top Words')
        print(self.LDA_model.removed_top_words)

    def get_coherenece(self):
        coherence = tp.coherence
        coherence_score = coherence.Coherence(self.LDA_model).get_score()

        return coherence_score

    def get_perplexity(self):
        perplexity = self.LDA_model.perplexity
        perplexity_score = np.log(perplexity)

        return perplexity_score

    def compute_LDA_Models(self, min_topics= 2, max_topics= 100, steps= 1):
        self.coherence_values = list()
        self.perplexity_values = list()
        self.lda_model_list = list()

        for num_topic in range(min_topics, max_topics, steps):
            self.LDA(num_topics= num_topic)
            self.train_model(max= 500, term= 100)

            self.lda_model_list.append(self.LDA_model)
            self.coherence_values.append(self.get_coherenece())
            self.perplexity_values.append(self.get_perplexity())

        max_cohernece = max(self.coherence_values)
        self.max_index = self.coherence_values.index(max_cohernece)

    def optimal_number_of_topics(self, min_topics= 2, max_topics= 100, steps= 1):
        self.compute_LDA_Models(min_topics= min_topics, max_topics= max_topics, steps= steps)

        # Coherence Visualization
        x = range(min_topics, max_topics, steps)
        plt.figure(figsize=(16, 8))
        plt.plot(x, self.coherence_values)
        plt.title('Coherence')
        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence Score')
        plt.show()

        # Perplexity Visualization
        plt.figure(figsize=(16, 8))
        plt.plot(x, self.perplexity_values)
        plt.title('Perplexity')
        plt.xlabel('Number of Topics')
        plt.ylabel('Perplexity Score')
        plt.show()

    def make_dataframe(self, model, top_n= 20):
        # top_n : 상위 몇개의 topic을 보여줄 설정
        lda_topics_list = list()
        columns = ['Num of Topic']
        for i in range(top_n):
            columns.append('Word #{}'.format(i+1))

        for i in range(model.k):
            res = model.get_topic_words(i, top_n= top_n)
            tmp = ['Topic #{}'.format(i)]
            for w, _ in res:
                tmp.append(w)

            lda_topics_list.append(tmp)
        
        dataframe = pd.DataFrame(lda_topics_list, columns= columns)
        dataframe.set_index(keys= ['Num of Topic'], drop= True, inplace= True)
        dataframe = dataframe.T

        return lda_topics_list, dataframe

    def show_topics(self, model_index= 0, top_n= 20, show_n= 5):
        model = self.lda_model_list[model_index]
        topic_list, df = self.make_dataframe(model= model, top_n= top_n)
        
        for topic in topic_list:
            print(topic[0], end= '\t')
            print(', '.join(w for w in topic[1: show_n + 1]))

        self.save_excel(dataframe= df)

    def save_excel(self, dataframe):
        CHECK = True
        FILE_NAME = ''

        while CHECK:
            FILE_NAME = 'LDA_Topics_{}.xlsx'.format(self.save_file_name)
            FILE_PATH = self.path + 'Save_Excel/'
            if FILE_NAME not in os.listdir(FILE_PATH):
                CHECK = False
                dataframe.to_excel(FILE_PATH + FILE_NAME, index= True)

            else:
                self.save_file_name += 1

    def best_result(self, top_n= 20):
        self.show_topics(model_index= self.max_index, top_n= top_n, show_n= 5)

    def show_result(self, index= 0, top_n= 20):
        self.show_topics(model_index= index, top_n= top_n, show_n= 5)