from itertools import count
from operator import index
import os
from pickle import FALSE
import pandas as pd
import numpy as np
from datetime import datetime

# LDA Library
import tomotopy as tp

# Visualization
import matplotlib.pyplot as plt

class Tomotopy_LDA_Models:
    def __init__(self, tokenized_doc, path='./'):
        self.path = path
        self.tokenized_doc = tokenized_doc

        # LDA Parameter
        self.num_topics = 10
        self.min_cf = 10
        self.tw = 'IDF'
        self.rm_top = 0
        self.alpha = 0.1
        self.eta = 0.01
        self.seed = np.int64(datetime.now().strftime('%Y'))

        # etc
        self.today = datetime.now().strftime('%Y%m%d')
        self.save_file_name = np.int64(self.today + '001')
        self.show_detail = False
        self.model_index = 0                                     # 여러개의 LDA 결과 중 해당 결과를 볼 때 사용하는 변수

    def LDA(self, num_topics = 10):
        # num_topic : 토픽의 갯수
        # min_df : 
        # min_cf : 전체 말뭉치에서 N회 미만 등장한 단어들 제거
        # tw : tp.TermWeight의 항목 중 하나를 선택하여 입력(tp.TermWeight.ONE / tp.TermWeight.IDF / tp.TermWeight.PMI)
        # IDF(정보량) : 해당 사건이 발생한 확률의 역수에 로그를 취한 값
        # PMI(점별 상호정보량) : 문헌이 출현할 확률과 단어가 출현할 확률의 

        self.num_topics = num_topics

        termweight = ''
        if self.tw == 'ONE':
            termweight = tp.TermWeight.ONE
        elif self.tw == 'IDF':
            termweight = tp.TermWeight.IDF
        elif self.tw == 'PMI':
            termweight = tp.TermWeight.PMI
        
        self.LDA_model = tp.LDAModel(k= self.num_topics, min_cf= self.min_cf, tw= termweight, rm_top= self.rm_top, alpha= self.alpha, eta= self.eta, seed= self.seed)

        for token in self.tokenized_doc:
            self.LDA_model.add_doc(token)

    def train_model(self):
        # 초기화
        self.LDA_model.train(0)

        if self.show_detail:
            self.show_model_detail()

        print('Training...', flush= True)

        for i in range(0, self.max_train, self.term_train):
            self.LDA_model.train(self.term_train)
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

    def compute_LDA_Models(self):
        self.coherence_values = list()
        self.perplexity_values = list()
        self.lda_model_list = list()

        for num_topic in range(self.min_topics, self.max_topics, self.steps):
            self.LDA(num_topics= num_topic)
            self.train_model()

            self.lda_model_list.append(self.LDA_model)
            self.coherence_values.append(self.get_coherenece())
            self.perplexity_values.append(self.get_perplexity())

        max_cohernece = max(self.coherence_values)
        self.max_index = self.coherence_values.index(max_cohernece)

    def optimal_number_of_topics(self, min_topics= 2, max_topics= 100, steps= 1, min_cf= 10, tw= 'IDF', rm_top= 0, alpha= 0.1, eta= 0.01, max_train= 500, term_train= 100, show_detail= False):
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.steps = steps

        self.min_cf = min_cf
        self.tw = tw
        self.rm_top = rm_top
        self.alpha = alpha
        self.eta = eta
    
        self.max_train = max_train
        self.term_train = term_train

        self.show_detail = show_detail

        self.compute_LDA_Models()

        # Coherence Visualization
        x = range(self.min_topics, self.max_topics, self.steps)
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

    def show_topics(self, top_n= 20, show_n= 5):
        model = self.lda_model_list[self.model_index]
        topic_list, df = self.make_dataframe(model= model, top_n= top_n)
        
        for topic in topic_list:
            print(topic[0], end= '\t')
            print(', '.join(w for w in topic[1: show_n + 1]))

        self.save_excel(dataframe= df)
        self.save_model(model= model)

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

    def save_model(self, model):
        FILE_NAME = 'LDA_Model_{}.bin'.format(self.save_file_name)
        FILE_PATH = self.path + 'Save_Model/'

        model.save(FILE_PATH + FILE_NAME)

    def best_result(self, top_n= 20, show_n = 5):
        self.model_index = self.max_index
        self.show_topics(top_n= top_n, show_n= show_n)

    def show_result(self, topic_choice= 0, top_n= 20, show_n = 5):
        self.model_index = topic_choice - self.min_topics
        self.show_topics(top_n= top_n, show_n= show_n)