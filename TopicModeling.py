from logging import warning
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np

import gensim
from gensim import corpora
from gensim.models import TfidfModel, CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import common_corpus

import pyLDAvis.gensim_models
import matplotlib.pyplot as plt

class LDA_Models:
    def __init__(self, tokenized_doc, path= './'):
        self.path = path
        self.tokenized_doc = tokenized_doc
        self.dictionary = corpora.Dictionary(tokenized_doc)
        
        # Bag of Words corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in tokenized_doc]

        # TF-IDF corpus
        tfidf = TfidfModel(self.corpus)
        self.corpus_tfidf = tfidf[self.corpus]

        self.today = np.int64(datetime.now().strftime('%Y%m%d') + '001')

        # Initialize
        # self.model = LdaModel(common_corpus, num_topics= 10)
        self.model_list = list()
        self.min_topics = -1
        self.max_topics = -1

    def LDA(self, num_topics= 10, use_tfidf = True):
        if use_tfidf:
            model = LdaModel(self.corpus_tfidf, num_topics= num_topics, id2word= self.dictionary)
            return model

        else:
            model = LdaModel(self.corpus, num_topics= num_topics, id2word= self.dictionary)
            return model

    def perplexityModel(self, min_topics= 2, max_topics= 20, is_return = False):
        '''
            Perplexity(PPL)
            의미 : 확률 모델이 결과를 얼마나 정확하게 예측하는지 판단, 값이 낮을수록 예측이 정확하게 이루어짐
            용도 : 동일 모델 내 파라미터에 따른 성능 평가할 때 주로 사용
            한계 : Perplexity가 낮다고 해서, 결과가 해석에 용이하다는 의미는 아님
        '''
        perplexity_values = []      # 초기화

        for model in self.model_list:
            perplexity_values.append(model.log_perplexity(self.corpus))

        min_values = min(perplexity_values)
        print('MIN Perplexity:', perplexity_values.index(min_values) + min_topics)

        # 시각화
        x = range(min_topics, max_topics + 1)
        plt.figure(figsize= (16, 8))
        plt.plot(x, perplexity_values)
        plt.title('LDA Perplexity')
        plt.xlabel('number of topics')
        plt.xticks(x)
        plt.ylabel('perplexity score')
        plt.show()

        if is_return:
            return perplexity_values.index(min_values) + min_topics

    def coherenceModel(self, min_topics= 2, max_topics= 20, top_topics= 20, is_return = False):
        '''
            의미 : 토픽이 얼마나 의미론적으로 일관성이 있는지 판단, 값이 높을수록 의미론적으로 일관성이 높음
            용도 : 해당 모델이 얼마나 실제로 의미 있는 결과를 내는 확인
        '''
        coherence_values = []       # 초기화

        for model in self.model_list:
            coherence_model = CoherenceModel(model= model, texts= self.tokenized_doc, dictionary= self.dictionary, topn= top_topics)
            coherence_lda = coherence_model.get_coherence()
            coherence_values.append(coherence_lda)

        max_values = max(coherence_values)
        print('MAX Coherence:', coherence_values.index(max_values) + min_topics)

        # 시각화
        x = range(min_topics, max_topics + 1)
        plt.figure(figsize= (16, 8))
        plt.plot(x, coherence_values)
        plt.title('LDA Coherence')
        plt.xlabel('number of topics')
        plt.xticks(x)
        plt.ylabel('coherence score')
        plt.show()

        if is_return:
            return coherence_values.index(max_values) + min_topics

    def visualization(self, num_topics= 10, use_tfidf = True):
        model = LdaModel(common_corpus, num_topics= 10)     # 초기화

        if len(self.model_list) == 0:
            model = self.LDA(num_topics= num_topics, use_tfidf= use_tfidf)
        
        else:
            model = self.model_list[num_topics]

        self.lda_vis = pyLDAvis.gensim_models.prepare(model, self.corpus, self.dictionary)

    def saveModel(self):
        CHECK = True
        FILE_NAME = ''      # 변수 초기화

        while CHECK:
            FILE_NAME = 'LDA_visualization_{}.html'.format(self.today)
            if FILE_NAME not in os.listdir(self.path):
                CHECK = False
                pyLDAvis.save_html(self.lda_vis, self.path + FILE_NAME)

            else:
                self.today += 1

    def calculation_lda(self, num_topics= 10, use_tfidf = True):
        num_topics = num_topics

        if len(self.model_list) > 0:
            num_topics = num_topics - self.min_topics
            
        self.visualization(num_topics= num_topics, use_tfidf= use_tfidf)
        self.saveModel()

    def perplex_and_coherence(self, min_topics= 2, max_topics= 20, top_topics= 10, use_tfidf = True):
        self.model_list = list()
        self.min_topics = min_topics
        self.max_topics = max_topics

        for num_topic in range(min_topics, max_topics + 1):
            model = self.LDA(num_topics= num_topic, use_tfidf= use_tfidf)
            self.model_list.append(model)
            
        min_values = self.perplexityModel(self.min_topics, self.max_topics, is_return= True)
        max_values = self.coherenceModel(self.min_topics, self.max_topics, top_topics, is_return= True)
        
        self.calculation_lda(num_topics= max_values, use_tfidf= use_tfidf)