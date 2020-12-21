#!/usr/bin/env python

# importing required modules
import pandas as pd
from collections import Counter

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

from datetime import datetime


# define a class
class AnomalyMIL1553:
    
    #constructor
    def __init__(self, 
                 training_file, # input training data
                 verbose = True, # whether to spit out the progress
                 pca_n_components = 10,  # how many PCA components
                 word_vectorizer_min_df = 2): #minimum words in the sentence
        
        self.verbose = verbose
        self.pca_n_components  = pca_n_components
        self.word_vectorizer_min_df = word_vectorizer_min_df

        # whether the model has been initialized with training data yet
        self.initialized = False
        
        #load the data
        self.xdata = self.load_data(data_file = training_file, 
                               min_df  = word_vectorizer_min_df,
                               n_components = pca_n_components)
        

    # method on loading the data
    def load_data(self, 
                  data_file, 
                  initialize = True,
                  min_df = 2,
                  n_components = 10):

        if not initialize and not self.initialized: raise SystemExit('The object should be initialized with a data first!')
            
        if self.verbose: print(datetime.now(), 'loading', data_file, '...')
        
        df = pd.read_csv(data_file)
        
        if self.verbose: print(datetime.now(), 'raw data shape:', df.shape)
            
        # handling addr, subaddr and rxtx columns
        df['addr'] = df['addr'].astype('object')
        df['subaddr'] = df['subaddr'].astype('object')
        df['rxtx'] = df['rxtx'] * 1
        df['fulladdr'] = df['addr'].astype('str')+ '-' + df['subaddr'].astype('str')
        

        #handling the data column
        if self.verbose: print(datetime.now(), 'applying OneHotEncoder ...')
        if initialize: 
            self.encoder = OneHotEncoder()
            self.encoder.fit(df[['addr','subaddr', 'fulladdr']])
        addr_subaddr_fulladdr = self.encoder.transform(df[['addr','subaddr', 'fulladdr']])
        addr_subaddr_fulladdr = pd.DataFrame(addr_subaddr_fulladdr.toarray())

        
        f = lambda s: ' '.join([s[i:i + 4] for i in range(0, len(s), 4)])
        sentences = df['data'].apply(f)
        
       # if self.verbose: print(datetime.now(), 'applying CountVectorizer ...')
       # if initialize: 
       #     self.count = CountVectorizer(min_df = min_df)
       #     self.count.fit(sentences)
       # sentences_matrix = self.count.transform(sentences)
       # sentences_matrix = pd.DataFrame(sentences_matrix.toarray())
    
    	# using a vectorizer for word sequences
        if self.verbose: print(datetime.now(), 'applying TfidfVectorizer ...')
        if initialize: 
            self.tfidf = TfidfVectorizer()
            self.tfidf.fit(sentences)
        sentences_matrix = self.tfidf.transform(sentences)
        sentences_matrix = pd.DataFrame(sentences_matrix.toarray())

        # applying PCA on the vectorized matrix	
        if self.verbose: print(datetime.now(), 'applying PCA ...')
        if initialize: 
            self.pca = PCA(n_components = n_components)
            self.pca.fit(sentences_matrix)
        sentences_matrix_pca = self.pca.transform(sentences_matrix)
        sentences_matrix_pca = pd.DataFrame(sentences_matrix_pca)

        # combining all the feature data
        if self.verbose: print(datetime.now(), 'concatenating ...')
        xdata = pd.concat([df[['rxtx', 'gap', 'count']], 
                                addr_subaddr_fulladdr, 
                                sentences_matrix_pca], 
                               axis=1)
        
        # the model is now initialized with input training data
        if initialize: self.initialized = True
            
        if self.verbose: print(datetime.now(), 'data loaded.')
            
        return xdata
    

    # building all the models
    def model(self, 
              nu = [0.001, 0.01, 0.001], 
              contamination = [0.001, 0.001, 0.001]):

        if self.verbose: print(datetime.now(), 'the model is being created ...')
            
        self.ocsvm_rbf = OneClassSVM(gamma = 'scale', kernel = 'rbf', nu = nu[0]) 
        self.ocsvm_sigmoid = OneClassSVM(gamma = 'auto', kernel = 'sigmoid', nu = nu[1]) 
        self.ocsvm_linear = OneClassSVM(kernel = 'linear', nu = nu[2]) 

        self.ifo = IsolationForest(contamination = contamination[0])  
        self.lof = LocalOutlierFactor(contamination = contamination[1], novelty = True)
        self.ee = EllipticEnvelope(contamination = contamination[2])
        
        if self.verbose: print(datetime.now(), 'the model is ready.')


    # fitting the data on all the models
    def fit(self):
        
        if self.verbose: print(datetime.now(), 'the model is being fitted ...')
        self.ocsvm_rbf.fit(self.xdata)
        self.ocsvm_sigmoid.fit(self.xdata)
        self.ocsvm_linear.fit(self.xdata)
        self.ifo.fit(self.xdata)
        self.lof.fit(self.xdata)
        self.ee.fit(self.xdata)
        if self.verbose: print(datetime.now(), 'the model is fitted.')

    # predicting the labels and then averaging
    def predict(self, xdata = None):
        if xdata is None: xdata = self.xdata
            
        if self.verbose: print(datetime.now(), 'the prediction is running...')
            
        pred_ocsvm_rbf = self.ocsvm_rbf.predict(xdata)
        pred_ocsvm_sigmoid = self.ocsvm_sigmoid.predict(xdata)
        pred_ocsvm_linear = self.ocsvm_linear.predict(xdata)
        pred_ifo = self.ifo.predict(xdata)
        pred_lof = self.lof.predict(xdata)
        pred_ee = self.ee.predict(xdata)

        pred = (
            (pred_ocsvm_rbf == -1) * 1 + 
            (pred_ocsvm_sigmoid == -1) * 1 + 
            (pred_ocsvm_linear == -1) * 1 + 
            (pred_ifo == -1) * 1 +
            (pred_lof == -1) * 1 +
            (pred_ee == -1) * 1
        ) / 6
        
        if self.verbose: print(datetime.now(), 'the prediction is ready.')

        return pred


