from FeatureGenerator import *
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from helpers import *
import os
import scipy.sparse

def convert_csr_df(csr_matrix, word2id):
    id2word = dict(zip(list(word2id.values()), list(word2id.keys()) ))
    coo = csr_matrix.tocoo()
    doc_list = coo.row.ravel()
    word_id_list = coo.col.ravel()
    word_list = [id2word[word_id] for word_id in word_id_list]
    tf_idf_score = coo.data.ravel()
    data = {'doc': doc_list, 'word' : word_list, 'tf_idf': tf_idf_score}
    df = pd.DataFrame(data, columns=['doc', 'word', 'tf_idf'])
    return df




class TfidfFeatureGenerator(FeatureGenerator):


    def __init__(self, name='tfidfFeatureGenerator',  data_name ='', data_dir='', processed_dir=''):
        super(TfidfFeatureGenerator, self).__init__(name)
        self.name = name
        self.data_name = data_name
        self.data_dir = data_dir
        self.processed_dir = processed_dir


    def process(self, df):
        filename_htfidf = "%s.headline.tfidf.pkl" % 'train'
        filename_htfidf_total = os.path.join(self.processed_dir, filename_htfidf)

        filename_btfidf = "%s.body.tfidf.pkl" % 'train'
        filename_btfidf_total = os.path.join(self.processed_dir, filename_btfidf)

        filename_simtfidf = "%s.sim.tfidf.pkl" % 'train'
        filename_simtfidf_total = os.path.join(self.processed_dir, filename_simtfidf)

        if os.path.exists(filename_htfidf_total) and  os.path.exists(filename_btfidf_total)  and os.path.exists(filename_btfidf_total):
            print(' \t Feature file for TF-IDF : {}, {} , {} already exists! '.format(filename_htfidf_total, filename_btfidf_total, filename_simtfidf_total ))
            return 1
        print(' \t Feature file for TF-IDF : {}, {} , {} does not exists, Creating it! '.format(filename_htfidf_total, filename_btfidf_total, filename_simtfidf_total ))
        # 1). create strings based on ' '.join(Headline_unigram + articleBody_unigram) [ already stemmed ]
        def cat_text(x):
            res =  ' '.join(x['Headline_unigram']) + ' ' + ' '.join(x['articleBody_unigram']) #'%s %s' % (' '.join(x['Headline_unigram']), ' '.join(x['articleBody_unigram']))
            return res
        df["all_text"] = list(df.apply(cat_text, axis=1))
        # print(df["all_text"])
        n_train = df[~df['target'].isnull()].shape[0]
        print ('tfidf, n_train:',n_train)
        n_test = df[df['target'].isnull()].shape[0]
        print ('tfidf, n_test:',n_test)

        # 2). fit a TfidfVectorizer on the concatenated strings
        # 3). sepatately transform ' '.join(Headline_unigram) and ' '.join(articleBody_unigram)
        vec = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2)
        vec.fit(df["all_text"]) # Tf-idf calculated on the combined training + test set
        vocabulary = vec.vocabulary_
        #word_list = list(vocabulary.keys())
        #word_list.sort(key = lambda item: vocabulary[item])

        vecH = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        xHeadlineTfidf = vecH.fit_transform(df['Headline_unigram'].map(lambda x: ' '.join(x))) # use ' '.join(Headline_unigram) instead of Headline since the former is already stemmed
        vocabulary = vecH.vocabulary_
        print('xHeadlineTfidf.shape:', xHeadlineTfidf.shape)

        # save train and test into separate files
        xHeadlineTfidfTrain = xHeadlineTfidf[:n_train, :]
        outfilename_htfidf_train = "train.headline.tfidf.pkl"
        outfilename_htfidf_train_total  = os.path.join( self.processed_dir, "train.headline.tfidf.pkl")
        with open(outfilename_htfidf_train_total, "wb") as outfile:
            pickle.dump(xHeadlineTfidfTrain, outfile, -1)
        print('headline tfidf features of training set saved in %s' % outfilename_htfidf_train)
        # df_count_feature =  convert_csr_df(xHeadlineTfidfTrain, vocabulary)
        # fname_out = os.path.join(self.processed_dir, "train.headline.tfidf.xlsx")
        # df_count_feature.to_excel(fname_out, index = False)
        if n_test > 0:
            # test set is available
            xHeadlineTfidfTest = xHeadlineTfidf[n_train:, :]
            outfilename_htfidf_test = "test.headline.tfidf.pkl"
            outfilename_htfidf_test_total = os.path.join(self.processed_dir, "test.headline.tfidf.pkl")
            with open(outfilename_htfidf_test_total, "wb") as outfile:
                pickle.dump(xHeadlineTfidfTest, outfile, -1)
            print('headline tfidf features of test set saved in %s' % outfilename_htfidf_test)
            # df_count_feature =  convert_csr_df(xHeadlineTfidfTest, vocabulary)
            # fname_out = os.path.join(self.processed_dir, "test.headline.tfidf.xlsx")
            # df_count_feature.to_excel(fname_out, index = False)


        vecB = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        xBodyTfidf = vecB.fit_transform(df['articleBody_unigram'].map(lambda x: ' '.join(x)))
        vocabulary = vecB.vocabulary_
        print('xBodyTfidf.shape:', xBodyTfidf.shape)

        # save train and test into separate files
        xBodyTfidfTrain = xBodyTfidf[:n_train, :]
        outfilename_btfidf_train = "train.body.tfidf.pkl"
        outfilename_btfidf_total = os.path.join(self.processed_dir, "train.body.tfidf.pkl")
        with open(outfilename_btfidf_total, "wb") as outfile:
            pickle.dump(xBodyTfidfTrain, outfile, -1)
        print('body tfidf features of training set saved in %s' % outfilename_btfidf_train)
        # df_count_feature =  convert_csr_df(xBodyTfidfTrain, vocabulary)
        # fname_out = os.path.join(self.processed_dir, "train.body.tfidf.xlsx")
        # df_count_feature.to_excel(fname_out, index = False)
        if n_test > 0:
            # test set is availble
            xBodyTfidfTest = xBodyTfidf[n_train:, :]
            outfilename_btfidf_test = "test.body.tfidf.pkl"
            outfilename_btfidf_test_total = os.path.join(self.processed_dir, "test.body.tfidf.pkl")
            with open(outfilename_btfidf_test_total, "wb") as outfile:
                pickle.dump(xBodyTfidfTest, outfile, -1)
            print('body tfidf features of test set saved in %s' % outfilename_btfidf_test)
            # df_count_feature =  convert_csr_df(xBodyTfidfTest, vocabulary)
            # fname_out = os.path.join(self.processed_dir, "test.body.tfidf.xlsx")
            # df_count_feature.to_excel(fname_out, index = False)


        # 4). compute cosine similarity between headline tfidf features and body tfidf features
        simTfidf = np.asarray(map(cosine_sim, xHeadlineTfidf, xBodyTfidf))[:, np.newaxis]
        print('simTfidf.shape:', simTfidf.shape)
        simTfidfTrain = simTfidf[:n_train]
        outfilename_simtfidf_train = "train.sim.tfidf.pkl"
        outfilename_simtfidf_train_total = os.path.join(self.processed_dir, "train.sim.tfidf.pkl")
        with open(outfilename_simtfidf_train_total, "wb") as outfile:
            pickle.dump(simTfidfTrain, outfile, -1)
        print('tfidf sim. features of training set saved in %s' % outfilename_simtfidf_train)
        df_count_feature = pd.DataFrame(data=simTfidfTrain)
        fname_out = os.path.join(self.processed_dir, "train.sim.tfidf.csv")
        df_count_feature.to_csv(fname_out, index = False)

        if n_test > 0:
            # test set is available
            simTfidfTest = simTfidf[n_train:]
            outfilename_simtfidf_test = "test.sim.tfidf.pkl"
            outfilename_simtfidf_test_total = os.path.join(self.processed_dir, "test.sim.tfidf.pkl")

            with open(outfilename_simtfidf_test_total, "wb") as outfile:
                pickle.dump(simTfidfTest, outfile, -1)
            print('tfidf sim. features of test set saved in %s' % outfilename_simtfidf_test)
            df_count_feature = pd.DataFrame(data=simTfidfTest)
            fname_out = os.path.join(self.processed_dir, "test.sim.tfidf.csv")
            df_count_feature.to_csv(fname_out, index = False)
        return 1


    def read(self, header='train'):

        filename_htfidf = "%s.headline.tfidf.pkl" % header
        filename_htfidf_total = os.path.join(self.processed_dir, filename_htfidf)
        with open(filename_htfidf_total, "rb") as infile:
            xHeadlineTfidf = pickle.load(infile)

        filename_btfidf = "%s.body.tfidf.pkl" % header
        filename_btfidf_total = os.path.join(self.processed_dir, filename_btfidf)
        with open(filename_btfidf_total, "rb") as infile:
            xBodyTfidf = pickle.load(infile)

        filename_simtfidf = "%s.sim.tfidf.pkl" % header
        filename_simtfidf_total = os.path.join(self.processed_dir, filename_simtfidf)
        with open(filename_simtfidf_total, "rb") as infile:
            print(infile)
            simTfidf = pickle.load(infile)

        print('xHeadlineTfidf.shape:', xHeadlineTfidf.shape)
        #print type(xHeadlineTfidf)
        print('xBodyTfidf.shape:', xBodyTfidf.shape)
        #print type(xBodyTfidf)
        print('simTfidf.shape',  simTfidf.shape)
        #print type(simTfidf)

        return [xHeadlineTfidf, xBodyTfidf, simTfidf.reshape(-1, 1)]
        #return [simTfidf.reshape(-1, 1)]

 #   Copyright 2017 Cisco Systems, Inc.
 #
 #   Licensed under the Apache License, Version 2.0 (the "License");
 #   you may not use this file except in compliance with the License.
 #   You may obtain a copy of the License at
 #
 #     http://www.apache.org/licenses/LICENSE-2.0
 #
 #   Unless required by applicable law or agreed to in writing, software
 #   distributed under the License is distributed on an "AS IS" BASIS,
 #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 #   See the License for the specific language governing permissions and
 #   limitations under the License.
