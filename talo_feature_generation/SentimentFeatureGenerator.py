from FeatureGenerator import *
import pandas as pd
import numpy as np
import pickle
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from helpers import *
import os


class SentimentFeatureGenerator(FeatureGenerator):


    def __init__(self, name='sentimentFeatureGenerator', data_name ='',  data_dir='', processed_dir=''):
        super(SentimentFeatureGenerator, self).__init__(name)
        self.name = name
        self.data_name = data_name
        self.data_dir = data_dir
        self.processed_dir = processed_dir


    def process(self, df):

        filename_hsenti = "%s.headline.senti.pkl" % 'train'
        filename_hsenti_total = os.path.join(self.processed_dir, filename_hsenti)

        filename_bsenti = "%s.body.senti.pkl" % 'train'
        filename_bsenti_total = os.path.join(self.processed_dir, filename_bsenti)




        if os.path.exists(filename_hsenti_total) and  os.path.exists(filename_bsenti_total):
            print(' \t  Feature file for SentimentFeature : {}, {}  already exists! '.format(filename_hsenti_total, filename_bsenti_total ))
            return 1
        print(' \t Feature file for SentimentFeature  : {}, {}  does not exists, Creating it! '.format(filename_hsenti_total, filename_bsenti_total))

        print('generating sentiment features for headline')

        n_train = df[~df['target'].isnull()].shape[0]
        n_test = df[df['target'].isnull()].shape[0]

        # calculate the polarity score of each sentence then take the average
        sid = SentimentIntensityAnalyzer()
        def compute_sentiment(sentences):
            result = []
            for sentence in sentences:
                vs = sid.polarity_scores(sentence)
                result.append(vs)
            return pd.DataFrame(result).mean()

        #df['headline_sents'] = df['Headline'].apply(lambda x: sent_tokenize(x.decode('utf-8')))
        df['headline_sents'] = df['Headline'].apply(lambda x: sent_tokenize(x))
        df = pd.concat([df, df['headline_sents'].apply(lambda x: compute_sentiment(x))], axis=1)
        df.rename(columns={'compound':'h_compound', 'neg':'h_neg', 'neu':'h_neu', 'pos':'h_pos'}, inplace=True)
        #print 'df:'
        #print df
        #print df.columns
        #print df.shape
        headlineSenti = df[['h_compound','h_neg','h_neu','h_pos']].values
        print('headlineSenti.shape:', headlineSenti.shape)

        headlineSentiTrain = headlineSenti[:n_train, :]
        outfilename_hsenti_train = "train.headline.senti.pkl"
        outfilename_hsenti_train_total = os.path.join(self.processed_dir, outfilename_hsenti_train)
        with open(outfilename_hsenti_train_total, "wb") as outfile:
            pickle.dump(headlineSentiTrain, outfile, -1)
        print('headline sentiment features of training set saved in %s' % outfilename_hsenti_train)
        df_sent_feature = pd.DataFrame(data=headlineSentiTrain, columns=['h_compound', 'h_neg', 'h_neu', 'h_pos'])
        fname_out = os.path.join(self.processed_dir, "train.headline.senti.csv")
        df_sent_feature.to_csv(fname_out, index = False)
        if n_test > 0:
            # test set is available
            headlineSentiTest = headlineSenti[n_train:, :]
            outfilename_hsenti_test = "test.headline.senti.pkl"
            outfilename_hsenti_test_total = os.path.join(self.processed_dir, outfilename_hsenti_test)
            with open(outfilename_hsenti_test_total, "wb") as outfile:
                pickle.dump(headlineSentiTest, outfile, -1)
            print('headline sentiment features of test set saved in %s' % outfilename_hsenti_test)
            df_sent_feature = pd.DataFrame(data=headlineSentiTest, columns=['h_compound', 'h_neg', 'h_neu', 'h_pos'])
            fname_out = os.path.join(self.processed_dir, "test.headline.senti.csv")
            df_sent_feature.to_csv(fname_out, index = False)
        print('headine senti done')

        #return 1

        print('for body')
        #df['body_sents'] = df['articleBody'].map(lambda x: sent_tokenize(x.decode('utf-8')))
        df['body_sents'] = df['articleBody'].map(lambda x: sent_tokenize(x))
        df = pd.concat([df, df['body_sents'].apply(lambda x: compute_sentiment(x))], axis=1)
        df.rename(columns={'compound':'b_compound', 'neg':'b_neg', 'neu':'b_neu', 'pos':'b_pos'}, inplace=True)

        bodySenti = df[['b_compound','b_neg','b_neu','b_pos']].values
        print('bodySenti.shape:', bodySenti.shape)

        bodySentiTrain = bodySenti[:n_train, :]
        outfilename_bsenti_train = "train.body.senti.pkl"
        outfilename_bsenti_train_total = os.path.join(self.processed_dir, outfilename_bsenti_train)
        with open(outfilename_bsenti_train_total, "wb") as outfile:
            pickle.dump(bodySentiTrain, outfile, -1)
        print('body sentiment features of training set saved in %s' % outfilename_bsenti_train)
        df_sent_feature = pd.DataFrame(data=bodySentiTrain, columns=['b_compound', 'b_neg', 'b_neu', 'b_pos'])
        fname_out = os.path.join(self.processed_dir, "train.body.senti.csv")
        df_sent_feature.to_csv(fname_out, index = False)
        if n_test > 0:
            # test set is available
            bodySentiTest = bodySenti[n_train:, :]
            outfilename_bsenti_test = "test.body.senti.pkl"
            outfilename_bsenti_test_total = os.path.join(self.processed_dir, outfilename_bsenti_test)
            with open(outfilename_bsenti_test_total, "wb") as outfile:
                pickle.dump(bodySentiTest, outfile, -1)
            print('body sentiment features of test set saved in %s' % outfilename_bsenti_test)
            df_sent_feature = pd.DataFrame(data=bodySentiTest, columns=['b_compound', 'b_neg', 'b_neu', 'b_pos'])
            fname_out = os.path.join(self.processed_dir, "test.body.senti.csv")
            df_sent_feature.to_csv(fname_out, index = False)
        print('body senti done')

        return 1


    def read(self, header='train'):

        filename_hsenti = "%s.headline.senti.pkl" % header
        filename_hsenti_total = os.path.join(self.processed_dir, filename_hsenti)
        with open(filename_hsenti_total, "rb") as infile:
            headlineSenti = pickle.load(infile)

        filename_bsenti = "%s.body.senti.pkl" % header
        filename_bsenti_total = os.path.join(self.processed_dir, filename_bsenti)
        with open(filename_bsenti_total, "rb") as infile:
            bodySenti = pickle.load(infile)

        print('headlineSenti.shape:', headlineSenti.shape)
        #print type(headlineSenti)
        print('bodySenti.shape:', bodySenti.shape)
        #print type(bodySenti)

        return [headlineSenti, bodySenti]

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
