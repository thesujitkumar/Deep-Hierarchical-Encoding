import sys
import os.path as path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from talo_feature_generation import ngram
import pandas as pd
import numpy as np
import pickle
import nltk
from talo_feature_generation.helpers import *
from talo_feature_generation.CountFeatureGenerator import CountFeatureGenerator # *
from talo_feature_generation.TfidfFeatureGenerator import TfidfFeatureGenerator #*
from talo_feature_generation.SvdFeatureGenerator import SvdFeatureGenerator
from talo_feature_generation.Word2VecFeatureGenerator import Word2VecFeatureGenerator
from talo_feature_generation.SentimentFeatureGenerator import SentimentFeatureGenerator
#from AlignmentFeatureGenerator import *
import os.path
nltk.download('vader_lexicon')
import sys
reload(sys)
import time
import argparse

sys.setdefaultencoding('utf8')

# data_type : { 0 : train, 1: validation, 2 : test}
def process(data_dir = '', processed_dir='', data_name='', data_type=0, fname_train='', fname_test='', fname_dev='' ):

    read = False
    out_fname_pkl = os.path.join(processed_dir, 'data.pkl')
    if not os.path.exists(out_fname_pkl):

        #body_train = pd.read_csv("train_bodies_processed.csv", encoding='utf-8')
        #stances_train = pd.read_csv("train_stances_processed.csv", encoding='utf-8')
        # training set
        #train = pd.merge(stances_train, body_train, how='left', on='Body ID')
        # train_fname = os.path.join(data_dir, "FNC_Bin_Train.csv")
        train_fname = os.path.join(data_dir, fname_train)
        train=pd.read_csv(train_fname, encoding='utf-8')
        print(' train df columns', train.columns)
        #print train

        targets = ['0', '1']
        targets_dict = dict(zip(targets, range(len(targets))))
        if data_name =='NELA':
            train['target'] = train['Label']
        else:
            train['target'] = train['Stance']
        print('train.shape:')
        print(train.shape)
        #exit(1)
        n_train = train.shape[0]
        #print n_train
        #exit(1)
        data = train
        # read test set, no 'Stance' column in test set -> target = NULL
        # concatenate training and test set
        test_flag = True
        if test_flag:
            #body_test = pd.read_csv("test_bodies_processed.csv", encoding='utf-8')
            #headline_test = pd.read_csv("test_stances_unlabeled.csv", encoding='utf-8')
            #test = pd.merge(headline_test, body_test, how="left", on="Body ID")
            test_fname = os.path.join(data_dir, fname_test)
            test=pd.read_csv(test_fname, encoding='utf-8')


            #development data_data

            dev_fname = os.path.join(data_dir, fname_dev)
            dev=pd.read_csv(dev_fname, encoding='utf-8')
            print("development set shape ",dev.shape)
            test = pd.concat((test, dev))
            print("merged test dev ",test.shape)
            #test.to_csv("NELA_merged_dev_test.csv",index=False)

            data = pd.concat((train, test)) # target = NaN for test set
            #print data
            print('data.shape: {}', data.shape)
            #print data.shape
            #exit(1)
            train = data[~data['target'].isnull()]
            #print train
            print('train.shape:', train.shape)

            test = data[data['target'].isnull()]
            #print test
            print('test.shape:', test.shape)

        #data = data.iloc[:100, :]

        #return 1

        print("generate unigram")
        data["Headline_unigram"] = data["Headline"].map(lambda x: preprocess_data(x))

        #Body
        if data_name == 'NELA':
            data["articleBody_unigram"] = data["Body"].map(lambda x: preprocess_data(x))
        else:
            data["articleBody_unigram"] = data["articleBody"].map(lambda x: preprocess_data(x))

        print("generate bigram")
        join_str = "_"
        data["Headline_bigram"] = data["Headline_unigram"].map(lambda x: ngram.getBigram(x, join_str))
        data["articleBody_bigram"] = data["articleBody_unigram"].map(lambda x: ngram.getBigram(x, join_str))

        print("generate trigram")
        join_str = "_"
        data["Headline_trigram"] = data["Headline_unigram"].map(lambda x: ngram.getTrigram(x, join_str))
        data["articleBody_trigram"] = data["articleBody_unigram"].map(lambda x: ngram.getTrigram(x, join_str))

        out_fname_pkl = os.path.join(processed_dir, 'data.pkl')
        with open(out_fname_pkl, 'wb') as outfile:
            pickle.dump(data, outfile, -1)
            print('dataframe saved in data.pkl')

        out_fname_csv= os.path.join(processed_dir, 'data.csv')
        data.to_csv(out_fname_csv, index=False,  encoding='utf-8')

    else:
        # out_fname_csv = os.path.join(processed_dir, 'data.csv')
        # data = pd.read_csv(out_fname_csv, encoding='utf-8')
        out_fname_pkl = os.path.join(processed_dir, 'data.pkl')
        with open(out_fname_pkl, 'rb') as infile:
            data = pickle.load(infile)
            print('data loaded')
            print('data.shape:')
            print(data.shape)
            if data_name == 'NELA':
                 data = data.rename(columns = {'Body':"articleBody"}) #
    #return 1

    # define feature generators
    countFG    = CountFeatureGenerator(data_name= data_name, data_dir=data_dir, processed_dir=processed_dir)
    tfidfFG    = TfidfFeatureGenerator(data_dir=data_dir, processed_dir=processed_dir)
    svdFG      = SvdFeatureGenerator(data_dir=data_dir, processed_dir=processed_dir)
    #word2vecFG = Word2VecFeatureGenerator(data_dir=data_dir, processed_dir=processed_dir)
    sentiFG    = SentimentFeatureGenerator(data_dir=data_dir, processed_dir=processed_dir)
    #walignFG   = AlignmentFeatureGenerator()
    #generators = [countFG, tfidfFG, svdFG, word2vecFG, sentiFG]
    generators = [countFG, tfidfFG, svdFG,  sentiFG]
    #generators = [svdFG, word2vecFG, sentiFG]
    #generators = [tfidfFG]
    #generators = [countFG]
    #generators = [walignFG]

    for g in generators[:]:
        g.process(data)

    for g in generators[:]:
        g.read('train')

    #for g in generators:
    #    g.read('test')

    print 'done'


if __name__ == "__main__":
    t1 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/NELA_Data',
                        help='path to dataset')
    parser.add_argument('--data_name', default='NELA',
                        help='Name of dataset')
    parser.add_argument('--input_file_train', default='train.csv',
                            help='Name of input  csv file')
    parser.add_argument('--input_file_test', default='test.csv',
                                help='Name of input  csv file')
    parser.add_argument('--input_file_dev', default='dev.csv',
                                    help='Name of input  csv file')
    # parser.add_argument('--data_type', default='dev',
    #                             help='Type of data file : test/train/dev')
    args = parser.parse_args()
    print('args : ', args)
    # data_dir = os.path.join('..', 'data', 'raw_data')
    # processed_dir =  os.path.join('..', 'data', 'processed_data')

    base_ip_dir = os.path.join(args.data, 'Raw_Data')
    parsed_dir = os.path.join(args.data, 'Parsed_Data')
    info_dir = os.path.join(args.data, 'Info_File')
    processed_dir = os.path.join(args.data, 'processed_data_feature_talo')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    #fname = "FNC_Bin_Dev.csv"
    # fname_train = "IOST_train_ver-2.csv"
    # fname_test = "IOST_test_ver-2.csv"
    # fname_dev="IOST_dev_ver-2.csv"
    process(data_dir=base_ip_dir, processed_dir=processed_dir, data_name = args.data_name, fname_train= args.input_file_train, fname_test= args.input_file_test, fname_dev=args.input_file_dev )
    t2 = time.time()
    #   Copyright 2017 Cisco Systems, Inc.
    print('Time taken : {}'.format(t2-t1))
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
