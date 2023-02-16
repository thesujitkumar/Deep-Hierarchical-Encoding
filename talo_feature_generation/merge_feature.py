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

import pandas as pd

def merge_features(data_dir = '', processed_dir='', data_name='', data_type=0, fname_train='', fname_test='', fname_dev='' ):
    # In[2]:
    # define feature generators
    countFG    = CountFeatureGenerator(data_name= data_name, data_dir=data_dir, processed_dir=processed_dir)
    tfidfFG    = TfidfFeatureGenerator(data_dir=data_dir, processed_dir=processed_dir)
    svdFG      = SvdFeatureGenerator(data_dir=data_dir, processed_dir=processed_dir)
    #word2vecFG = Word2VecFeatureGenerator(data_dir=data_dir, processed_dir=processed_dir)
    sentiFG    = SentimentFeatureGenerator(data_dir=data_dir, processed_dir=processed_dir)
    #walignFG   = AlignmentFeatureGenerator()
    #generators = [countFG, tfidfFG, svdFG, word2vecFG, sentiFG]
    generators = [countFG, tfidfFG, svdFG,  sentiFG]

    for g in generators[:]:
        xBasicCounts = g.read('train')
        print(type(xBasicCounts))
    #test_count=pd.read_excel("test.basic.xlsx")


    # In[3]:


    test_count.columns


    # In[4]:


    no_word_head=test_count['count_of_Headline_unigram']
    no_word_body=test_count['count_of_articleBody_unigram']



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
    merge_features(data_dir=base_ip_dir, processed_dir=processed_dir, data_name = args.data_name, fname_train= args.input_file_train, fname_test= args.input_file_test, fname_dev=args.input_file_dev )
    #process(data_dir=base_ip_dir, processed_dir=processed_dir, data_name = args.data_name, fname_train= args.input_file_train, fname_test= args.input_file_test, fname_dev=args.input_file_dev )
    t2 = time.time()
    #   Copyright 2017 Cisco Systems, Inc.
    print('Time taken : {}'.format(t2-t1))
