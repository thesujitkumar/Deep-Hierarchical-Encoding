from FeatureGenerator import *
import ngram
import pickle
import pandas as pd
from nltk.tokenize import sent_tokenize
from helpers import *
import hashlib
import os


class CountFeatureGenerator(FeatureGenerator):


    def __init__(self, name='countFeatureGenerator',  data_name ='', data_dir='', processed_dir='', ):
        super(CountFeatureGenerator, self).__init__(name)
        self.name = name
        self.data_name = data_name
        self.data_dir = data_dir
        self.processed_dir = processed_dir


    def process(self, df):
        filename_bcf = "{}.{}.pkl".format('train', self.name)
        filename_bcf_total = os.path.join(self.processed_dir, filename_bcf)
        if os.path.exists(filename_bcf_total):
            print('\t Count Feature file : {} already exists! '.format(filename_bcf_total))
            return 1
        print('\t Count Feature file : {} does not exists!, Creating it '.format(filename_bcf_total))
        grams = ["unigram", "bigram", "trigram"]
        feat_names = ["Headline", "articleBody"]
        print("generate counting features")
        for feat_name in feat_names:
            for gram in grams:
                df["count_of_%s_%s" % (feat_name, gram)] = list(df.apply(lambda x: len(x[feat_name + "_" + gram]), axis=1))
                df["count_of_unique_%s_%s" % (feat_name, gram)] = \
		            list(df.apply(lambda x: len(set(x[feat_name + "_" + gram])), axis=1))
                df["ratio_of_unique_%s_%s" % (feat_name, gram)] = \
                    map(try_divide, df["count_of_unique_%s_%s"%(feat_name,gram)], df["count_of_%s_%s"%(feat_name,gram)])

        # overlapping n-grams count
        for gram in grams:
            df["count_of_Headline_%s_in_articleBody" % gram] = \
                list(df.apply(lambda x: sum([1. for w in x["Headline_" + gram] if w in set(x["articleBody_" + gram])]), axis=1))
            df["ratio_of_Headline_%s_in_articleBody" % gram] = \
                map(try_divide, df["count_of_Headline_%s_in_articleBody" % gram], df["count_of_Headline_%s" % gram])

        # number of sentences in headline and body
        for feat_name in feat_names:
            #df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(sent_tokenize(x.decode('utf-8').encode('ascii', errors='ignore'))))
            df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(sent_tokenize(x)))
            #print df['len_sent_%s' % feat_name]

        # dump the basic counting features into a file
        feat_names = [ n for n in df.columns \
                if "count" in n \
                or "ratio" in n \
                or "len_sent" in n]

        # binary refuting features
        _refuting_words = [
            'fake',
            'fraud',
            'hoax',
            'false',
            'deny', 'denies',
            # 'refute',
            'not',
            'despite',
            'nope',
            'doubt', 'doubts',
            'bogus',
            'debunk',
            'pranks',
            'retract'
        ]

        _hedging_seed_words = [
            'alleged', 'allegedly',
            'apparently',
            'appear', 'appears',
            'claim', 'claims',
            'could',
            'evidently',
            'largely',
            'likely',
            'mainly',
            'may', 'maybe', 'might',
            'mostly',
            'perhaps',
            'presumably',
            'probably',
            'purported', 'purportedly',
            'reported', 'reportedly',
            'rumor', 'rumour', 'rumors', 'rumours', 'rumored', 'rumoured',
            'says',
            'seem',
            'somewhat',
            # 'supposedly',
            'unconfirmed'
        ]

        #df['refuting_words_in_headline'] = df['Headline'].map(lambda x: 1 if w in x else 0 for w in _refuting_words)
        #df['hedging_words_in_headline'] = df['Headline'].map(lambda x: 1 if w in x else 0 for w in _refuting_words)
        #check_words = _refuting_words + _hedging_seed_words
        check_words = _refuting_words
        for rf in check_words:
            fname = '%s_exist' % rf
            feat_names.append(fname)
            df[fname] = df['Headline'].map(lambda x: 1 if rf in x else 0)

        # number of body texts paired up with the same headline
        #df['headline_hash'] = df['Headline'].map(lambda x: hashlib.md5(x).hexdigest())
        #nb_dict = df.groupby(['headline_hash'])['Body ID'].nunique().to_dict()
        #df['n_bodies'] = df['headline_hash'].map(lambda x: nb_dict[x])
        #feat_names.append('n_bodies')
        # number of headlines paired up with the same body text
        #nh_dict = df.groupby(['Body ID'])['headline_hash'].nunique().to_dict()
        #df['n_headlines'] = df['Body ID'].map(lambda x: nh_dict[x])
        #feat_names.append('n_headlines')
        print('BasicCountFeatures:')

        # split into train, test portion and save in separate files
        train = df[~df['target'].isnull()]
        print('train:')
        # print train[['Headline_unigram','Body ID', 'count_of_Headline_unigram']]
        xBasicCountsTrain = train[feat_names].values
        outfilename_bcf_train ="{}.{}.pkl".format('train', self.name)  # "train." +  self.name + ".pkl" #"train.basic.pkl"
        outfilename_bcf_train_total = os.path.join(self.processed_dir, outfilename_bcf_train)
        with open(outfilename_bcf_train_total, "wb") as outfile:
            pickle.dump(feat_names, outfile, -1)
            pickle.dump(xBasicCountsTrain, outfile, -1)
        #print 'basic counting features for training saved in %s' % outfilename_bcf_train
        df_count_feature = pd.DataFrame(data=xBasicCountsTrain, columns = feat_names)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        #fname_out = os.path.join(self.processed_dir, "train.basic.xlsx")
        fname_out = os.path.join(self.processed_dir, "train." +  self.name + ".csv")
        df_count_feature.to_csv(fname_out, index = False)


        test = df[df['target'].isnull()]
        print('test:')
        # print test[['Headline_unigram','Body ID', 'count_of_Headline_unigram']]
        #return 1
        if test.shape[0] > 0:
            # test set exists
            print('saving test set')
            xBasicCountsTest = test[feat_names].values
            outfilename_bcf_test = "{}.{}.pkl".format('train', self.name)
            outfilename_bcf_test_total = os.path.join(self.processed_dir, outfilename_bcf_test)   #self.processed_dir, "test.basic.pkl")
            with open(outfilename_bcf_test_total, 'wb') as outfile:
                pickle.dump(feat_names, outfile, -1)
                pickle.dump(xBasicCountsTest, outfile, -1)
                print('basic counting features for test saved in {}'.format(outfilename_bcf_test))
            df_count_feature = pd.DataFrame(data=xBasicCountsTest, columns = feat_names)
            #fname_out = os.path.join(self.processed_dir, "test.basic.xlsx")
            fname_out = os.path.join(self.processed_dir, "test." +  self.name + ".csv")
            df_count_feature.to_csv(fname_out, index = False)

        return 1


    def read(self, header='train'):
        #filename_bcf = "%s.basic.pkl" % header
        filename_bcf = "{}.{}.pkl".format(header, self.name)
        #outfilename_bcf_train =  "train." +  self.name + ".pkl" #"train.basic.pkl"
        filename_bcf_total = os.path.join(self.processed_dir, filename_bcf)
        with open(filename_bcf_total, "rb") as infile:
            feat_names = pickle.load(infile)
            xBasicCounts = pickle.load(infile)
            print('feature names: ', feat_names)
            print('xBasicCounts.shape:', xBasicCounts.shape)
            #print type(xBasicCounts)

        return [xBasicCounts]

if __name__ == '__main__':

    cf = CountFeatureGenerator()
    cf.read()

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
