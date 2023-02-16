from FeatureGenerator import *
from TfidfFeatureGenerator import *
import pandas as pd
import numpy as np
from scipy.sparse import vstack
import pickle
from sklearn.decomposition import TruncatedSVD
from helpers import *
import os

class SvdFeatureGenerator(FeatureGenerator):


    def __init__(self, name='svdFeatureGenerator', data_name ='',  data_dir='', processed_dir=''):
        super(SvdFeatureGenerator, self).__init__(name)
        self.name = name
        self.data_name = data_name
        self.data_dir = data_dir
        self.processed_dir = processed_dir


    def process(self, df):
        filename_hsvd = "%s.headline.svd.pkl" % 'train'
        filename_hsvd_total = os.path.join(self.processed_dir, filename_hsvd)

        filename_bsvd = "%s.body.svd.pkl" % 'train'
        filename_bsvd_total = os.path.join(self.processed_dir, filename_bsvd)

        filename_simsvd = "%s.sim.svd.pkl" % 'train'
        filename_simsvd_total = os.path.join(self.processed_dir, filename_simsvd)

        if os.path.exists(filename_hsvd_total) and  os.path.exists(filename_bsvd_total)  and os.path.exists(filename_simsvd_total):
            print(' \t Feature file for SVDFeature : {}, {} , {} already exists! '.format(filename_hsvd_total, filename_bsvd_total, filename_simsvd_total ))
            return 1
        print(' \t Feature file for SVDFeature  : {}, {} , {} does not exists, Creating it! '.format(filename_hsvd_total, filename_bsvd_total, filename_simsvd_total ))
        n_train = df[~df['target'].isnull()].shape[0]
        print('SvdFeatureGenerator, n_train:',n_train)
        n_test  = df[df['target'].isnull()].shape[0]
        print('SvdFeatureGenerator, n_test:',n_test)

        tfidfGenerator = TfidfFeatureGenerator('tfidf',  data_dir=self.data_dir, processed_dir=self.processed_dir)
        featuresTrain = tfidfGenerator.read('train')
        xHeadlineTfidfTrain, xBodyTfidfTrain = featuresTrain[0], featuresTrain[1]

        xHeadlineTfidf = xHeadlineTfidfTrain
        xBodyTfidf = xBodyTfidfTrain
        if n_test > 0:
            # test set is available
            featuresTest  = tfidfGenerator.read('test')
            xHeadlineTfidfTest,  xBodyTfidfTest  = featuresTest[0],  featuresTest[1]
            xHeadlineTfidf = vstack([xHeadlineTfidfTrain, xHeadlineTfidfTest])
            xBodyTfidf = vstack([xBodyTfidfTrain, xBodyTfidfTest])

        # compute the cosine similarity between truncated-svd features
        svd = TruncatedSVD(n_components=50, n_iter=15)
        xHBTfidf = vstack([xHeadlineTfidf, xBodyTfidf])
        svd.fit(xHBTfidf) # fit to the combined train-test set (or the full training set for cv process)
        print('xHeadlineTfidf.shape:', xHeadlineTfidf.shape)
        xHeadlineSvd = svd.transform(xHeadlineTfidf)
        print('xHeadlineSvd.shape:', xHeadlineSvd.shape)

        xHeadlineSvdTrain = xHeadlineSvd[:n_train, :]
        outfilename_hsvd_train = "train.headline.svd.pkl"
        outfilename_hsvd_train_total = os.path.join(self.processed_dir, outfilename_hsvd_train)
        with open(outfilename_hsvd_train_total, "wb") as outfile:
            pickle.dump(xHeadlineSvdTrain, outfile, -1)
        print('headline svd features of training set saved in %s' % outfilename_hsvd_train)
        df_svd_feature = pd.DataFrame(data=xHeadlineSvdTrain)
        fname_out = os.path.join(self.processed_dir, "train.headline.svd.csv")
        df_svd_feature.to_csv(fname_out, index = False)
        if n_test > 0:
            # test set is available
            xHeadlineSvdTest = xHeadlineSvd[n_train:, :]
            outfilename_hsvd_test = "test.headline.svd.pkl"
            outfilename_hsvd_test_total = os.path.join(self.processed_dir, outfilename_hsvd_test)
            with open(outfilename_hsvd_test_total, "wb") as outfile:
                pickle.dump(xHeadlineSvdTest, outfile, -1)
            print('headline svd features of test set saved in %s' % outfilename_hsvd_test)
            df_svd_feature = pd.DataFrame(data=xHeadlineSvdTest)
            fname_out = os.path.join(self.processed_dir, "test.headline.svd.csv")
            df_svd_feature.to_csv(fname_out, index = False)
        xBodySvd = svd.transform(xBodyTfidf)
        print('xBodySvd.shape:', xBodySvd.shape)

        xBodySvdTrain = xBodySvd[:n_train, :]
        outfilename_bsvd_train = "train.body.svd.pkl"
        outfilename_bsvd_train_total = os.path.join(self.processed_dir, outfilename_bsvd_train)
        with open(outfilename_bsvd_train_total, "wb") as outfile:
            pickle.dump(xBodySvdTrain, outfile, -1)
        print('body svd features of training set saved in %s' % outfilename_bsvd_train)
        df_svd_feature = pd.DataFrame(data=xBodySvdTrain)
        fname_out = os.path.join(self.processed_dir, "train.body.svd.csv")
        df_svd_feature.to_csv(fname_out, index = False)
        if n_test > 0:
            # test set is available
            xBodySvdTest = xBodySvd[n_train:, :]
            outfilename_bsvd_test = "test.body.svd.pkl"
            outfilename_bsvd_test_total = os.path.join(self.processed_dir, outfilename_bsvd_test)
            with open(outfilename_bsvd_test_total, "wb") as outfile:
                pickle.dump(xBodySvdTest, outfile, -1)
            print('body svd features of test set saved in %s' % outfilename_bsvd_test)
            df_svd_feature = pd.DataFrame(data=xBodySvdTest)
            fname_out = os.path.join(self.processed_dir, "test.body.svd.csv")
            df_svd_feature.to_csv(fname_out, index = False)
        simSvd = np.asarray(map(cosine_sim, xHeadlineSvd, xBodySvd))[:, np.newaxis]
        print('simSvd.shape:', simSvd.shape)

        simSvdTrain = simSvd[:n_train]
        outfilename_simsvd_train = "train.sim.svd.pkl"
        outfilename_simsvd_train_total = os.path.join(self.processed_dir, outfilename_simsvd_train)
        with open(outfilename_simsvd_train_total, "wb") as outfile:
            pickle.dump(simSvdTrain, outfile, -1)
        print('svd sim. features of training set saved in %s' % outfilename_simsvd_train)
        df_svd_feature = pd.DataFrame(data=simSvdTrain)
        fname_out = os.path.join(self.processed_dir, "train.sim.svd.csv")
        df_svd_feature.to_csv(fname_out, index = False)
        if n_test > 0:
            # test set is available
            simSvdTest = simSvd[n_train:]
            outfilename_simsvd_test = "test.sim.svd.pkl"
            outfilename_simsvd_test_total = os.path.join(self.processed_dir, outfilename_simsvd_test)
            with open(outfilename_simsvd_test_total, "wb") as outfile:
                pickle.dump(simSvdTest, outfile, -1)
            print('svd sim. features of test set saved in %s' % outfilename_simsvd_test)
            df_svd_feature = pd.DataFrame(data=simSvdTest)
            fname_out = os.path.join(self.processed_dir, "test.sim.svd.csv")
            df_svd_feature.to_csv(fname_out, index = False)
        return 1


    def read(self, header='train'):

        filename_hsvd = "%s.headline.svd.pkl" % header
        filename_hsvd_total = os.path.join(self.processed_dir, filename_hsvd)
        with open(filename_hsvd_total, "rb") as infile:
            xHeadlineSvd = pickle.load(infile)

        filename_bsvd = "%s.body.svd.pkl" % header
        filename_bsvd_total = os.path.join(self.processed_dir, filename_bsvd)
        with open(filename_bsvd_total, "rb") as infile:
            xBodySvd = pickle.load(infile)

        filename_simsvd = "%s.sim.svd.pkl" % header
        filename_simsvd_total = os.path.join(self.processed_dir, filename_simsvd)
        with open(filename_simsvd_total, "rb") as infile:
            simSvd = pickle.load(infile)

        print('xHeadlineSvd.shape:', xHeadlineSvd.shape)
        #print type(xHeadlineSvd)
        print('xBodySvd.shape:', xBodySvd.shape)
        #print type(xBodySvd)
        print('simSvd.shape:', simSvd.shape)
        #print type(simSvd)

        return [xHeadlineSvd, xBodySvd, simSvd.reshape(-1, 1)]
        #return [simSvd.reshape(-1, 1)]

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
