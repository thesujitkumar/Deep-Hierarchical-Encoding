from __future__ import division
from __future__ import print_function

import os
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import pickle
import torch.nn.functional as F
import math
# IMPORT CONSTANTS
from treelstm import Constants
# NEURAL NETWORK MODULES/LAYERS
from treelstm import SimilarityTreeLSTM
# DATA HANDLING CLASSES
from treelstm import Vocab
# DATASET CLASS FOR SICK DATASET
from treelstm import Dataset
# METRICS CLASS FOR EVALUATION
from treelstm import Metrics
# UTILITY FUNCTIONS
from treelstm import utils
# TRAIN AND TEST HELPER FUNCTIONS
from treelstm import Trainer
# CONFIG PARSER
from config import parse_args
import pandas as pd
import time
import gc
from tqdm import tqdm


def label_exteract(data_dic):    # Function to exteract label
    label_list=[data_dic[idx]['headline']['label'] for idx in tqdm(data_dic, total = len(data_dic))]
    target_val=torch.LongTensor(label_list)
    del label_list
    return target_val

# MAIN BLOCK
def build_input():
    t_start = time.time()
    global args
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')

    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)


    train_dir =   os.path.join(args.data, 'train/')#'data/sick/train/'
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/') #'data/sick/test/' #
    print(train_dir, test_dir,dev_dir )



    # write unique words from all token files
    vocab_file_name = '{}_{}_{}d.vocab'.format(args.data_name, args.emb_name, args.input_dim)
    vocab_file = os.path.join(args.data, vocab_file_name ) # 'FNC_Bin_Data_glove_200d.vocab') #
    if not os.path.isfile(vocab_file):
        token_files_b = [os.path.join(split, 'b.toks') for split in [train_dir,test_dir,dev_dir]]
        token_files_a = [os.path.join(split, 'a.toks') for split in [train_dir,test_dir,dev_dir]]
        token_files = token_files_a + token_files_b
        vocab_file = os.path.join(args.data,  vocab_file_name) # 'FNC_Bin_Data_glove_200d.vocab')
        utils.build_vocab(token_files, vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=vocab_file,
                  data=[Constants.PAD_WORD, Constants.UNK_WORD,
                        Constants.BOS_WORD, Constants.EOS_WORD])
    logger.debug('==> corpus vocabulary size : {}, size in bytes :{} bytes  '.format(vocab.size(), sys.getsizeof(vocab)))

    t_90 = time. time()
    print('time taken in building  Data vocab : {}'.format(t_90 - t_start))


    train_dataset = Dataset(train_dir, vocab, args.num_classes)


    t_106 = time.time()
    print(' Time taken to build Dataset class : {}, len of ltree : {}, rtree len : {}'.format((t_106 - t_start), len(train_dataset.ltrees), len(train_dataset.rtrees)) )

    # number of sentences for each paragraph
    info_dir = os.path.join('data', args.data_name + '_Data','Info_File')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    list_info = pickle.load(open(os.path.join(info_dir, 'info_train.pickle'), "rb" ))
    t_2 = time.time()
    print('\t Time taken to load info train file : {}, length of list info : {}'.format((t_2 - t_start), len(list_info)))

    final_data =  {}
    idx = 0

    count_sent = 0
    for b_id, body in tqdm(enumerate(list_info[:]), total=len(list_info)): #for item in list_info:
        final_data[b_id] = { 'headline' : {} , 'body_list': {}}
        rtree, rsent, label = train_dataset.get_headline(b_id)

        final_data[b_id]['headline'] = { 'rtree' :  rtree , 'rsent' : rsent, 'label' : label }

        cur_body_list = []
        for p_id, para in enumerate(body):
            final_data[b_id]['body_list'][p_id] = []
            cur_para_list = []
            for s_id in range(para):
                # print(b_id, p_id, s_id, count_sent)
                ltree,lsent = train_dataset.get_sentence_body(count_sent)
                final_data[b_id]['body_list'][p_id].append((ltree, lsent))
                #cur_body_list.append((b_id, p_id, s_id, train_dataset[idx]))
                count_sent += 1




    fname = os.path.join(train_dir, 'train_data.pkl')
    t_132 = time.time()
    print(' Length of final train data:', len(final_data))


    t3 = time.time()

    fout = open(fname, 'wb')
    pickle.dump(final_data, fout)
    fout.close()
    build_train_fold(final_data,  data_type= 'train') # divide the train data in 5000 SO that is can run on small RAM cpu



    print('time taken in filling list info : {}'.format(t3 - t_start))


    # Bulding vocab for test dataset
    test_dataset = Dataset(test_dir, vocab, args.num_classes)

    t_106 = time.time()

    print(' Time taken to build Dataset class : {}, len of ltree : {}, rtree len : {}'.format((t_106 - t_start), len(test_dataset.ltrees), len(test_dataset.rtrees)) )

    # number of sentences for each paragraph
    list_info = pickle.load(open(os.path.join( info_dir, 'info_test.pickle'), "rb" ))
    t_2 = time.time()
    print('\t Time taken to load info train file : {}, length of list info : {}'.format((t_2 - t_start), len(list_info)))

    final_data =  {}
    idx = 0

    count_sent = 0

    for b_id, body in tqdm(enumerate(list_info[:]), total = len(list_info)): #for item in list_info:
        final_data[b_id] = { 'headline' : {} , 'body_list': {}}
        rtree, rsent, label = test_dataset.get_headline(b_id)
        final_data[b_id]['headline'] = { 'rtree' :  rtree , 'rsent' : rsent, 'label' : label }

        cur_body_list = []
        for p_id, para in enumerate(body):
            final_data[b_id]['body_list'][p_id] = []
            cur_para_list = []
            for s_id in range(para):
                # print(b_id, p_id, s_id, count_sent)
                ltree,lsent = test_dataset.get_sentence_body(count_sent)
                final_data[b_id]['body_list'][p_id].append((ltree, lsent))
                #cur_body_list.append((b_id, p_id, s_id, train_dataset[idx]))
                count_sent += 1



    t_132 = time.time()
    fname = os.path.join(test_dir,'test_data.pkl')
    print(' Lenght of test data:', len(final_data))

    fout = open(fname, 'wb')
    pickle.dump(final_data, fout)
    fout.close()
    build_train_fold(final_data,  data_type= 'test') # divide the TEST data in 5000 SO that is can run on small RAM cpu


    print(' \t Building train and test data completed in : {}'.format(t_132-t_start))


    # Bulding Devlopment dataset
    dev_dataset = Dataset(dev_dir, vocab, args.num_classes)

    # number of sentences for each paragraph
    list_info = pickle.load(open(os.path.join(info_dir, 'info_dev.pickle'), "rb" ))
    t_2 = time.time()

    final_data =  {}
    idx = 0

    count_sent = 0
    for b_id, body in tqdm(enumerate(list_info[:]), total=len(list_info)): #for item in list_info:
        final_data[b_id] = { 'headline' : {} , 'body_list': {}}
        rtree, rsent, label = test_dataset.get_headline(b_id)
        final_data[b_id]['headline'] = { 'rtree' :  rtree , 'rsent' : rsent, 'label' : label }

        cur_body_list = []
        for p_id, para in enumerate(body):
            final_data[b_id]['body_list'][p_id] = []
            cur_para_list = []
            for s_id in range(para):
                # print(b_id, p_id, s_id, count_sent)
                ltree,lsent = test_dataset.get_sentence_body(count_sent)
                final_data[b_id]['body_list'][p_id].append((ltree, lsent))
                #cur_body_list.append((b_id, p_id, s_id, train_dataset[idx]))
                count_sent += 1



    t_132 = time.time()
    fname = os.path.join(dev_dir,'dev_data.pkl')
    print(' lenght of final data for Dev: ',len(final_data))
    fout = open(fname, 'wb')
    pickle.dump(final_data, fout)
    fout.close()
    build_train_fold(final_data, data_type= 'dev')
    print('\\t Program completed in : {}'.format(t_132-t_start))

    print("program completed with train test and dev")

    emb_file_name = '{}_{}_{}d.pth'.format(args.data_name, args.emb_name, args.input_dim)
    emb_file = os.path.join(args.data, emb_file_name )
    if os.path.isfile(emb_file):
        print(' Embed path : {} exists'.format(emb_file))
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = utils.load_word_vectors(
            os.path.join(args.glove, 'glove.6B.200d'))
        logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.zeros(vocab.size(), glove_emb.size(1), dtype=torch.float, device=device)
        emb.normal_(0, 0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD,
                                    Constants.BOS_WORD, Constants.EOS_WORD]):
            emb[idx].zero_()
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)
    # plug these into embedding matrix inside model
    #model.emb.weight.data.copy_(emb)
    print(' emb shape :', emb.shape)


def build_train_fold(train_data, data_type= 'train'):
    # fname = os.path.join(train_dir, 'train_data.pkl')
    # fin = open(fname , 'rb')
    # train_data = pickle.load(fin)
    #fin.close()
    train_dir =   os.path.join(args.data, 'train/')#'data/sick/train/'
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/') #'data/sick/test/' #
    if data_type == 'train':
        out_dir = train_dir
    elif data_type == 'test':
        out_dir = test_dir
    elif data_type == 'dev':
        # Only do label extraction for development dataset
        out_dir = dev_dir
        # load dev data
        dev_label = label_exteract(train_data)
        # saving dev label
        fname_out = os.path.join(out_dir, 'dev_label.pkl')
        print(len(dev_label))
        print("the number of sample in dev is",len(dev_label))
        fout = open(fname_out, 'wb')
        pickle.dump(dev_label, fout)
        fout.close()
        return

    print("the no of news article pair",len(train_data))

    t_132 = time.time()
    key_list = list(train_data.keys())
    key_list.sort()
    final_data_new = {}
    no_of_subfiles =  int(math.ceil((len(key_list) / 5000)))
    for file_no in range(0, no_of_subfiles):
        final_data_new = {}
        print(" Processing fold : {}".format(file_no))
        for b_id in key_list[5000*file_no:5000*(file_no+1)]:
            final_data_new[b_id] = train_data[b_id]
            if b_id == len(train_data)-1:
                break
        fname_out = os.path.join(out_dir, 'Fold-{}.pkl'.format(file_no))
        print("the number of sample in training  is : {} in fold : {}".format(len(final_data_new), file_no))
        fout = open(fname_out, 'wb')
        pickle.dump(final_data_new, fout)
        fout.close()
    t_finish = time.time()
    print('Time taken : {}'.format(t_finish-t_132))
    ## ----- End of split loop for train_data


    # save train label pkl file
    train_label = label_exteract(train_data)
    fname_out = os.path.join(out_dir, 'train_label.pkl')
    print("the number of sample in training  is",len(train_label))
    fout = open(fname_out, 'wb')
    pickle.dump(train_label, fout)
    fout.close()

def main():
    t1= time.time()
    build_input()
    t2 = time.time()
    print(' Total time taken : {}'.format(t2-t1))
if __name__ =='__main__':
    main()
