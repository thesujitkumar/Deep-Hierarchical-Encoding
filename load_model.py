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
import pandas as pd


# IMPORT CONSTANTS
from treelstm import Constants
# NEURAL NETWORK MODULES/LAYERS
from treelstm import HE_LSTM, SynHE, InHE
#from treelstm import model_t, model_s, model_new
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

import time
import gc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def label_exteract(data_dic):    # Function to exteract label
    label_list=[]
    for idx in (range(len(data_dic.keys()))):  # exteract labels from dictionary for train data
        label = data_dic[idx]['headline']['label']
        label_list.append(label)
    target_val=torch.LongTensor(label_list)
    del label_list
    return target_val

# MAIN BLOCK
def main():
    t_start = time.time()
    global args
    args = parse_args()
    log_dir = os.path.join(args.save, args.model_name)
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
    #dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/') #'data/sick/test/' #
    print(train_dir, test_dir )

    # write unique words from all token files
    vocab_file_name = '{}_{}_{}d.vocab'.format(args.data_name, args.emb_name, args.input_dim)
    vocab_file = os.path.join(args.data, vocab_file_name ) # 'FNC_Bin_Data_glove_200d.vocab') #
    #sick_vocab_file = os.path.join(args.data, 'IOST_Data_glove_200d.vocab') #
    if not os.path.isfile(vocab_file):
        token_files_b = [os.path.join(split, 'b.toks') for split in [train_dir,test_dir]]
        token_files_a = [os.path.join(split, 'a.toks') for split in [train_dir,test_dir]]
        token_files = token_files_a + token_files_b
        sick_vocab_file = os.path.join(args.data, 'FNC_Data.vocab')
        utils.build_vocab(token_files, sick_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=vocab_file,
                  data=[Constants.PAD_WORD, Constants.UNK_WORD,
                        Constants.BOS_WORD, Constants.EOS_WORD])
    logger.debug('==> corpus vocabulary size : {}, size in bytes :{} bytes  '.format(vocab.size(), sys.getsizeof(vocab)))

    t_90 = time. time()
    print('time taken in building FNC  Data vocab : {}'.format(t_90 - t_start))

    test_file = os.path.join(args.data, 'sick_test.pth')


    # definining model


    feature_name_train = os.path.join(train_dir, args.feature_fname)
    print('feature_name_train :', feature_name_train)
    df_feature = pd.read_excel(feature_name_train, engine='openpyxl')
    feature_dim = df_feature.shape[1]
    print(' \t\t Len of feature : {}'.format(feature_dim))
    del df_feature

    args.freeze_embed= True
    if args.model_name == 'SynHE':
	    model = SynHE.SimilarityTreeLSTM(
		vocab.size(),
		args.input_dim,
		args.mem_dim,
		args.hidden_dim,
        feature_dim,
        args.sparse,
		args.num_classes,
		args.freeze_embed,
        args.max_num_para,
        args.max_num_sent,
        args.domain_feature)
    elif args.model_name == 'HE_LSTM':
	    model = HE_LSTM.SimilarityTreeLSTM(
		vocab.size(),
		args.input_dim,
		args.mem_dim,
		args.hidden_dim,
        feature_dim,
        args.sparse,
		args.num_classes,
		args.freeze_embed,
        args.max_num_para,
        args.max_num_sent,
        args.max_num_word,
        args.domain_feature)
    elif args.model_name == 'InHE':
	    model = InHE.SimilarityTreeLSTM(
		vocab.size(),
		args.input_dim,
		args.mem_dim,
		args.hidden_dim,
        feature_dim,
        args.sparse,
		args.num_classes,
		args.freeze_embed,
        args.max_num_para,
        args.max_num_sent,
        args.domain_feature)

    print(' Total number of parameter : {}'.format(count_parameters(model)))
    print('Number of parameters in DOCLstm:{}'.format(count_parameters(model.doclstm)))
   # print('Number of parameters in Tree-lstm :{}'.format(count_parameters(model.doclstm.childsumtreelstm)))
    print( 'Size of model : {} byte '.format(sys.getsizeof(model)))

    criterion =  nn.CrossEntropyLoss() #nn.KLDivLoss() #   nn.CrossEntropyLoss()

    #NELA_Data_GLOV_EMBED_200d.pth
    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file_name = '{}_{}_{}d.pth'.format(args.data_name, args.emb_name, args.input_dim)
    emb_file = os.path.join(args.data, emb_file_name )

    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = utils.load_word_vectors(
            os.path.join(args.glove, 'glove.840B.300d'))
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
    model.doclstm.emb.weight.data.copy_(emb)
    del emb
    gc.collect()

    # delete emb
    model.to(device), criterion.to(device)
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                         model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    # exteract labels for training data
    trainer = Trainer(args, model, criterion, optimizer, device, args.batchsize, args.num_classes, args.file_len, args.domain_feature)

    checkpoint_path  = '{}.pt'.format(os.path.join(log_dir, args.expname))
    # checkpoint_path  = '{}.pt'.format(os.path.join('checkpoints_back', args.expname))
    checkpoint= torch.load(checkpoint_path)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    trainer.model.eval()
    trainer.optimizer.load_state_dict(checkpoint['optim_state_dict'])
    #trainer.optimizer.eval()
    test_loss, test_pred = trainer.test(2)
    #test_pred = trainer.test(2)
    print("test predictions hape",test_pred.size())
    fname = os.path.join(test_dir, 'test_label.pkl')
    fin = open(fname , 'rb')
    ground_truth = pickle.load(fin)
    print("the length of total label:",len(ground_truth))
    fin.close()
    test_accuracy = metrics.accuracy(test_pred, ground_truth)
    test_fmeasure = metrics.fmeasure(test_pred, ground_truth)
    logger.info(' test \tLoss: {}\tAccuracy: {}\tF1-score: {}'.format(
                 test_loss, test_accuracy , test_fmeasure))
    df=pd.DataFrame(test_pred)
    df.to_csv("Model_T_no_of_para_2.csv",index=False)
    #del ground_truth


if __name__ =='__main__':
    main()
