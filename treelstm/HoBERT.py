##  model with simle LSTM for sentence encoding (without tree lstm)

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from . import Constants
from transformers import BertTokenizer, BertModel





class DocLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent, max_num_word):
        super(DocLSTM, self).__init__()

        self.max_num_para = max_num_para
        self.max_num_sent = max_num_sent
        # self.max_num_word = max_num_word
        self.mem_dim = mem_dim
        self.in_dim = in_dim
        self.body_LSTM = nn.LSTM(2*mem_dim, mem_dim, 1, bidirectional=True)
        self.News_LSTM = nn.LSTM(in_dim, mem_dim, 1, bidirectional=True )
        self.sentence_transform = nn.Linear(in_dim, 2*mem_dim)
        torch.manual_seed(0)
        self.sent_pad =   torch.randn(1, in_dim)
        self.para_pad = torch.randn(1, 2*mem_dim) #torch.Size([1, 1, 150])
        # self.word_pad=

    def forward(self, body):

        rsent = body['headline']['rsent']


        sent_encoded = self.sentence_transform(rsent)

        body=body['body_list']
        count=0
        para_enc_list=[]
        for p_id in body:
            sent_encoded_List= []
            for s_id, sentence in enumerate(body[p_id]):
                
                lsent = sentence
                sent_encoded_List.append(lsent)
                "encdoing of each segmente on top of encdoings of each segments using BERT"

            sent_encoded_List += [ self.sent_pad] * (self.max_num_sent - len(self.sent_pad))
            news_article_inp = torch.cat(sent_encoded_List[:self.max_num_sent], 0)
            del sent_encoded_List

            out_News_article, (h_News_article, c_News_article)=self.News_LSTM(news_article_inp.contiguous().view(self.max_num_sent, 1, self.in_dim))
            h_News_article_2d = h_News_article.view(2,self.mem_dim)
            h_left= h_News_article_2d[0]
            h_right= h_News_article_2d[1]
            Bi_Para_h= torch.cat((h_left,h_right), 0)
            para_enc_list.append(Bi_Para_h.view(1, 2*self.mem_dim))


        para_enc_list += [ self.para_pad] * (self.max_num_para - len(self.para_pad))
        body_LSTM_input = torch.cat(para_enc_list[:self.max_num_para], 0)

        out_Body_article, (h_Body_article, c_Body_article)=self.body_LSTM(body_LSTM_input.contiguous().view(self.max_num_para, 1, 2*self.mem_dim))
        h_Body_article_2d = h_Body_article.view(2,self.mem_dim)
        h_left= h_Body_article_2d[0]
        h_right= h_Body_article_2d[1]
        Bi_Body_h= torch.cat((h_left,h_right), 0)



        del out_News_article, c_News_article, body, rsent, lsent, para_enc_list
        return sent_encoded, Bi_Body_h




# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes



        self.wh = nn.Linear((8 * self.mem_dim) , self.hidden_dim)  # for only deep feature.
        self.wp = nn.Linear(self.mem_dim, 2)

    def forward(self,  lvec, rvec):
        mult_dist = torch.mul(lvec, rvec) #dot product between body and headline representation
        abs_dist = torch.abs(torch.add(lvec, -rvec))  # absoulte difference between body and headline representation
        vec_dist = torch.cat((mult_dist, abs_dist), 1)  # concatenation of absoulte difference and multiplication
        vec_cat=torch.cat((lvec,rvec),1) # concatenation of body and headline vectors
        entail=torch.cat((vec_dist,vec_cat),1)



        out = torch.sigmoid(self.wh(entail)) # for model with only deep feature
        out =self.wp(out) # No softmax
        #print(out)
        return out




# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self,  in_dim, mem_dim, hidden_dim, sparsity, freeze, num_classes, \
        max_num_para, max_num_sent, max_num_word):
        super(SimilarityTreeLSTM, self).__init__()
        self.mem_dim = mem_dim
        self.doclstm = DocLSTM( in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent, max_num_word)
        self.similarity = Similarity( mem_dim, hidden_dim, num_classes)
    def forward(self, body,  feature_vec = None):
        head, body  = self.doclstm(body)
        output = self.similarity(head.view(1, 2*self.mem_dim), body.view(1, 2*self.mem_dim) )
        return output
