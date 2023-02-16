##  model with simle LSTM for sentence encoding (without tree lstm)

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from . import Constants





class DocLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent, max_num_word):
        super(DocLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.max_num_para = max_num_para
        self.max_num_sent = max_num_sent
        self.max_num_word = max_num_word
        self.mem_dim = mem_dim
        self.in_dim = in_dim
        self.body_LSTM = nn.LSTM(mem_dim, mem_dim, 1)
        self.Para_LSTM = nn.LSTM(mem_dim, mem_dim, 1)
        self.sentence_LSTM = nn.LSTM(in_dim, mem_dim, 1)
        torch.manual_seed(0)

        self.sent_pad =   torch.randn(1, 1, mem_dim)

        self.para_pad = torch.randn(1, 1, mem_dim) #torch.Size([1, 1, 150])
        self.word_pad= torch.randn(1, in_dim)


    def forward(self, body):

        rtree = body['headline']['rtree']
        rsent = body['headline']['rsent']

        rinputs_list=[]

        for word in rsent:
            rinputs_list.append(self.emb(word).view(1, self.in_dim))

        if len(rsent) < self.max_num_word:
            rinputs_list += [ self.word_pad] * (self.max_num_word - len(rinputs_list))

        seq_head_input=torch.cat(rinputs_list[:self.max_num_word],0)


        #out_seq_hed, (h_seq_hed, c_seq_hed)=self.Headline_LSTM(seq_head_input.contiguous().view(12, 1, 200))


        rinputs=self.emb(rsent)
        ab, (rhidden,c_headline) = self.sentence_LSTM(seq_head_input.contiguous().view(self.max_num_word, 1, self.in_dim))

        #c_headline, rhidden = self.childsumtreelstm(rtree, rinputs,h_seq_hed.view(1,150))
        p_hidden_list = []
        body=body['body_list']
        count=0
        for p_id in body:
            count=count+1
            if count > self.max_num_para:   # condition for only two paragrphs
               break
            s_hidden_list = []
            for s_id, sentence in enumerate(body[p_id]):

                ltree,lsent = sentence
                linputs = self.emb(lsent)

                linputs_list=[]
                for word in lsent:
                    linputs_list.append(self.emb(word).view(1, self.in_dim))

                if len(lsent) < self.max_num_word:
                    linputs_list += [ self.word_pad] * (self.max_num_word - len(linputs_list))

                seq_head_input=torch.cat(linputs_list[:self.max_num_word],0)

                lstate,(lhidden,ab) = self.sentence_LSTM(seq_head_input.contiguous().view(self.max_num_word, 1, self.in_dim))
                del linputs_list

                 #modified code above
                s_hidden_list.append(lhidden)
            if len(s_hidden_list) <self.max_num_sent:
                s_hidden_list += [self.sent_pad] * (self.max_num_sent - len(s_hidden_list))
            sentences_encoding = torch.cat(s_hidden_list[:self.max_num_sent], 0)
            out_para, (h_para, c_para)=self.Para_LSTM(sentences_encoding.contiguous().view(self.max_num_sent, 1, self.mem_dim)) #  encoding of kth paragraph in body

            p_hidden_list.append(h_para)
            del s_hidden_list
        p_hidden_list += [ self.para_pad] * (self.max_num_para - len(p_hidden_list))
        paragraph_encoding = torch.cat(p_hidden_list[:self.max_num_para], 0)
        del p_hidden_list

        out_body, (h_body, c_body)=self.body_LSTM(paragraph_encoding)
        del body

        del rtree, rsent
        del rinputs_list, rinputs
        del out_para, c_para, lstate,
        del out_body
        return h_body, rhidden




# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes, feature_dim,  domain_feature):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.domain_feature = domain_feature

        if self.domain_feature : # contact deep feature + domain feature
            self.wh = nn.Linear(((4 * self.mem_dim) + self.feature_dim), self.hidden_dim) # for combined model
        else: # use only deep feature
            self.wh = nn.Linear((4 * self.mem_dim) , self.hidden_dim)  # for only deep feature.
        self.wp = nn.Linear(self.hidden_dim, 2)

    def forward(self, lvec, rvec, feature_vec):
        mult_dist = torch.mul(lvec, rvec) #dot product between body and headline representation
        abs_dist = torch.abs(torch.add(lvec, -rvec))  # absoulte difference between body and headline representation
        vec_dist = torch.cat((mult_dist, abs_dist), 1)  # concatenation of absoulte difference and multiplication
        vec_cat=torch.cat((lvec,rvec),1) # concatenation of body and headline vectors
        entail=torch.cat((vec_dist,vec_cat),1)

        """ Merge the feature vecot befor going to MLP"""
        if self.domain_feature: #for combined feature model
            concat_vec = torch.cat( (entail , torch.FloatTensor(feature_vec).reshape(1,len(feature_vec)) ), dim=1)
            #print(' concst vec shape : ', concat_vec.shape)
            out = torch.sigmoid(self.wh(concat_vec)) # Calling MLP for combined model
        else:
            out = torch.sigmoid(self.wh(entail)) # for model with only deep feature
        out =self.wp(out) # No softmax
        return out




# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, feature_dim, sparsity, freeze, num_classes, \
        max_num_para, max_num_sent, max_num_word, domain_feature):
        super(SimilarityTreeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.doclstm = DocLSTM(vocab_size, in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent, max_num_word)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes, feature_dim, domain_feature)
    def forward(self, body,  feature_vec= None):
        c_body, c_headline  = self.doclstm(body)
        output = self.similarity( c_body.view(1, self.hidden_dim) , c_headline.view(1, self.hidden_dim),  feature_vec)
        return output
