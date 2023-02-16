##  model with tree LSTM  with attention for sentence encoding
import torch.nn as nn
import torch
#import torch.nn.functional as F
import gc
from . import Constants



# module for childsumtreelstm forawrd without recursion





### test save



class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.W = nn.Linear(self.mem_dim,self.mem_dim)
        self.U = nn.Linear(self.mem_dim,self.mem_dim)

        self.small_w = nn.Linear(self.mem_dim,1)
        self.Wa = nn.Linear(self.mem_dim,self.mem_dim,bias=True)


    def node_forward(self, inputs, tree, child_c, child_h, seq_att,indicator):


        if(indicator == 0 ):

            child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        else:

            M = torch.tanh(self.W(child_h)+self.U(seq_att))
            prod = self.small_w(M)
            attention_weights = torch.softmax(prod,dim=1)
            child_h_sum = torch.matmul(torch.transpose(attention_weights,0,1),child_h)
            child_h_sum = self.Wa(child_h_sum)




        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, torch.tanh(c))

        return c, h


    def forward(self, tree, inputs,seq_att,indicator):
        # Make a node list in increasing order of depth using BFS
        node_list = [tree]
        start = 0
        end = 1
        while(start!=end):
            #print(start, end, len(node_list))
            for i in range(start,end):
                #print(i)
                if node_list[i].num_children >0:
                    children_list = node_list[i].children
                    node_list += children_list
            start = end
            end = len(node_list)
        # Initalizing h & c dictionary for each node id
        hc_dict = {}
        # processing each node in decreasing order of depth
        for node in node_list[-1::-1]:
            if node.num_children == 0:
                child_c_prev =torch.zeros(1, self.mem_dim )            #inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
                child_h_prev = torch.zeros(1, self.mem_dim )           #inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
                child_c, child_h = self.node_forward(inputs[node.idx], node, child_c_prev, child_h_prev, seq_att,indicator)
                hc_dict[node.idx] = { 'h': child_h, 'c' : child_c}
            else:
                child_c_list = []
                child_h_list = []
                for child in node.children:
                    child_c_list.append(hc_dict[child.idx]['c'])
                    child_h_list.append(hc_dict[child.idx]['h'])
                child_c_prev = torch.cat(child_h_list, dim=0)
                child_h_prev = torch.cat(child_c_list, dim=0)
                del child_c_list,child_h_list
                child_c, child_h = self.node_forward(inputs[node.idx], node, child_c_prev, child_h_prev, seq_att,indicator)
                hc_dict[node.idx] = { 'h': child_h, 'c' : child_c}
        root_h = hc_dict[tree.idx]['h'].clone()
        root_c = hc_dict[tree.idx]['c'].clone()
        del tree
        del node_list
        del hc_dict
        #gc.collect()
        return root_h, root_c



class DocLSTM(nn.Module):
    def __init__(self,  vocab_size, in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent):
        super(DocLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.max_num_para = max_num_para
        self.max_num_sent = max_num_sent
        self.mem_dim = mem_dim
        self.body_LSTM = nn.LSTM(mem_dim, mem_dim, 1)
        self.Para_LSTM = nn.LSTM(mem_dim, mem_dim, 1)
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)

        torch.manual_seed(0)
        self.sent_pad =   torch.randn(1, mem_dim)
        self.para_pad = torch.randn(1, 1, mem_dim)
        self.word_pad= torch.randn(1, in_dim)


    def forward(self, body):

        rtree = body['headline']['rtree']
        rsent = body['headline']['rsent']
        s=self.word_pad
        rinputs=self.emb(rsent)
        c_headline, rhidden = self.childsumtreelstm(rtree, rinputs,s,0)
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
                lstate, lhidden = self.childsumtreelstm(ltree, linputs,rhidden,1) # encoding of sentecnes at sentence level
                s_hidden_list.append(lhidden)

            if len(s_hidden_list) <self.max_num_sent:
                s_hidden_list += [self.sent_pad] * (self.max_num_sent - len(s_hidden_list))

            sentences_encoding = torch.cat(s_hidden_list[: self.max_num_sent], 0)
            out_para, (h_para, c_para)=self.Para_LSTM(sentences_encoding.contiguous().view(self.max_num_sent, 1, self.mem_dim)) #  encoding of kth paragraph in body

            p_hidden_list.append(h_para)
            del s_hidden_list
        p_hidden_list += [ self.para_pad] * (self.max_num_para  - len(p_hidden_list))
        paragraph_encoding = torch.cat(p_hidden_list[:self.max_num_para], 0)
        del p_hidden_list

        out_body, (h_body, c_body)=self.body_LSTM(paragraph_encoding) # ccall body_LSTM function to encode the several paragraph to final represnetation

        del body
        del rtree, rsent
        del out_para, c_para
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
        max_num_para, max_num_sent, domain_feature):
        # feature_dim is length of domain dependent vector
        super(SimilarityTreeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.doclstm = DocLSTM(vocab_size, in_dim, mem_dim, sparsity, freeze, max_num_para, max_num_sent)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes, feature_dim, domain_feature)
    def forward(self, body, feature_vec = None):
        c_body, c_headline  = self.doclstm(body)
        #print(' hidden dim : ', self.hidden_dim)
        output = self.similarity( c_body.view(1, self.hidden_dim) , c_headline,  feature_vec)
        return output
