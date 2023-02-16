import os
import torch
import pickle
from config import parse_args
import pandas as pd
import time
import gc
from tqdm import tqdm
import math


"Import IF bert"
from transformers import BertTokenizer, BertModel
# from sentence_transformers import SentenceTransformer


"When Bert"
tokenizer= BertTokenizer.from_pretrained('bert-base-uncased')
Bert_encoder= BertModel.from_pretrained("bert-base-uncased")
for param in Bert_encoder.parameters():
    param.requires_grad = False #= True

# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
global args
args = parse_args()

train_dir =   os.path.join(args.data, 'Train/')#'data/sick/train/'
dev_dir = os.path.join(args.data, 'Dev/')
test_dir = os.path.join(args.data, 'Test/')

print("train directory",train_dir)
print("train directory",test_dir)
print("train directory",dev_dir)




def label_exteract(data_dic):    # Function to exteract label
    label_list=[data_dic[idx]['headline']['label'] for idx in tqdm(data_dic, total = len(data_dic))]
    target_val=torch.LongTensor(label_list)
    del label_list
    return target_val

def build_train_fold(train_data, data_type= 'train'):
    # fname = os.path.join(train_dir, 'train_data.pkl')
    # fin = open(fname , 'rb')
    # train_data = pickle.load(fin)
    #fin.close()
    train_dir =   os.path.join(args.data, 'Train/')#'data/sick/train/'
    dev_dir = os.path.join(args.data, 'Dev/')
    test_dir = os.path.join(args.data, 'Test/') #'data/sick/test/' #
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


    ## ----- End of split loop for train_data


    # save train label pkl file
    if data_type == 'train':
        train_label = label_exteract(train_data)
        fname_out = os.path.join(out_dir, 'train_label.pkl')
        print("the number of sample in training  is",len(train_label))
        fout = open(fname_out, 'wb')
        pickle.dump(train_label, fout)
        fout.close()
    elif data_type == 'test':
        out_dir = test_dir
        train_label = label_exteract(train_data)
        fname_out = os.path.join(out_dir, 'test_label.pkl')
        print("the number of sample in training  is",len(train_label))
        fout = open(fname_out, 'wb')
        pickle.dump(train_label, fout)
        fout.close()


def build_input():
    global args
    args = parse_args()

    train_dir =   os.path.join(args.data, 'Train/')#'data/sick/train/'
    dev_dir = os.path.join(args.data, 'Dev/')
    test_dir = os.path.join(args.data, 'Test/') #'data/sick/test/' #
    print(train_dir, test_dir,dev_dir )

    info_dir = os.path.join('data', args.data_name + '_Data','Info_File')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    list_info = pickle.load(open(os.path.join(info_dir, 'info_train.pickle'), "rb" ))
    df_head=pd.read_csv(os.path.join(train_dir, 'b.txt'))
    df_body=pd.read_csv(os.path.join(train_dir, 'a.txt'))
    df_label=pd.read_csv(os.path.join(train_dir, 'label.txt'))

    final_data =  {}
    idx = 0
    head=[]
    lsent_body=[]
    count_sent =0# 750#0
    for b_id, body in tqdm(enumerate(list_info[:]), total=len(list_info)):
        final_data[b_id] = { 'headline' : {} , 'body_list': {}}
        head= df_head['headline'][b_id]
        label= df_label['label'][b_id]

        "Get Bert Encoding Of Headline"
        print(len(str(head).split(' ')))
        if len(str(head).split(' ')) > 50:
            print("guess")
            head=' '.join(str(head).split(' ') [0: 50])
            print("In IF conditions Headline",len(str(head).split(' ')))
        print("after IF",len(str(head).split(' ')))
        head_tokens= tokenizer(str(head), return_tensors='pt')
        with torch.no_grad():
            head_encoding= Bert_encoder(**head_tokens)
            head_enc=head_encoding[1]

        "Sentence Transformer"
        #head_enc= model.encode(str(head))


        final_data[b_id]['headline'] = {'rsent' : head_enc, 'label' : label }
        cur_body_list = []
        for p_id, para in enumerate(body):
            final_data[b_id]['body_list'][p_id] = []
            cur_para_list = []
            for s_id in range(para):
                lsent= df_body['body'][count_sent]

                print(len(str(lsent).split(' ')))
                if len(str(lsent).split(' ')) > 50:
                    print("guess")
                    lsent=' '.join(str(lsent).split(' ') [0: 50])
                    print("In IF conditions",len(str(lsent).split(' ')))
                print("after IF",len(str(lsent).split(' ')))

                "If BERT"
                sent_tokens= tokenizer(str(lsent), return_tensors='pt')
                with torch.no_grad():
                     sent_encoding= Bert_encoder(**sent_tokens)
                     sent_enc= sent_encoding[1]

                "If Sentence Transformer"
                #sent_enc= model.encode(str(lsent))



                final_data[b_id]['body_list'][p_id].append((sent_enc))
                count_sent +=1

    fname = os.path.join(train_dir, 'train_data.pkl')
    print(' Length of final train data:', len(final_data))
    fout = open(fname, 'wb')
    pickle.dump(final_data, fout)
    fout.close()

    build_train_fold(final_data,  data_type= 'train')

    "Build Test Data Dictionary"
    info_dir = os.path.join('data', args.data_name + '_Data','Info_File')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    list_info = pickle.load(open(os.path.join(info_dir, 'info_test.pickle'), "rb" ))
    df_head=pd.read_csv(os.path.join(test_dir, 'b.txt'))
    df_body=pd.read_csv(os.path.join(test_dir, 'a.txt'))
    df_label=pd.read_csv(os.path.join(test_dir, 'label.txt'))

    final_data =  {}
    idx = 0
    head=[]
    lsent_body=[]
    count_sent = 0
    for b_id, body in tqdm(enumerate(list_info[:]), total=len(list_info)):
        final_data[b_id] = { 'headline' : {} , 'body_list': {}}
        head= df_head['headline'][b_id]
        label= df_label['label'][b_id]

        print(len(str(head).split(' ')))
        if len(str(head).split(' ')) > 50:
            print("guess")
            head=' '.join(str(head).split(' ') [0: 50])
            print("In IF conditions Headline",len(str(head).split(' ')))
        print("after IF",len(str(head).split(' ')))


        "Get Bert Encoding Of Headline"
        head_tokens= tokenizer(str(head), return_tensors='pt')
        with torch.no_grad():
            head_encoding= Bert_encoder(**head_tokens)
            head_enc=head_encoding[1]

        "Sentence Transformer"
        #head_enc= model.encode(str(head))



        final_data[b_id]['headline'] = {'rsent' : head_enc, 'label' : label }
        cur_body_list = []
        for p_id, para in enumerate(body):
            final_data[b_id]['body_list'][p_id] = []
            cur_para_list = []
            for s_id in range(para):
                lsent= df_body['body'][count_sent]
                print(len(str(lsent).split(' ')))
                if len(str(lsent).split(' ')) > 50:
                    print("guess")
                    lsent=' '.join(str(lsent).split(' ') [0: 50])
                    print("In IF conditions",len(str(lsent).split(' ')))
                print("after IF",len(str(lsent).split(' ')))
                "If BERT"
                sent_tokens= tokenizer(str(lsent), return_tensors='pt')
                with torch.no_grad():
                    sent_encoding= Bert_encoder(**sent_tokens)
                    sent_enc= sent_encoding[1]

                "If Sentence Transformer"
                #sent_enc= model.encode(str(lsent))


                final_data[b_id]['body_list'][p_id].append((sent_enc))
                count_sent +=1

    fname = os.path.join(test_dir, 'test_data.pkl')
    print(' Length of final test data:', len(final_data))
    fout = open(fname, 'wb')
    pickle.dump(final_data, fout)
    fout.close()

    build_train_fold(final_data,  data_type= 'test')

    "Build Test Data Dictionary"
    info_dir = os.path.join('data', args.data_name + '_Data','Info_File')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    list_info = pickle.load(open(os.path.join(info_dir, 'info_dev.pickle'), "rb" ))
    df_head=pd.read_csv(os.path.join(dev_dir, 'b.txt'))
    df_body=pd.read_csv(os.path.join(dev_dir, 'a.txt'))
    df_label=pd.read_csv(os.path.join(dev_dir, 'label.txt'))

    final_data =  {}
    idx = 0
    head=[]
    lsent_body=[]
    count_sent = 0
    for b_id, body in tqdm(enumerate(list_info[:]), total=len(list_info)):
        final_data[b_id] = { 'headline' : {} , 'body_list': {}}
        head= df_head['headline'][b_id]
        label= df_label['label'][b_id]

        "Get Bert Encoding Of Headline"
        print(len(str(head).split(' ')))
        if len(str(head).split(' ')) > 50:
            print("guess")
            head=' '.join(str(head).split(' ') [0: 50])
            print("In IF conditions Headline",len(str(head).split(' ')))
        print("after IF",len(str(head).split(' ')))
        head_tokens= tokenizer(str(head), return_tensors='pt')
        with torch.no_grad():
            head_encoding= Bert_encoder(**head_tokens)
            head_enc=head_encoding[1]

        "Sentence Transformer"
        #head_enc= model.encode(str(head))



        final_data[b_id]['headline'] = {'rsent' : head_enc, 'label' : label }
        cur_body_list = []
        for p_id, para in enumerate(body):
            final_data[b_id]['body_list'][p_id] = []
            cur_para_list = []
            for s_id in range(para):
                lsent= df_body['body'][count_sent]
                "If BERT"
                print(len(str(lsent).split(' ')))
                if len(str(lsent).split(' ')) > 50:
                    print("guess")
                    lsent=' '.join(str(lsent).split(' ') [0: 50])
                    print("In IF conditions",len(str(lsent).split(' ')))
                print("after IF",len(str(lsent).split(' ')))
                sent_tokens= tokenizer(str(lsent), return_tensors='pt')
                with torch.no_grad():
                    sent_encoding= Bert_encoder(**sent_tokens)
                    sent_enc= sent_encoding[1]

                "If Sentence Transformer"
                #sent_enc= model.encode(str(lsent))

                final_data[b_id]['body_list'][p_id].append((sent_enc))
                count_sent +=1

    fname = os.path.join(dev_dir, 'dev_data.pkl')
    print(' Length of final development data:', len(final_data))
    fout = open(fname, 'wb')
    pickle.dump(final_data, fout)
    fout.close()

    build_train_fold(final_data,  data_type= 'dev')











def main():
    t1= time.time()
    build_input()
    t2 = time.time()
    print(' Total time taken : {}'.format(t2-t1))
if __name__ =='__main__':
    main()
