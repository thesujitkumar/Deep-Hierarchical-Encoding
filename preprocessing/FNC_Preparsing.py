#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import pickle
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
import os
from tqdm import tqdm
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
import argparse

# In[3]:


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# In[4]:

import re
def striphtml(data):
    p = re.compile(r'<(.*)>.*?|<(.*) />')
    return p.sub('', data)

def preprocess_paragraph(sentence):
    sentence=str(sentence)
    sentence = striphtml(sentence)
    #sentence = sentence.lower()
    #sentence =  striphtml(sentence)
    #sentence=sentence.replace('{html}',"")
    #cleanr = re.compile('<.*?>')
    #cleantext = re.sub(cleanr, '', sentence)
    #rem_url=re.sub(r'http\S+', '',cleantext)
    #rem_num = re.sub('[0-9]+', '', rem_url)
    #tokenizer = RegexpTokenizer(r'\w+')
    #tokens = tokenizer.tokenize(rem_num)
    #filtered_words = [w for w in tokens if len(w) > 1 ]
    #stem_words=[stemmer.stem(w) for w in filtered_words]
    #lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]

    #return " ".join(filtered_words)
    return sentence

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    #sentence =  striphtml(sentence)
    #sentence=sentence.replace('{html}',"")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 1 ]
    #stem_words=[stemmer.stem(w) for w in filtered_words]
    #lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]

    return " ".join(filtered_words)

# In[5]:

def preprocess_data(base_ip_dir, parsed_dir, info_dir, dataset_name, fname, data_type = ''):
    fname_total = os.path.join(base_ip_dir, fname)
    # to load FNC Data sets
    df = pd.read_csv(fname_total)
    #df1 = pd.read_csv('train_stances.csv')
    print(df.columns)
    # save to b.txt

    parse_dir_total = os.path.join(parsed_dir, data_type)
    if not os.path.exists(parse_dir_total):
        os.makedirs(parse_dir_total)
    fname_out_total = os.path.join(parse_dir_total,  'b.txt')
    df['Headline'].to_csv(fname_out_total, header=None,  index=None)

    fname_out_total = os.path.join(parse_dir_total,'label.txt')
    df['Stance'].to_csv(fname_out_total, header=None, index=None)

    tempList = df['articleBody']
    print(len(tempList))
    print(tempList[0:2])


    import time
    count1=0
    t1 = time.time()
    print('Total lines :', len(tempList))
    l =  len(tempList)
    finalList = []
    listOflistOfSentences = [] # list to hold list of sentences
    count = 0
    zero_s_count = 0
    total_s_count = 0
    zero_s_list = []
    for i, item in tqdm(enumerate(tempList,0), total=len(tempList)):
        # if (count+1) % 5000 ==0:
        #     print('Processed {} line out of {}'.format(count, l))
        paras = item.split('\n\n')
        temp_list = []
        for p in paras:
            temp_sent=[]
            p = preprocess_paragraph(p)
            sents = p.strip().split('.')
            for s in sents:
                text=preprocess(s.strip())
                if(not text):
                    continue
                temp_sent.append(text)
                finalList.append(text)
                total_s_count +=1
            if(len(temp_sent)==0):
               zero_s_list.append(sents)
               #print(i)
               count1=count1 +1
               zero_s_count +=1
            else:
                temp_list.append(len(temp_sent))
        listOflistOfSentences.append(temp_list)
        count +=1
    #print("the length of final list",len(finalList))
    #print(listOflistOfSentences[0:5])
    # print("number of empty", count1)
    #
    # print (' zero_s_count : {}, total_s_count : {}, per : {}'.format(zero_s_count, total_s_count, round((zero_s_count*100 /total_s_count),2)))
    # print('zero_s_list:', zero_s_list)

    converted_list = []

    #print(finalList)
    for element in finalList:
        #print(element)
        converted_list.append(element.strip())


    df_temp = pd.DataFrame(converted_list)
    fname_out_total = os.path.join(parse_dir_total,  'b.txt')
    df_temp.to_csv(fname_out_total,index=False,header=None) # a.txt



    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    fname_out = "info_{}.pickle".format(data_type)
    fname_out_total = os.path.join(info_dir, fname_out)
    with open(fname_out_total, "wb") as f:
        pickle.dump(listOflistOfSentences,f)

    t2 = time.time()
    print('time taken : {}'.format(t2-t1))
            # In[26]:


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/FNC_Data',
                        help='path to dataset')
    parser.add_argument('--data_name', default='FNC',
                        help='Name of dataset')
    parser.add_argument('--input_file', default='FNC_Bin_Dev.csv',
                            help='Name of input  csv file')
    parser.add_argument('--data_type', default='dev',
                                help='Type of data file : test/train/dev')
    args = parser.parse_args()

    #dataset_name = 'FNC'
    #fname = 'FNC_Bin_Dev.csv'
    base_ip_dir = os.path.join(args.data, 'Raw_Data')
    parsed_dir = os.path.join(args.data, 'Parsed_Data')
    info_dir = os.path.join(args.data, 'Info_File')
    #data_type = 'dev' # train/ test / dev

    preprocess_data(base_ip_dir, parsed_dir, info_dir, dataset_name=args.data_name, fname= args.input_file, data_type = args.data_type)
    pass

if __name__ == '__main__':
    main()
