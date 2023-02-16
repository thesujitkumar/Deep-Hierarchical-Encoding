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
import time
from tqdm import tqdm
import os
import argparse

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


# In[3]:


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# In[4]:


def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"")
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






def preprocess_data(base_ip_dir, parsed_dir, info_dir, dataset_name, fname, data_type = ''):
    t1 = time.time()


    # to load NELA Data sets
    fname_total = os.path.join(base_ip_dir, fname)
    df = pd.read_csv(fname_total)
    print(' Comns name : ', df.columns)

    parse_dir_total = os.path.join(parsed_dir, data_type)
    if not os.path.exists(parse_dir_total):
        os.makedirs(parse_dir_total)
    fname_out_total = os.path.join(parse_dir_total,  'b.txt')
    df['Headline'].to_csv(fname_out_total, header=None,  index=None)

    fname_out_total = os.path.join(parse_dir_total,'label.txt')
    df['Label'].to_csv(fname_out_total, header=None, index=None)

    tempList = df['Body']
    print(len(tempList))
    print(tempList[0:2])

    print('Total lines :', len(tempList))
    l =  len(tempList)
    finalList = []
    listOflistOfSentences = [] # list to hold list of sentences
    count = 0
    for i, item in tqdm(enumerate(tempList,0), total=len(tempList)):
        # if (count+1) % 2000 ==0:
        #     print('Processed {} line out of {}'.format(count, l))
        paras = item.split('<EOP>')
        temp_list = []
        for p in paras:
            temp_sent=[]
            sents = p.strip().split('<EOS>')
            for s in sents:
                text=preprocess(s.strip())
                if(not text):
                    continue
                temp_sent.append(text)
                finalList.append(text)
            temp_list.append(len(temp_sent))
        listOflistOfSentences.append(temp_list)
        count +=1
    print("the length of final list",len(finalList))
    print(listOflistOfSentences[0:5])


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
    with open(fname_out_total,"wb") as f:
      pickle.dump(listOflistOfSentences,f)


    t2 = time.time()
    print('time taken : {}'.format(t2-t1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/NELA_Data',
                        help='path to dataset')
    parser.add_argument('--data_name', default='NELA',
                        help='Name of dataset')
    parser.add_argument('--input_file', default='dev.csv',
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
