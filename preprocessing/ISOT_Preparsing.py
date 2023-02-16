
import pandas as pd

import os
import pickle
import spacy
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
from tqdm import tqdm

from tqdm import tqdm
import os
import argparse
import time

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()




#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('punkt')


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
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    # stem_words=[stemmer.stem(w) for w in filtered_words]
    # lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)



def preprocess_data(base_ip_dir, parsed_dir, info_dir, dataset_name, fname, data_type = ''):

    # to load ISOT Data sets
    fname_total = os.path.join(base_ip_dir, fname)
    df = pd.read_csv(fname_total)
    print(' Comns name : ', df.columns)

    parse_dir_total = os.path.join(parsed_dir, data_type)
    if not os.path.exists(parse_dir_total):
        os.makedirs(parse_dir_total)



    #exit(1)
    #nlp = spacy.load("en_core_web_sm")


    # nlp = spacy.load("en_core_web_sm", exclude=["parser"])
    # nlp.enable_pipe("senter")
    #nlp.add_pipe("sentencizer")
    #import spacy
    nlp = spacy.load("en_core_web_sm", exclude=["parser"])
    config = {"punct_chars": ["."]}
    sentencizer = nlp.add_pipe("sentencizer", config=config)

    body_list = df['articleBody']
    # delte row with zero paragraph in the body
    refine_sent=[]
    count_list=[]
    info_list = []
    del_row_index = []
    listOflistOfSentences = []
    for i,b in tqdm(enumerate(body_list[:]), total = len(body_list)):
        body_count_list = []
        no_sent=[]
        doc = nlp(b)
        #print(' Original Sent: \t' , i, b)
        #print('\tParsed Sent')
        paragraph_sent_count = 0
        no_paragraph = 0
        sent_count = 0
        total_sent_count = sum(1 for x in doc.sents)
        for j,sent in enumerate(doc.sents):
            if len(sent.text.split()) >1 : # to ignore empty sentence
                # print(i, j , sent.text)
                text=preprocess(sent.text)
                refine_sent.append(text)
                no_sent.append(text)
                sent_count +=1
                paragraph_sent_count += 1
                #print("lenght of list",len(no_sent))

            if ((sent_count) % 5 ==0)  or (j+1 == total_sent_count): # setting paragraph length to 5
                if paragraph_sent_count > 0:
                    body_count_list.append(paragraph_sent_count)
                    no_paragraph +=1
                    #print(i, j , sent_count, 'End of paragraph', no_paragraph, paragraph_sent_count, total_sent_count)
                paragraph_sent_count = 0

        count_list.append(len(no_sent))
        if len(body_count_list) ==0:
            del_row_index.append(i)
        else:
            listOflistOfSentences.append(body_count_list)
            #print(body_count_list)
            #print(i, b)
    print(' Row to be delested:', del_row_index)

    #print(listOflistOfSentences)

    print(df.shape)
    #del_row_index = [207, 264, 272, 370, 499, 569, 656, 672, 703, 723, 869, 899, 911, 960, 1068, 1080, 1081, 1104, 1143, 1217, 1388, 1607, 1610, 1612, #1649, 1877, 1930, 1990, 2085, 2099, 2102, 2173, 2276, 2277, 2334, 2388, 2390, 2396, 2460, 2578, 2667, 2781, 2892, 2940, 3042, 3093, 3283, 3418, #3429, 3443, 3528]
    df_new  = df.drop(df.index[del_row_index ])
    df_new.reset_index()
    print(df_new.shape)


    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    fname_out = "info_{}.pickle".format(data_type)
    fname_out_total = os.path.join(info_dir, fname_out)
    with open(fname_out_total, "wb") as f:
        pickle.dump(listOflistOfSentences,f)

    # saving Headlines
    fname_out_total = os.path.join(parse_dir_total,  'b.txt')
    df_new['Headline'].to_csv(fname_out_total, header=None,  index=None)



    # saving labels
    fname_out_total = os.path.join(parse_dir_total,'label.txt')
    df_new['Stance'].to_csv(fname_out_total, header=None, index=None)

    # Saving article body
    fname_out_total = os.path.join(parse_dir_total,  'a.txt')
    df_new['articleBody'].to_csv(fname_out_total, header=None,  index=None)



    # fname_total = os.path.join(base_dir, 'IOST_train_ver-2.csv')
    # df_new.to_csv(fname_total, index=False)


def main():
    t1 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/IOST_Data',
                        help='path to dataset')
    parser.add_argument('--data_name', default='IOST',
                        help='Name of dataset')
    parser.add_argument('--input_file', default='IOST_dev_ver-2.csv',
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

    t2 = time.time()
    print(' Time taken : {} for processing dataset {}, dataset type : {}'.format((t2-t1), args.data_name, args.data_type ) )

if __name__ == '__main__':
    main()
