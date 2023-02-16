from __future__ import division
from __future__ import print_function
from tqdm import tqdm

import argparse
import os
import glob
import pandas as pd
import sys
import time
#from config import parse_args
global args

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def dependency_parse(filepath, cp='', tokenize=True):
    print('\t In Dependency parsing function , filepath is :' + filepath)
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.parents')
    relpath = os.path.join(dirpath, filepre + '.rels')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s'
           % (cp, tokpath, parentpath, relpath, tokenize_flag, filepath))
    print('cmd :', cmd)
    os.system(cmd)


def constituency_parse(filepath, cp='', tokenize=True):
    dirpath = os.path.dirname(filepath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    tokpath = os.path.join(dirpath, filepre + '.toks')
    parentpath = os.path.join(dirpath, filepre + '.cparents')
    tokenize_flag = '-tokenize - ' if tokenize else ''
    cmd = ('java -cp %s ConstituencyParse -tokpath %s -parentpath %s %s < %s'
           % (cp, tokpath, parentpath, tokenize_flag, filepath))
    print('cmd :', cmd)
    os.system(cmd)


def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        print( 'file path : ' , filepath)
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')


def split(filepath, dst_dir):
    data = pd.read_csv(filepath)
    with open(filepath) as datafile, \
            open(os.path.join(dst_dir, 'a.txt'), 'w') as afile, \
            open(os.path.join(dst_dir, 'b.txt'), 'w') as bfile,  \
            open(os.path.join(dst_dir, 'id.txt'), 'w') as idfile, \
            open(os.path.join(dst_dir, 'label.txt'), 'w') as labfile:
        for idx in range(2000):
            i = data.iloc[idx]['Body ID']
            a = data.iloc[idx]['Sentence A']
            b = data.iloc[idx]['Sentence B']
            label = data.iloc[idx]['label']
            idfile.write(str(i))
            idfile.write('\n')
            afile.write(str(a))
            afile.write('\n')
            bfile.write(str(b))
            bfile.write('\n')
            labfile.write(str(label))
            labfile.write('\n')

def parse(dirpath, cp=''):
    print("dependency parsing on a.txt")
    dependency_parse(os.path.join(dirpath, 'a.txt'), cp=cp, tokenize=True)
    print("dependency parser on b.txt")
    dependency_parse(os.path.join(dirpath, 'b.txt'), cp=cp, tokenize=True)
    print("constituency parser a.txt")
    constituency_parse(os.path.join(dirpath, 'a.txt'), cp=cp, tokenize=True)
    print("constituency parser b.txt")
    constituency_parse(os.path.join(dirpath, 'b.txt'), cp=cp, tokenize=True)


if __name__ == '__main__':
    print('=' * 80)
    print('Parsing Fake News Dataset')
    print('=' * 80)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/FNC_Data/Parsed_Data',
                        help='path to dataset')

    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    #data_dir = os.path.join(base_dir, 'data')
    
    
    sick_dir = os.path.join(args.data)
    print(sick_dir)
    lib_dir = os.path.join(base_dir, 'lib')
    
    train_dir =  os.path.join(args.data, 'train/')
    test_dir =  os.path.join(args.data, 'test/')
    dev_dir =  os.path.join(args.data, 'dev/')
    print(train_dir)
    print(train_dir)
    print(train_dir)
    make_dirs([train_dir, dev_dir, test_dir])
    

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])

    # split into separate files
    #split(os.path.join(sick_dir, 'final_train.csv'), train_dir)
    #split(os.path.join(sick_dir, 'final_test.csv'), test_dir)
    #split(os.path.join(sick_dir, 'SICK_test_annotated.txt'), test_dir)

    # parse sentences
    parse(train_dir, cp=classpath)
    parse(dev_dir, cp=classpath)
    parse(test_dir, cp=classpath)

    # get vocabulary
    build_vocab(
        glob.glob(os.path.join(sick_dir, '*/*.toks')),
        os.path.join(sick_dir, 'vocab.txt'))
    build_vocab(
        glob.glob(os.path.join(sick_dir, '*/*.toks')),
        os.path.join(sick_dir, 'vocab-cased.txt'),
        lowercase=False)
