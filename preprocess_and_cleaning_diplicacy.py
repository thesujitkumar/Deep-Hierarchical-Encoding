import os

import sys

from config import parse_args
import time
global args
args = parse_args()
train_dir =   os.path.join(args.data, 'train/')#'data/sick/train/'
#dev_dir = os.path.join(args.data, 'dev/')
test_dir = os.path.join(args.data, 'test/') #'data/sick/test/' #


train_files = os.listdir(train_dir)

train_a_files = [ fname for fname in train_files if fname.startswith('a.')]

fin = open(os.path.join(train_dir, 'info_train.pickle'), 'rb')

d = pickle.load(fin) # Load info list
finc.close()

s_count = 0
sentence2body = {}

for b_id, p_list in enumerate(d[:10]):

    s_sum = sum(p_list)
    print(b_id, p_list, s_count, s_sum)
    for s_id in range(s_count, s_count + s_sum):
        sentence2body[s_id] = b_id
    s_count += s_sum
