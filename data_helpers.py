from loader import load_data, seq_classes, seq_tokens
import json,random,os
import loader

import numpy as np

# load pre split dataset
with open('./data/train.json') as fp:
    training_data = json.load(fp) # use for training and hyperparam opt - generate splits for CV
with open('./data/test.json') as fp:
    test_data = json.load(fp) # use for final evaluation
with open('./data/blind.json') as fp:
    blind_data = json.load(fp) # unlabelled data, predict for submission

# ToDo: n-fold cross validation
dev_split = 0.1
dev_ix = round(len(training_data)*(1-dev_split))
train_data = training_data[:dev_ix]
dev_data = training_data[dev_ix:]
# dev_data=dev_data[:200]

num_classes = len(seq_classes)
num_tokens = len(seq_tokens)
pad_token = seq_tokens.index('_')
seq_classes_rev = {v:k for k,v in seq_classes.items()}

def get_classes_ratios(data):
    data_dict = {c:[] for c in range(num_classes)}
    for x in data:
        data_dict[x[1]].append(x[0])
    class_ratios = {c:len(data_dict[c])/sum(len(data_dict[c2]) for c2 in data_dict.keys()) for c in data_dict.keys()}
    return class_ratios
print('Class ratios (train):',get_classes_ratios(train_data))
print('Class ratios (dev):',get_classes_ratios(dev_data))
print('Class ratios (test):',get_classes_ratios(test_data))
training_class_ratios = get_classes_ratios(train_data)

# print(train_data_dict)

train_data_dict = {c:[] for c in range(num_classes)}
for x in train_data:
    train_data_dict[x[1]].append(x[0])



# Use this to check all tokens are expected
print('Checking training tokens')
loader.check_tokens(train_data)
print('Checking dev tokens')
loader.check_tokens(dev_data)
print('Checking test tokens')
loader.check_tokens(test_data)

# print('All train set length: ', len(training_data))
print('Training set length: ', len(train_data))
print('Dev set length: ', len(dev_data))
print('Test set length: ', len(test_data))

# ToDo: can we do this in advance?
def get_batch(data, start, length):
    batch_tokens=[]
    batch_ys=[]
    max_seq_length=0
    for sample in data[start:(start+length)]:
        batch_ys.append(sample[1])
        batch_tokens.append([seq_tokens.index(char) for char in sample[0]])
        max_seq_length = max(max_seq_length, len(sample[0]))
    batch_xs = np.zeros([length, max_seq_length], int)
    for i in range(len(batch_tokens)):
        this_len = len(batch_tokens[i])
        batch_xs[i,:this_len] = np.asarray(batch_tokens[i])
    return batch_xs, batch_ys

def get_batch_reweighted(data_dict, length, weights):
    batch_tokens=[]
    batch_ys=[]
    max_seq_length=0
    for i in range(length):
        ix = random.random()
        this_class = np.random.choice([c for c in range(num_classes)])
        sample = np.random.choice(data_dict[this_class])
        batch_ys.append(this_class)
        batch_tokens.append([seq_tokens.index(char) for char in sample])
        max_seq_length = max(max_seq_length, len(sample))
    batch_xs = np.zeros([length, max_seq_length], int)
    for i in range(len(batch_tokens)):
        this_len = len(batch_tokens[i])
        batch_xs[i,:this_len] = np.asarray(batch_tokens[i])
    return batch_xs, batch_ys
