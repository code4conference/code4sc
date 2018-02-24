# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from torch.autograd import Variable 
import torch.nn as nn           
import torch.nn.functional as F  
import numpy as np
torch.manual_seed(1337)

import config
conf = config.config()
import collections as col

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

#we use https://spacy.io/models/ to parse sentences
f_parsing=open('./your_own_parsing_ouput.txt', "r")
toy_parsing=[]
temp=[]
for line in f_parsing.readlines():
    if line.split()==[]:
        toy_parsing.append(temp)
        temp=[]
    else:
        temp.append(line.split())
        
f_parsing.close()


pos_seq=[]
dep_seq=[]
child=[]
father=[]
ori_seq=[]
lower_seq=[]
for sent in toy_parsing:
    child_dict={}
    temp_father=[]
    for k in range(len(sent)+1):
        child_dict[k]=[]
        temp_father.append(k)
    
    temp_pos=['BOS']
    temp_dep=['BOS']
    temp_ori_seq=['BOS']
    temp_lower_seq=['BOS']
    for token in sent:
        temp_ori_seq.append(token[2])
        temp_lower_seq.append(token[2].lower())
        child_dict[int(token[1])].append(int(token[0]))
        temp_father[int(token[0])]=int(token[1])
        temp_pos.append(token[5])
        temp_dep.append(token[4])
    
    temp_pos[-1]='EOS'
    temp_dep[-1]='EOS'
    temp_ori_seq[-1]='EOS'
    temp_lower_seq[-1]='EOS'
    
    ori_seq.append(temp_ori_seq)
    lower_seq.append(temp_lower_seq)
    
    father.append(temp_father)
    child.append(child_dict)
    pos_seq.append(temp_pos)
    dep_seq.append(temp_dep)

#%%
"""
inputs are word_seq, dep_seq, pos_seq
"""
for w, d, p in zip(lower_seq, dep_seq, pos_seq):
    if len(w)!=len(d) or len(w)!=len(p) or len(d)!=len(p):
        print "something wrong!"


def mapping(_list):
    _2id={}
    for i, item in enumerate(_list):
        _2id[item]=i+1

    return _2id
    

def prepare_data2id(word_seq, dep_seq, pos_seq):
    word_vocab=set()
    dep_vocab=set()
    pos_vocab=set()
    bag_of_words = []
    
    for w_seq_i, d_seq_i, p_seq_i in zip(word_seq, dep_seq, pos_seq):
        for w_i, d_i, p_i in zip(w_seq_i, d_seq_i, p_seq_i):
            bag_of_words.append(w_i)
            word_vocab.add(w_i)
            dep_vocab.add(d_i)
            pos_vocab.add(p_i)
    
    freq_words = col.Counter(bag_of_words).items()
    freq_words=sorted(freq_words, key=lambda s:s[-1], reverse=True)
    assert len(freq_words)>conf.vocab_size
    freq_vocab=[]
    for w in freq_words[:conf.vocab_size]:
        freq_vocab.append(w[0])
    
    vocab_set=set(freq_vocab)
    dep_vocab=list(dep_vocab)
    pos_vocab=list(pos_vocab)
    
    word2id=mapping(freq_vocab)
    dep2id=mapping(dep_vocab)
    pos2id=mapping(pos_vocab)
    
    word_seq_id=[]
    dep_seq_id=[]
    pos_seq_id=[]
    target_seq_id=[]
    for w_seq_i, d_seq_i, p_seq_i in zip(word_seq, dep_seq, pos_seq):
        temp_w=[]
        temp_dep=[]
        temp_pos=[]
        for w_i, d_i, p_i in zip(w_seq_i, d_seq_i, p_seq_i):
            if w_i in vocab_set:
                temp_w.append(word2id[w_i])
            else:
                temp_w.append(len(freq_vocab))
            
            temp_dep.append(dep2id[d_i])
            temp_pos.append(pos2id[p_i])
        
        word_seq_id.append(temp_w)
        dep_seq_id.append(temp_dep)
        pos_seq_id.append(temp_pos)
        target_seq_id.append(temp_w[1:-1])
        
        
        
    
    
    return word_seq_id, target_seq_id, dep_seq_id, pos_seq_id, \
           word2id,     dep2id,        pos2id,     freq_vocab
           

word_seq_id, target_seq_id, dep_seq_id, pos_seq_id, \
word2id,     dep2id,        pos2id,     freq_vocab = prepare_data2id(lower_seq, dep_seq, pos_seq)

#%%

import RNN, masked_cross_entropy
lm_model = RNN.vanilla_RNN(freq_vocab, word2id, dep2id, pos2id,)
if USE_CUDA:
    lm_model = lm_model.cuda()

optimizer = optim.Adam(lm_model.parameters(),lr=conf.lr)

#%%
train_word_seq_id, train_target_seq_id, train_dep_seq_id, train_pos_seq_id = \
word_seq_id[2000:], target_seq_id[2000:], dep_seq_id[2000:], pos_seq_id[2000:]

val_word_seq_id, val_target_seq_id, val_dep_seq_id, val_pos_seq_id = \
word_seq_id[1000:2000], target_seq_id[1000:2000], dep_seq_id[1000:2000], pos_seq_id[1000:2000]

test_word_seq_id, test_target_seq_id, test_dep_seq_id, test_pos_seq_id = \
word_seq_id[:1000], target_seq_id[:1000], dep_seq_id[:1000], pos_seq_id[:1000]


#%%
bz = conf.batch_size
for epoch in range(100):
    #total_loss = 0
    losses=[]
    
    train_data = zip(train_word_seq_id, train_target_seq_id, train_dep_seq_id, train_pos_seq_id)
    np.random.shuffle(train_data)
    train_word_seq_id, train_target_seq_id, train_dep_seq_id, train_pos_seq_id= zip(*train_data)

    nb = len(train_word_seq_id)/bz
    for b_i in range(nb):
        h0 = lm_model.init_hidden(bz)
        lm_model.zero_grad()
        logits, probs, word_padded_ids, target_padded_ids, indexs, mask = \
        lm_model(train_word_seq_id[b_i*bz:(b_i+1)*bz], 
                 train_dep_seq_id[b_i*bz:(b_i+1)*bz],
                 train_pos_seq_id[b_i*bz:(b_i+1)*bz],
                 train_target_seq_id[b_i*bz:(b_i+1)*bz],
                 h0,
                 is_training=True)
        
        seq_lens = Variable(torch.sum(LongTensor(mask), 1))
        loss = masked_cross_entropy.compute_loss(logits, target_padded_ids, seq_lens)
        
        
        loss.backward()
        #torch.nn.utils.clip_grad_norm(lm_model.parameters(), 0.5) # gradient clipping
        optimizer.step()
    
     
           
