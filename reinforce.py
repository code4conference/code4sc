#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn            
import torch.nn.functional as F   
import torch.optim as optim       
import numpy as np
from torch.distributions import Categorical

import policy_config
conf = policy_config.config()

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


def reward_length(x):
    return 16*x**2*(1-x)**2

def reward_grammar(word_seq_id, dep_seq_id, pos_seq_id, language_model, pred_actions, mask, indexs):
    
    new_word_seq_id = np.array(word_seq_id)[indexs].tolist()
    new_dep_seq_id  = np.array(dep_seq_id)[indexs].tolist()
    new_pos_seq_id  = np.array(pos_seq_id)[indexs].tolist()
    
    word2id = language_model.word2id
    dep2id = language_model.dep2id
    pos2id = language_model.pos2id

    new_word_seq_id1 = []
    new_dep_seq_id1 = []
    new_pos_seq_id1 = []
    for i in range(len(pred_actions)):
        temp_word = [word2id['BOS']]
        temp_dep  = [dep2id['BOS']]
        temp_pos  = [pos2id['BOS']]
        for j in range(sum(mask[i])):
            if pred_actions[i][j]==1:
                temp_word.append(new_word_seq_id[i][j])
                temp_dep.append(new_dep_seq_id[i][j])
                temp_pos.append(new_pos_seq_id[i][j])
        
        temp_word.append(word2id['EOS'])
        temp_dep.append(dep2id['EOS'])
        temp_pos.append(pos2id['EOS'])
        
        new_word_seq_id1.append(temp_word)
        new_dep_seq_id1.append(temp_dep)
        new_pos_seq_id1.append(temp_pos)
            
    new_target_seq_id1=[]
    for sent in new_word_seq_id1:
        new_target_seq_id1.append(sent[1:-1])
    
    h0 = language_model.init_hidden(conf.batch_size)
    
    _, probs, _, target_padded_ids, indexs, mask = \
    language_model(new_word_seq_id1, new_dep_seq_id1, new_pos_seq_id1, new_target_seq_id1, h0, False)
    
    
    ori_index = np.argsort(indexs).tolist()
    probs=probs[ori_index].cpu().data.numpy()
    mask=mask[ori_index].cpu().data.numpy()
    target_padded_ids=target_padded_ids[ori_index].cpu().data.numpy()
 
    
    print 
    ppl=[]
    for i in range(len(mask)):
        l=int(sum(mask[i]))
        temp=[]
        for probability, id_ in zip(probs[i][:l], target_padded_ids[i][:l]):
            print probability[id_]
            temp.append(np.log(probability[id_]))
        
        ppl.append(np.exp(-sum(temp)/float(len(temp))))
    
    rewards = FloatTensor(ppl)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    return rewards
     

def select_action(word_seq_id, dep_seq_id, pos_seq_id, policy, language_model, optimizer):
    h0 = policy.init_hidden(conf.batch_size)
    _, probs, indexs, mask, pred = policy(word_seq_id, dep_seq_id, pos_seq_id, h0, True)
    
    pred_actions=[]
    for prob, index, mask_i in zip(probs, indexs, mask):
        length = sum(mask_i).astype('int64')
        m = Categorical(prob[:length])
        actions = m.sample()
        pred_actions.append(actions.data.numpy().tolist()+[-1]*(mask.shape[1]-length))
        policy.saved_log_probs.append(m.log_prob(actions))
        policy.pred_actions=pred_actions
    
    
    policy.length_r = reward_length(model.length_r)
    policy.grammar_r =reward_grammar(word_seq_id, dep_seq_id, pos_seq_id, language_model, 
                                    pred_actions, mask, indexs)
    
    batch_loss=0
    for i in range(mask.shape[0]):
        batch_loss+=finish_episode(policy, i)#/float(length)
    batch_loss = batch_loss/conf.batch_size 
    
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()
    
    policy.saved_log_probs  = []
    policy.length_r  = []
    policy.grammar_r = []
    
    return batch_loss

def finish_episode(model, index):
    model_loss = []
    for log_prob in model.saved_log_probs[index]:
        model_loss.append(-log_prob * (model.grammar_r[index]+model.length_r[index]))
    
    model_loss = torch.cat(model_loss).sum()
    return model_loss


    
    