# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable 
import torch.nn as nn           
import torch.nn.functional as F  
import numpy as np

import config
conf = config.config()

USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

def prepare_sequence(word_seq_id, dep_seq_id, pos_seq_id, target_seq_id):
    seq_lens = [len(seq_id) for seq_id in word_seq_id]
    max_l    = max(seq_lens)
    indexs = np.argsort(seq_lens)[::-1].tolist()
    
    word_seq_id=np.array(word_seq_id)[indexs]
    dep_seq_id=np.array(dep_seq_id)[indexs]
    pos_seq_id=np.array(pos_seq_id)[indexs]
    target_seq_id=np.array(target_seq_id)[indexs]
    
    word_seq_id1=[]
    dep_seq_id1=[]
    pos_seq_id1=[]
    target_seq_id1=[]
    mask=[]
    for w_seq, d_seq, p_seq, t_seq in zip(word_seq_id, dep_seq_id, pos_seq_id, target_seq_id):
        if len(w_seq)!=len(d_seq) or len(d_seq)!=len(p_seq) or len(p_seq)!=len(w_seq):
            print "sth wrong with w_seq, d_seq, p_seq"
        
        word_seq_id1.append(w_seq+[0]*(max_l-len(w_seq)))
        dep_seq_id1.append(d_seq+[0]*(max_l-len(d_seq)))
        pos_seq_id1.append(p_seq+[0]*(max_l-len(p_seq)))
        if conf.num_directions == 2:
            target_seq_id1.append(t_seq+[0]*(max_l-len(t_seq)-2))
            mask.append([1]*(len(w_seq)-2)+[0]*(max_l-len(w_seq)))
        else:
            target_seq_id1.append(t_seq+[0]*(max_l-len(t_seq)))
            mask.append([1]*len(w_seq)+[0]*(max_l-len(w_seq)))
    
    return Variable(LongTensor(word_seq_id1)),\
           Variable(LongTensor(dep_seq_id1)), \
           Variable(LongTensor(pos_seq_id1)), \
           Variable(LongTensor(target_seq_id1)), \
           seq_lens,\
           indexs,\
           np.array(mask),\
           max_l
    

class vanilla_RNN(nn.Module):
    def __init__(self, freq_vocab, word2id, dep2id, pos2id):
        super(vanilla_RNN, self).__init__()
        self.word_embeddings = nn.Embedding(len(word2id)+1, conf.emb_dim)
        self.dep_embeddings  = nn.Embedding(len(dep2id)+1,  conf.dep_dim)
        self.pos_embeddings  = nn.Embedding(len(pos2id)+1,   conf.pos_dim)
        self.init_all_embeddings()
        
        self.word_embeddings = self.word_embeddings.cpu()
        self.dep_embeddings  = self.dep_embeddings.cpu()
        self.pos_embeddings  = self.pos_embeddings.cpu()
        
        #initialize RNN
        self.rnn = nn.RNN(conf.emb_dim+conf.dep_dim+conf.pos_dim, 
                          conf.hidden_dim, conf.num_layers, 
                          batch_first=True, bidirectional=True if conf.num_directions==2 else False)
        self.params_init(self.rnn.named_parameters())
        
        #initialize linear
        self.linear = nn.Linear(conf.hidden_dim*conf.num_directions, conf.vocab_size+1)
        self.params_init(self.linear.named_parameters())
        
        self.freq_vocab=freq_vocab
        self.word2id=word2id
        self.dep2id=dep2id
        self.pos2id=pos2id
        
    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(conf.num_layers*conf.num_directions, batch_size, conf.hidden_dim))
        return h0.cuda() if USE_CUDA else h0

    def init_all_embeddings(self):
        self.word_embeddings.weight = nn.init.xavier_uniform(self.word_embeddings.weight)
        self.dep_embeddings.weight = nn.init.xavier_uniform(self.dep_embeddings.weight)
        self.pos_embeddings.weight = nn.init.xavier_uniform(self.pos_embeddings.weight)
        
    def params_init(self, params):
        for name, param in params:
            if len(param.data.shape)==2:
                print(name)
                nn.init.kaiming_normal(param, a=0, mode='fan_in')
            if len(param.data.shape)==1:
                nn.init.normal(param)


    def forward(self, word_seq_id, dep_seq_id, pos_seq_id, target_seq_ids, h0, is_training=False):
        word_padded_ids, dep_padded_ids, pos_padded_ids, target_padded_ids, seq_lens, indexs, mask, max_l = \
        prepare_sequence(word_seq_id, dep_seq_id, pos_seq_id, target_seq_ids)
        
        word_vecs = self.word_embeddings(word_padded_ids)
        dep_vecs  = self.dep_embeddings(dep_padded_ids)
        pos_vecs  = self.pos_embeddings(pos_padded_ids)
        input_x = torch.cat((word_vecs, dep_vecs, pos_vecs), 2)
        
        '''
        input_seq_packed = torch.nn.utils.rnn.pack_padded_sequence(input_x, seq_lens, batch_first=True)
        out_pack, hx = self.rnn(input_seq_packed, self.hidden)
        out, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=True)
        '''
        out, hx = self.rnn(input_x, h0)
        #out  = out.contiguous().view(-1, conf.hidden_dim*2)
        #mask = mask.view(-1)
        if conf.num_directions==2:
            forward_out, backward_out = out[:, :-2, :conf.hidden_dim], out[:, 2:, conf.hidden_dim:]
            out_cat = torch.cat((forward_out, backward_out), dim=-1)
       
        logits = self.linear(out_cat if conf.num_directions==2 else out )
        probs=0
        if is_training==False:
            probs = F.softmax(logits, dim=2)
        #pred = probs.data.cpu().numpy().argmax(2)
        return logits, probs, word_padded_ids, target_padded_ids, indexs, mask
        
 
    
