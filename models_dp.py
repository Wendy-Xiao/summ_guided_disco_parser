from collections import Counter
from random import random
from nltk import word_tokenize
from torch import nn
from torch.autograd import Variable
import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import json
import re
import numpy as np
import torch
import torch.nn.functional as F
import os
from transformer.EncoderForSumm import *
from transformers import BertModel, BertConfig
# from transformers import BertConfig
# from bert_model import BertModel
import math



# class Transformer(nn.module):
# 	def __init__(self, n_layers, n_head, d_model, d_ff=2048, dropout=0.1):
# 		self.layers = nn.ModuleList(
# 			[nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout)
# 			 for _ in range(n_layers)])
# 		self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
# 	def forward(self, src, mask):

class Bert(nn.Module):
	def __init__(self, temp_dir, load_pretrained_bert, bert_config=None, finetune=True):
		super(Bert, self).__init__()
		if(load_pretrained_bert):
			self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
			if finetune:
				self.model.train()
				for param in self.model.parameters():
					param.requires_grad = True
			else:
				self.model.eval()
				for param in self.model.parameters():
					param.requires_grad = False
		else:
			self.model = BertModel(bert_config)

	def forward(self, x, segs, mask,return_attention=False):
		if return_attention:
			last_hidden_state, _, attentions = self.model(x, token_type_ids=segs, attention_mask =mask, output_attentions=True,return_dict=False)
			return last_hidden_state,attentions
		else:
			last_hidden_state, _ = self.model(x, token_type_ids=segs, attention_mask =mask,return_dict=False)
			return last_hidden_state

class BertSentEncoder(nn.Module):
	def __init__(self,temp_dir,unit='edu',max_length=512, finetune=False,batch_size=512):
		super(BertSentEncoder, self).__init__()
		self.model = Bert(temp_dir,True,finetune=finetune)
		self.unit = unit
		self.max_length=max_length
		self.batch_size=batch_size


	def _cut_unit(self,src):
		if len(src)>self.max_length:
			src = src[:self.max_length-1]
			src=src+[102]
		return src

	def _pad(self, data, pad_id):
		width = max(len(d) for d in data)
		# rtn_data = torch.
		rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
		return rtn_data

	def get_bert_representation_single(self, src, d_span=None, device=0):
		if self.unit == 'sent':
			src = [102] + src
			position_of_sep = np.where(np.array(src)==102)[0]
			unit_list = [self._cut_unit(src[(position_of_sep[i]+1):(position_of_sep[i+1]+1)]) for i in range(len(position_of_sep)-1)]
		elif self.unit=='edu':
			unit_list = [self._cut_unit([101]+src[d_span[i][0]:d_span[i][1]]+[102]) for i in range(len(d_span))]
		# print(unit_list)
		output = []
		num_batch = math.ceil(len(unit_list)/self.batch_size)
		for i in range(num_batch):
			start = i*self.batch_size
			end = (i+1) *self.batch_size
			unit_input = unit_list[start:end]
			segs = [[0]*len(unit) for unit in unit_input]
			mask = [[1]*len(unit) for unit in unit_input]
			unit = torch.tensor(self._pad(unit_input, 0)).to(device)
			segs = torch.tensor(self._pad(segs, 0)).to(device)
			mask = torch.tensor(self._pad(mask, 0)).to(device)
			out = self.model(unit,segs,mask)[:,0]
			output.append(out)
		output = torch.cat(output,0) #length * embedding_size
		return output

	def get_bert_representation_parallel(self, src_batch, d_span_batch=None, device=0):
		all_unit_list_batch =[]
		all_data_length=[]
		for i in range(len(src_batch)):
			src = src_batch[i]
			d_span = d_span_batch[i]
			if self.unit == 'sent':
				src = [102] + src
				position_of_sep = np.where(np.array(src)==102)[0]
				unit_list = [self._cut_unit(src[(position_of_sep[i]+1):(position_of_sep[i+1]+1)]) for i in range(len(position_of_sep)-1)]
			elif self.unit=='edu':
				unit_list = [self._cut_unit([101]+src[d_span[i][0]:d_span[i][1]]+[102]) for i in range(len(d_span))]
			all_unit_list_batch.extend(unit_list)
			all_data_length.append(len(unit_list))

		output = []
		num_batch = math.ceil(len(all_unit_list_batch)/self.batch_size)
		for i in range(num_batch):
			start = i*self.batch_size
			end = (i+1) *self.batch_size
			unit_input = all_unit_list_batch[start:end]
			segs = [[0]*len(unit) for unit in unit_input]
			mask = [[1]*len(unit) for unit in unit_input]
			unit = torch.tensor(self._pad(unit_input, 0)).to(device)
			segs = torch.tensor(self._pad(segs, 0)).to(device)
			mask = torch.tensor(self._pad(mask, 0)).to(device)
			out = self.model(unit,segs,mask)[:,0]
			output.append(out)
		output = torch.cat(output,0) 

		start=0
		bert_representation=[]
		for l in all_data_length:
			bert_representation.append(output[start:start+l])
			start+=l
		del output
		return pad_sequence(bert_representation,batch_first=True)




	def forward(self,src_batch,d_span_batch,device):
		# bert_representation = [self.get_bert_representation_single(src_batch[i], d_span_batch[i],device) for i in range(len(src_batch))]
		# bert_representation = pad_sequence(bert_representation,batch_first=True)
		bert_representation = self.get_bert_representation_parallel(src_batch, d_span_batch, device)
		return bert_representation




class PositionalEncoding(nn.Module):
	def __init__(self, dropout, dim, max_len=1000):
		pe = torch.zeros(max_len, dim)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
							  -(math.log(10000.0) / dim)))
		pe[:, 0::2] = torch.sin(position.float() * div_term)
		pe[:, 1::2] = torch.cos(position.float() * div_term)
		pe = pe.unsqueeze(0)
		super(PositionalEncoding, self).__init__()
		self.register_buffer('pe', pe)
		self.dropout = nn.Dropout(p=dropout)
		self.dim = dim

	def forward(self, emb, step=None):
		emb = emb * math.sqrt(self.dim)
		if (step):
			emb = emb + self.pe[:, step][:, None, :]

		else:
			emb = emb + self.pe[:, :emb.size(1)]
		emb = self.dropout(emb)
		return emb

	def get_emb(self, emb):
		return self.pe[:, :emb.size(1)]


class EDUExtractor(nn.Module):
	def __init__(self,dim):
		super(EDUExtractor, self).__init__()
		self_attention = []
		self_attention.append(nn.Linear(dim, dim))
		self_attention.append(nn.ReLU())
		self_attention.append(nn.Linear(dim, 1))
		self.self_attention = nn.Sequential(*self_attention)
		# for param in self.model.parameters():
		# 	param.requires_grad = False

	def forward(self, top_vec, edu_span,edu_mask):
		# edu_span: batch * length * edu_num
		# top_vec: batch * length * bert_embedding
		# edu_mask: batch * edu_num

		inv_edu_mask = 1-edu_mask.unsqueeze(1).expand(edu_span.size())
		scores = self.self_attention(top_vec) #batch * length * 1

		scores = scores * edu_span # batch * length * edu_num
		inf_entrnace = ~((edu_span+inv_edu_mask).type(torch.bool))
		scores.masked_fill_(inf_entrnace,float('-inf'))
		edu_scores = torch.nn.functional.softmax(scores,dim=1)
		# edu_scores[edu_scores!=edu_scores]=0
		edu_representation = torch.bmm(edu_scores.permute(0,2,1),top_vec)
		return edu_representation



    
class DiscoExtSumm_dp(nn.Module):
    def __init__(self, args, load_pretrained_bert, bert_config=None, args_dict=False):
        super(DiscoExtSumm_dp, self).__init__()

        dropout=args.dropout
        unit=args.unit
        bert_dir=args.bert_dir
        n_layers, n_head, d_k, d_v, d_inner,d_mlp=args.n_layers, args.n_head, args.d_k, args.d_v, args.d_inner,args.d_mlp

        self.posemb ='sequential'
        self.PositionEncoder = PositionalEncoding(dropout, 768)	

        self.tree_attn = False
        self.bert_unit_encoder = True
        self.unit = unit

        self.bertsentencoder =BertSentEncoder(bert_dir, unit)

#         self.RTransformer = RTransEncoder(n_layers, n_head, d_k, d_v,768, d_inner)
        # self.balance_tree_seq = nn.Sequential(nn.Linear(768, 100),nn.ReLU(),nn.Linear(100, 1))
        self.Transformer = Encoder(n_layers, n_head, d_k, d_v,768, d_inner)
        # self.Transformer = nn.TransformerEncoder(n_layers, n_head,)
        self.Dropoutlayer = nn.Dropout(p=dropout)
        self.Decoderlayer = self.build_decoder(768,d_mlp,dropout)

    def self_attention(self,tree_embed,seq_embed):
        tree_scoe = self.balance_tree_seq(tree_embed)
        seq_scoe = self.balance_tree_seq(seq_embed)
        scores = F.softmax(torch.cat([tree_scoe,seq_scoe],dim=2),dim=2)
        attentive_embed = scores[:,:,0].unsqueeze(2)*tree_embed+scores[:,:,1].unsqueeze(2)*seq_embed
        print(scores[0])
        del tree_scoe,seq_scoe,scores
        return attentive_embed

    def build_decoder(self,input_size,mlp_size,dropout,num_layers=1):
        decoder = []
        for i in range(num_layers):
            decoder.append(nn.Linear(input_size, mlp_size))
            decoder.append(nn.ReLU())
            decoder.append(nn.Dropout(p=dropout))
        decoder.append(nn.Linear(mlp_size, 1))
        return nn.Sequential(*decoder)


    def forward(self, batch,device):

        ##### use bert as doc-encoder or sent encoder
        unit_repre = self.bertsentencoder(batch.src_list,batch.d_span_list,device)



        #### position embedding

        pos_emb = self.PositionEncoder.pe[:, :unit_repre.size()[1]].expand(unit_repre.size()) #batch * edu_num * embedding_dim
        inputs = unit_repre+pos_emb

        out,attn = self.Transformer(inputs,batch.unit_mask,return_attns=True)
        # print(len(attn))
        # print(attn[0].shape)
        attn = [a.detach().cpu() for a in attn]
        # del unit_repre
#         out = self.RTransformer(inputs, batch.unit_mask, batch.hi_attn_map)
        # print(batch.unit_mask)

        # out = self.self_attention(out,unit_repre)
#         out = unit_repre
        out = self.Dropoutlayer(out)
        # out = self.Dropoutlayer(out)
        out = self.Decoderlayer(out) # batch * length * 1
        return out,attn,unit_repre
