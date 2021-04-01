from random import random
from torch import nn
import re
import numpy as np
import torch
import os
from pathlib import Path
import sys
import argparse
from transformers import BertConfig

from models_dp import DiscoExtSumm_dp
from dep_parsing_summ import chu_liu_edmond,chu_liu_edmond_hierarchical_avg,Eisner,Eisner_sent_constraint
from run_dp import evaluate_batch
from dataloader_dp import SummarizationDataset,SummarizationDataLoader
from evaluation import evaluate_dep_tree_list


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-bert_dir", default='./bert_model/')
	parser.add_argument("-d_v", type=int, default=64)
	parser.add_argument("-d_k", type=int, default=64)
	parser.add_argument("-d_inner", type=int, default=3072)
	parser.add_argument("-d_mlp", type=int, default=100)

	parser.add_argument("-n_layers", type=int, default=2)
	parser.add_argument("-n_head", type=int, default=1)
	parser.add_argument("-dropout", type=float, default=0.1)

	# input and output directory.
	# parser.add_argument("-train_inputs_dir", default='/scratch/wenxiao/DiscoBERT_CNNDM_tree_embed/train', type=str)
	parser.add_argument("-unit", default='edu', type=str)
	parser.add_argument("-device", default=1, type=int)
	parser.add_argument("-attention_type", default='self-attention', type=str)
	parser.add_argument("-model_name", default='edu_bertunitencoder_transformer_singlehead_64', type=str)

	args = parser.parse_args()
	print(args)


	test_inputs_dir_list=['./data/rstdt/all/', './data/instruction_dataset/all','./data/GUM_binary/']
	dataset_list=['rstdt-full','instructional','GUM']

	model_name = args.model_name
	ref_path = './ref_%s/'%(model_name)
	hyp_path = './hyp_%s/'%(model_name)
	save_path = './models/%s/'%(model_name)
	bert_config= './bert_config_uncased_base.json'
	unit = args.unit
	use_edu=(unit=='edu')
	bert_unit_encoder = True

	if not os.path.exists(hyp_path):
		os.makedirs(hyp_path)
	if not os.path.exists(ref_path):
		os.makedirs(ref_path)

	device = torch.device("cuda:%d"%(args.device))
	torch.cuda.set_device(args.device)

	config = BertConfig.from_json_file(bert_config)
	config.max_position_embeddings=2048

	print('load model')
	model = DiscoExtSumm_dp(args,load_pretrained_bert=True,bert_config=config).to(device)
	MODEL_PATH = save_path+'/best_r2'
	# MODEL_PATH = save_path
	if torch.cuda.is_available():
		model=model.to(device)
		
	try:
		model.load_state_dict(torch.load(MODEL_PATH,map_location=device),strict=False)
	except:
		trained_states = torch.load(MODEL_PATH,map_location=device)
		del trained_states['bertsentencoder.model.model.embeddings.position_ids']
		model.load_state_dict(trained_states)
	# model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
	model.eval()
	######################
	## load data ##
	######################
	for i_data, test_inputs_dir in enumerate(test_inputs_dir_list):

		print('load data')
		test = SummarizationDataset(test_inputs_dir,to_shuffle=False, is_test=True)
		test_dataloader = SummarizationDataLoader(test,is_test=True,device=device,batch_size=1, unit=unit)	
		print('start evaluate')
		sys.stdout.flush()

		attn_list = []
		disco_to_sent_list = []
		ids = []
		unit_repre_list = []
		sent_to_para_list=[]
		for batch in test_dataloader:

			batch_ids,attn= evaluate_batch(model,batch,device)
			attn_list.append(attn)
			disco_to_sent_list.extend(batch.disco_to_sent)
			ids.extend(batch_ids)
			unit_repre_list.append(unit_repre)
			sent_to_para_list.extend(batch.sent_to_para)
		#     batch_disco_txt = batch.unit_txt[0]
		#     if attn[0].shape[1]<=30:
		#     break

		######################
		## load gt tree ##
		######################
		dep_tree_gt_list = []
		all_gt = torch.load('./summ_guided_dp_gt/%s-gt.pt'%(dataset_list[i_data]))
		for f in ids:
			dep_tree_gt=all_gt[f]['dep_tree']
			dep_tree_gt_list.append(dep_tree_gt)

		all_result={}
		for layer in [0,1]:
			for cons in ['nocons','sent_cons']:
				for t in ['cle','eisner']:
					print('Dataset: %s, Model:%s, Layer use: %d, constraint type: %s, construct type: %s'\
						%(dataset_list[i_data], args.model_name, layer,cons,t))
					sys.stdout.flush()	
					# print(ids)
					# print(len(attn_list))
					dep_tree_cand_list = []
					result={}
					# disco_to_sent_list[]
					for i in range(len(attn_list)):
						# can be changed to each attention head here.
						mat=torch.mean(attn_list[i][layer],0)
						G={}
						for src in range(mat.shape[0]):
							G[src]={}
							self_attn = mat[src][src]
							for dst in range(mat.shape[1]):
								if src!=dst:
						#             G[src][dst]=1-(mat[src][dst]/(mat[src].sum()-self_attn)).detach().numpy()
									G[src][dst]=float(mat[dst][src])
								else:
									G[src][dst]=mat[dst][src]

						if t=='cle':
							root = int(mat.sum(dim=0).argmax())
							if cons=='nocons':
								dep_tree_cand=chu_liu_edmond(G,root)
							elif cons=='sent_cons':
								dep_tree_cand=chu_liu_edmond_hierarchical_avg(G,root,disco_to_sent_list[i])

						elif t=='eisner':
							if cons=='nocons':
								dep_tree_cand=Eisner(G)
							elif cons=='sent_cons':
								dep_tree_cand=Eisner_sent_constraint(G,disco_to_sent_list[i])

						dep_tree_cand_list.append(dep_tree_cand)
						result[ids[i]] = dep_tree_cand
						# if i%20==0:
						# 	print(i)
						# const_tree_cand=cky_local_max(mat)
					# const_tree_gt = [[[node[0][0]-1,node[0][1]-1],node[1]] for node in list_nodes]
					all_result['%d-%s-%s'%(layer,cons,t)] = result
					evaluate_dep_tree_list(dep_tree_gt_list,dep_tree_cand_list)
		save_path='./summ_guided_dp_result/%s-%s-dep.pt'%(dataset_list[i_data],args.model_name)
		torch.save(all_result,save_path)
		del attn_list
	del model

















