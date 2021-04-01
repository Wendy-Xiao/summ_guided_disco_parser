from random import random
from torch import nn
import re
import numpy as np
import torch
import torch.nn.functional as F
import os
from pathlib import Path
import sys 

from models_dp import DiscoExtSumm_dp
from const_parsing_summ import cky_local_global_max, cky_local_global_max_with_sent_constraint,cky_local_global_max_with_sent_para_cons
from dep_parsing_summ import chu_liu_edmond,chu_liu_edmond_hierarchical_avg,Eisner,Eisner_sent_constraint
from run_dp import evaluate_batch
from dataloader_dp import SummarizationDataset,SummarizationDataLoader
from evaluation import evaluate_dep_tree_list,evaluate_const_tree_list



test_inputs_dir_list=['./data/rstdt/all/', './data/instruction_dataset/all','./data/GUM_binary/']
dataset_list=['rstdt-full','instructional','GUM']

######################
## load data ##
######################
device = torch.device('cpu')
unit='edu'
for i_data, test_inputs_dir in enumerate(test_inputs_dir_list):

	print('load data')
	test = SummarizationDataset(test_inputs_dir,to_shuffle=False, is_test=True)
	test_dataloader = SummarizationDataLoader(test,is_test=True,device=device,batch_size=1, unit=unit)	
	print('start evaluate')
	sys.stdout.flush()

	disco_to_sent_list = []
	ids = []
	sent_to_para_list=[]
	for batch in test_dataloader:
		disco_to_sent_list.extend(batch.disco_to_sent)
		ids.extend(batch.ids)
		sent_to_para_list.extend(batch.sent_to_para)

	######################
	## load dep gt tree ##
	######################
	print('Dependency Tree')
	dep_tree_gt_list = []
	all_gt = torch.load('./summ_guided_dp_gt/%s-gt.pt'%(dataset_list[i_data]))
	for f in ids:
		dep_tree_gt=all_gt[f]['dep_tree']
		dep_tree_gt_list.append(dep_tree_gt)


	for cons in ['nocons','sent_cons']:
		for t in ['cle','eisner']:
			print('Dataset: %s, constraint type: %s, construct type: %s'\
				%(dataset_list[i_data],cons,t))
			sys.stdout.flush()	
			micro_f1_list=[]
			macro_f1_list=[]
			for i_runtime in range(10):
				print('Iteration %d'%(i_runtime))
				dep_tree_cand_list = []
				# disco_to_sent_list[]
				for i in range(len(ids)):
					mat = F.normalize(torch.rand([len(disco_to_sent_list[i]),len(disco_to_sent_list[i])]),p=1,dim=1)
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

				micro_f1,macro_f1=evaluate_dep_tree_list(dep_tree_gt_list,dep_tree_cand_list)
				micro_f1_list.append(micro_f1)
				macro_f1_list.append(macro_f1)
			print('Avg Micro F1: ',np.mean(micro_f1_list),', Std: ',np.std(micro_f1_list))
			print('Avg Macro F1: ',np.mean(macro_f1_list),', Std: ',np.std(macro_f1_list))
			sys.stdout.flush()	

	######################
	## load const gt tree ##
	######################
	print('Constituency Tree')
	const_tree_gt_list = []
	all_gt = torch.load('/scratch/wenxiao/summ_guided_dp_gt/%s-gt.pt'%(dataset_list[i_data]))
	for f in ids:
	    const_tree_gt=all_gt[f]['const_tree']
	    const_tree_gt_list.append(const_tree_gt)

	for cons in ['nocons','sent_cons']:
		print('Dataset: %s, constraint type: %s'\
			%(dataset_list[i_data],cons))
		sys.stdout.flush()	
		micro_f1_list=[]
		macro_f1_list=[]
		for i_runtime in range(10):
			print('Iteration %d'%(i_runtime))
			const_tree_cand_list = []
			# disco_to_sent_list[]
			for i in range(len(ids)):
				mat = F.normalize(torch.rand([len(disco_to_sent_list[i]),len(disco_to_sent_list[i])]),p=1,dim=1)
				if cons=='nocons':
					const_tree_cand=cky_local_global_max(mat)
				elif cons=='sent_cons':
					const_tree_cand=cky_local_global_max_with_sent_constraint(mat,disco_to_sent_list[i])
				elif cons =='sent_para_cons':
					const_tree_cand=cky_local_global_max_with_sent_para_cons(mat,disco_to_sent_list[i],sent_to_para_list[i])
				const_tree_cand_list.append(const_tree_cand)
			micro_f1,macro_f1=evaluate_const_tree_list(const_tree_gt_list,const_tree_cand_list)
			micro_f1_list.append(micro_f1)
			macro_f1_list.append(macro_f1)
		print('Avg Micro F1: ',np.mean(micro_f1_list),', Std: ',np.std(micro_f1_list))
		print('Avg Macro F1: ',np.mean(macro_f1_list),', Std: ',np.std(macro_f1_list))
		sys.stdout.flush()	

	######################
	## rb baseline ##
	######################
	result_all={}
	for cons in ['nocons','sent_cons','sent_para_cons']:
		print('Dataset: %s, constraint type: %s'\
				%(test_inputs_dir,cons))
		sys.stdout.flush()	
		const_tree_bl_list = []
		result={}
		for i in range(len(ids)):
			if cons=='nocons':
			    const_tree_cand=right_branching(len(disco_to_sent_list[i]))
			elif cons=='sent_cons':
			    const_tree_cand=right_branching_with_sent_constraint(len(disco_to_sent_list[i]),disco_to_sent_list[i])
			else:
				const_tree_cand=right_branching_with_sent_para_constraint(len(disco_to_sent_list[i]),disco_to_sent_list[i],sent_to_para_list[i])
			const_tree_bl_list.append(const_tree_cand)
			result[ids[i]]=const_tree_cand

		result_all[cons]=result
		evaluate_const_tree_list(const_tree_gt_list,const_tree_bl_list)

	save_path='./summ_guided_dp_result/%s-rb.pt'%(dataset_list[i_data])
	torch.save(result_all,save_path)




















