

def evaluate_const_tree(const_tree_gt,const_tree_cand):
	gt_list = [(node[0][0],node[0][1]) for node in const_tree_gt]
	cand_list = [(node[0][0],node[0][1]) for node in const_tree_cand]
	num_match = len(set(gt_list).intersection(set(cand_list)))
	total = len(gt_list)
	assert len(gt_list)==len(cand_list)
	return num_match/total

def evaluate_const_tree_micro(const_tree_gt,const_tree_cand):
	gt_list = [(node[0][0],node[0][1]) for node in const_tree_gt]
	cand_list = [(node[0][0],node[0][1]) for node in const_tree_cand]
	num_match = len(set(gt_list).intersection(set(cand_list)))
	total = len(gt_list)
	assert len(gt_list)==len(cand_list)
	return num_match,total

def evaluate_const_tree_ns(const_tree_gt,const_tree_cand):
	gt_list = [(node[0][0],node[0][1],node[1]) for node in const_tree_gt]
#     print(gt_list)
	cand_list = [(node[0][0],node[0][1],node[1]) for node in const_tree_cand]
#     print(cand_list)
#     print(set(cand_list).intersection(set(gt_list)))
	num_match = len(set(cand_list).intersection(set(gt_list)))
#     total = len(gt_list)
#     assert len(gt_list)==len(cand_list)
	gt_total = len(gt_list)
	cand_total = len(cand_list)
	return num_match,gt_total,cand_total

def evaluate_const_tree_micro_nonbinarized(const_tree_gt,const_tree_cand):
	gt_list = [(node[0][0],node[0][1]) for node in const_tree_gt]
	cand_list = [(node[0][0],node[0][1]) for node in const_tree_cand]
	num_match = len(set(gt_list).intersection(set(cand_list)))
	gt_total = len(gt_list)
	cand_total = len(cand_list)
	return num_match,gt_total,cand_total



def evaluate_const_tree_list(const_tree_gt_list,const_tree_cand_list,silent=False):
	total_macro = 0
	total_micro = 0
	correct_micro =0
	total_ns= 0
	total_cand=0
	total_gt=0
	all_acc=[]
	for i in range(len(const_tree_gt_list)):
	#     if 'news' not in ids[i]:
	#         continue
		c,g,cand=evaluate_const_tree_micro_nonbinarized(const_tree_gt_list[i],const_tree_cand_list[i])
		total_macro+=(c/g)
		correct_micro+=c
		total_cand+=cand
		total_gt+=g
	p = correct_micro/total_cand
	r = correct_micro/total_gt
	micro_f1 = 2*(p*r)/(p+r)
	macro_f1 = total_macro/len(const_tree_gt_list)
	micro_f1 *=100
	macro_f1 *=100
	if not silent:
		print(correct_micro)
		print(total_gt)
		print(total_cand)
		print('micro f1: ',micro_f1)
		print('macro f1: ',macro_f1)
	return micro_f1,macro_f1


def unlabeled_attachment_score(dep_tree_gt,dep_tree_cand):
	total=0
	correct=0
	for key in dep_tree_gt.keys():
		if key in dep_tree_cand.keys():
			correct += len(set(dep_tree_gt[key]).intersection(set(dep_tree_cand[key])))
		total+=len(dep_tree_gt[key])
	return correct,total

def evaluate_dep_tree_list(dep_tree_gt_list,dep_tree_cand_list,silent=False):
	correct_all=0
	total_all=0
	total_macro=0
	for i in range(len(dep_tree_gt_list)):
		correct,total=unlabeled_attachment_score(dep_tree_gt_list[i],dep_tree_cand_list[i])
		correct_all+=correct
		total_all+=total
		total_macro+=(correct/total)
	micro_f1 = correct_all/total_all
	macro_f1 = total_macro/len(dep_tree_gt_list)
	micro_f1 *=100
	macro_f1 *=100
	if not silent:
		print('correct num: ',correct_all)
		print('total_num: ',total_all)
		print('micro f1: ',micro_f1)
		print('macro f1: ', macro_f1)
	return micro_f1,macro_f1

