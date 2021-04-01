from random import random
import re
import numpy as np
import torch
import os
import sys



def reverse_graph(G):
	'''Return the reversed graph where g[dst][src]=G[src][dst]'''
	g={}
	for src in G.keys():
		for dst in G[src].keys():
			if dst not in g.keys():
				g[dst]={}
			g[dst][src]=G[src][dst]
	return g

def build_max(rg,root):
	'''Find the max in-edge for every node except for the root.'''
	mg = {}
	for dst in rg.keys():
		if dst==root:
			continue
		max_ind=-100
		max_value = -100
		for src in rg[dst].keys():
			if rg[dst][src]>=max_value:
				max_ind = src
				max_value = rg[dst][src]
		mg[dst]={max_ind:max_value}
	return mg

def find_circle(mg):
	'''Return the firse circle if find, otherwise return None'''
		
	for start in mg.keys():
		visited=[]
		stack = [start]
		while stack:
			n = stack.pop()
			if n in visited:
				C = []
				while n not in C:
					C.append(n)
					n = list(mg[n].keys())[0]
				return C
			visited.append(n)
			if n in mg.keys():
				stack.extend(list(mg[n].keys()))
	return None
		
def chu_liu_edmond(G,root):
	''' G: dict of dict of weights
			G[i][j] = w means the edge from node i to node j has weight w.
		root: the root node, have outgoing edges only.
	'''
	# reversed graph rg[dst][src] = G[src][dst]
	G=remove_self_edge(G)
	rg = reverse_graph(G)
	# root only has out edge
	rg[root]={}
	# the maximum edge for each node other than root
	mg = build_max(rg,root)
	
	# check if mg is a tree (contains a circle)
	C = find_circle(mg)
	# if there is no circle, it means mg is what we want
	if not C:
		return reverse_graph(mg)
	# Now consider the nodes in the circle C as one new node vc
	all_nodes = G.keys()
	vc = max(all_nodes)+1
	
	#The new graph G_prime with V_prime=V\C+{vc} 
	V_prime = list(set(all_nodes)-set(C))+[vc]
	G_prime = {}
	vc_in_idx={}
	vc_out_idx={}
	# Now add the edges to G_prime
	for u in all_nodes:
		for v in G[u].keys():
			# First case: if the source is not in the circle, and the dest is in the circle, i.e. in-edges for C
			# Then we only keep one edge from each node that is not in C to the new node vc with the largest difference (G[u][v]-list(mg[v].values())[0])
			# To specify, for each node u in V\C, there is an edge between u and vc if and only if there is an edge between u and any node v in C,
			# And the weight of edge u->vc = max_{v in C} (G[u][v] - mg[v].values) The second term represents the weight of max in-edge of v.
			# Then we record that the edge u->vc is originally the edge u->v with v=argmax_{v in C} (G[u][v] - mg[v].values)
			
			if (u not in C) and (v in C):
				if u not in G_prime.keys():
					G_prime[u]={}
				w = G[u][v]-list(mg[v].values())[0]
				if (vc not in  G_prime[u]) or (vc in  G_prime[u] and w > G_prime[u][vc]):
					G_prime[u][vc] = w
					vc_in_idx[u] = v
			# Second case: if the source is in the circle, but the dest is not in the circle, i.e out-edge for C
			# Then we only keep one edge from the new node vc to each node that is not in C
			# To specify, for each node v in V\C, there is an edge between vc and v iff there is an edge between any edge u in C and v.
			# And the weight of edge vc->v = max_{u in C} G[u][v] 
			# Then we record that the edge vc->v originally the edge u->v with u=argmax_{u in C} G[u][v] 
			elif (u in C) and (v not in C):
				if vc not in G_prime.keys():
					G_prime[vc]={}
				w = G[u][v]
				if (v not in  G_prime[vc]) or (v in  G_prime[vc] and w > G_prime[vc][v]):
					G_prime[vc][v] = w
					vc_out_idx[v] = u
			# Third case: if the source and dest are all not in the circle, then just add the edge to the new graph.
			elif (u not in C) and (v not in C):
				if u not in G_prime.keys():
					G_prime[u]={}
				G_prime[u][v] = G[u][v]
	# Recursively run the algorihtm on the new graph G_prime
	# The result A should be a tree with nodes V\C+vc, then we just need to break the circle C and plug the subtree into A
	# To break the circle, we need to use the in-edge of vc, say u->vc to replace the original selected edge u->v, 
	# where v was the original edge we recorded in the first case above.
	# Then if vc has out-edges, we also need to replace them with the original edges, recorded in the second case above.
	A = chu_liu_edmond(G_prime,root)

	all_nodes_A = list(A.keys())
	for src in all_nodes_A:
		# The number of out-edges varies, could be 0 or any number <=|V\C|
		if src==vc:
			for node_in in A[src].keys():
				orig_out = vc_out_idx[node_in]
				if orig_out not in A.keys():
					A[orig_out] = {}
				A[orig_out][node_in]=G[orig_out][node_in]
			del A[vc]
		else:
			for dst in A[src]:
				# There must be only one in-edge to vc.
				if dst==vc:
					orig_in = vc_in_idx[src]
					A[src][orig_in] = G[src][orig_in]
					del A[src][dst]
	
	# Now add the edges from the circle to the result.
	# Remember not to include the one with new in-edge
	for node in C:
		if node != orig_in:
			src = list(mg[node].keys())[0]
			if src not in A.keys():
				A[src] = {}
			A[src][node] = mg[node][src]
	return A    
def find_max_edges_between_sent_cluster(G,sent_src,sent_dst):
	max_ind_src=-100
	max_ind_dst=-100
	max_value=-100
	for edu_src in sent_src:
		for edu_dst in sent_dst:
			if G[edu_src][edu_dst]>max_value:
				max_ind_src=edu_src
				max_ind_dst=edu_dst
				max_value=G[edu_src][edu_dst]
	return max_ind_src,max_ind_dst,max_value

def remove_self_edge(G):
	for src in G.keys():
		if src in G[src].keys():
			del G[src][src]
	return G

def combine_two_graphs(g1,g2):
	for src in g2.keys():
		if src not in g1.keys():
			g1[src]={}
		for dst in g2[src].keys():
			g1[src][dst]=g2[src][dst]
	return g1



def find_max_edges_between_sent_cluster_hierarchical(G,sent_src,sent_dst):
	max_ind_src=-100
	max_ind_dst=-100
	max_value=-100
	sum_weight=0
	num_edge=0
	for edu_src in sent_src:
		for edu_dst in sent_dst:
			#Assume G is fully connected
			sum_weight+=G[edu_src][edu_dst]
			num_edge+=1
			if G[edu_src][edu_dst]>max_value:
				max_ind_src=edu_src
				max_ind_dst=edu_dst
				max_value=G[edu_src][edu_dst]
	return max_ind_src,max_ind_dst,max_value,sum_weight/num_edge

def chu_liu_edmond_hierarchical_avg(G,root,edu_to_sent_mapping):
	sent_to_edu={}
	for edu,sent in enumerate(edu_to_sent_mapping):
		sent_to_edu[sent]=sent_to_edu.get(sent,[])+[edu]
	sent_V = sent_to_edu.keys()
	sent_root = edu_to_sent_mapping[root]
	sent_G = {}
	sent_edge_mapping = {}
	# Build the sentence graph, and record the original edu source and edu dest for each edge 
	for sent_src in sent_V:
		for sent_dst in sent_V:
			if sent_dst==sent_root:
				continue
			max_ind_src,max_ind_dst,max_value,weight=find_max_edges_between_sent_cluster_hierarchical(G,sent_to_edu[sent_src],sent_to_edu[sent_dst])
			if sent_src not in sent_edge_mapping.keys():
				sent_edge_mapping[sent_src]={}
			if sent_src not in sent_G.keys():
				sent_G[sent_src]={}
			sent_edge_mapping[sent_src][sent_dst] = (max_ind_src,max_ind_dst,max_value)
			sent_G[sent_src][sent_dst] = weight
			
	# Get the sentence tree
	sent_tree=chu_liu_edmond(sent_G,sent_root)
	edu_tree={}
	edu_dst_nodes = {}
	
	# Build the edu tree, start with the cross sentence edges
	for sent_src in sent_tree.keys():
		for sent_dst in sent_tree[sent_src].keys():
			edu_src,edu_dst,weight = sent_edge_mapping[sent_src][sent_dst]
			if edu_src not in edu_tree:
				edu_tree[edu_src]={}
			edu_tree[edu_src][edu_dst] = weight
			edu_dst_nodes[sent_dst]=edu_dst
			
	# Build the edu tree with the in-sentence edges.
	for sent_cluster in sent_to_edu.keys():
		edus = sent_to_edu[sent_cluster]
		if sent_cluster==sent_root:
			cluster_root=root
		else:
			cluster_root = edu_dst_nodes[sent_cluster]
		cluster_G = {}
		for edu_src in edus:
			for edu_dst in edus:
				if edu_dst==cluster_root:
					continue
				if edu_src not in cluster_G.keys():
					cluster_G[edu_src]={}
				cluster_G[edu_src][edu_dst] = G[edu_src][edu_dst]
		cluster_tree=chu_liu_edmond(cluster_G,cluster_root)
		edu_tree = combine_two_graphs(edu_tree,cluster_tree)
	return edu_tree

def find_cluster_root(G,cluster_nodes):
	max_value=-100
	max_idx=-100
	for node in cluster_nodes:
		if node in G.keys():
			all_out_edges = sum([G[node][dst] for dst in G[node].keys()])
			if all_out_edges>max_value:
				max_value=all_out_edges
				max_idx = node
	return max_idx

def compute_edges_between_sents(G,sent_src,sent_dst):
	sum_weight=0
	num_edge=0
	for edu_src in sent_src:
		for edu_dst in sent_dst:
			#Assume G is fully connected
			sum_weight+=G[edu_src][edu_dst]
			num_edge+=1
	return sum_weight/num_edge
	
def chu_liu_edmond_hierarchical_root_first(G,root,edu_to_sent_mapping):
	sent_to_edu={}
	for edu,sent in enumerate(edu_to_sent_mapping):
		sent_to_edu[sent]=sent_to_edu.get(sent,[])+[edu]
	sent_V = sent_to_edu.keys()
	sent_root = edu_to_sent_mapping[root]
	sent_G = {}
	sent_root_mapping = {}
	# Build the sentence graph, and record the original edu source and edu dest for each edge 
	for sent in sent_V:
		sent_root_mapping[sent] = find_cluster_root(G,sent_to_edu[sent])

	for sent_src in sent_V:
		for sent_dst in sent_V:
			if sent_dst==sent_root:
				continue
			weight=compute_edges_between_sents(G,sent_to_edu[sent_src],sent_to_edu[sent_dst])
			if sent_src not in sent_G.keys():
				sent_G[sent_src]={}
			sent_G[sent_src][sent_dst] = weight
			
	# Get the sentence tree
	sent_tree=chu_liu_edmond(sent_G,sent_root)
	edu_tree={}
	edu_dst_nodes = {}
	
	# Build the edu tree, start with the cross sentence edges
	for sent_src in sent_tree.keys():
		for sent_dst in sent_tree[sent_src].keys():
			edu_src=sent_root_mapping[sent_src]
			edu_dst = sent_root_mapping[sent_dst]
			if edu_src not in edu_tree:
				edu_tree[edu_src]={}
			edu_tree[edu_src][edu_dst] = G[edu_src][edu_dst]
			
	# Build the edu tree with the in-sentence edges.
	for sent_cluster in sent_V:
		edus = sent_to_edu[sent_cluster]
		if sent_cluster==sent_root:
			cluster_root=root
		else:
			cluster_root = sent_root_mapping[sent_cluster]
		cluster_G = {}
		for edu_src in edus:
			for edu_dst in edus:
				if edu_dst==cluster_root:
					continue
				if edu_src not in cluster_G.keys():
					cluster_G[edu_src]={}
				cluster_G[edu_src][edu_dst] = G[edu_src][edu_dst]
		cluster_tree=chu_liu_edmond(cluster_G,cluster_root)
		edu_tree = combine_two_graphs(edu_tree,cluster_tree)
	return edu_tree

##### eisner
def Eisner(G):
	num_v = len(G.keys())
	weight_matrix=torch.zeros(num_v,num_v,2,2)
	selection_matrix = torch.zeros(num_v,num_v,2,2)
	for m in range(1,num_v):
		for i in range(num_v-m):
			j=i+m
			##d=0, c=0
			max_score=0
			max_id=-1
			for q in range(i,j):  
				score = weight_matrix[i,q,1,1]+weight_matrix[q+1,j,0,1]+G[j][i]
				if score>max_score:
					max_score=score
					max_id=q
			weight_matrix[i,j,0,0]=max_score
			selection_matrix[i,j,0,0] = max_id
			
			##d=1, c=0
			max_score=0
			max_id=-1
			for q in range(i,j):  
				score = weight_matrix[i,q,1,1]+weight_matrix[q+1,j,0,1]+G[i][j]
				if score>max_score:
					max_score=score
					max_id=q
			weight_matrix[i,j,1,0]=max_score
			selection_matrix[i,j,1,0] = max_id
			
			
			##d=0, c=1
			max_score=0
			max_id=-1
			for q in range(i,j+1):  
				score = weight_matrix[i,q,0,1]+weight_matrix[q,j,0,0]
				if score>max_score:
					max_score=score
					max_id=q
			weight_matrix[i,j,0,1]=max_score
			selection_matrix[i,j,0,1] = max_id
			
			##d=1, c=1
			max_score=0
			max_id=-1
			for q in range(i,j+1):  
				score = weight_matrix[i,q,1,0]+weight_matrix[q,j,1,1]
				if score>max_score:
					max_score=score
					max_id=q
			weight_matrix[i,j,1,1]=max_score
			selection_matrix[i,j,1,1] = max_id
	dep_tree=Traceback(selection_matrix,0,num_v-1,1,1)
	return dep_tree


def check_validation(G,edu_to_sent_mapping,sent_to_edu,i,j):
	valid=True
	if edu_to_sent_mapping[i]!=edu_to_sent_mapping[j]:
		### not valid if the left-most sentence not complete
		if sent_to_edu[edu_to_sent_mapping[i]][0]!=i:
			valid=False
		### not valid if the right-most sentence not complete
		if sent_to_edu[edu_to_sent_mapping[j]][-1]!=j:
			valid=False
	return valid
	
def Eisner_sent_constraint(G,edu_to_sent_mapping):
	sent_to_edu={}
	for edu,sent in enumerate(edu_to_sent_mapping):
		sent_to_edu[sent]=sent_to_edu.get(sent,[])+[edu]

	num_v = len(G.keys())
	weight_matrix=torch.zeros(num_v,num_v,2,2)
	selection_matrix = torch.zeros(num_v,num_v,2,2)
	for m in range(1,num_v):
		for i in range(num_v-m):
			j=i+m
			valid =  check_validation(G,edu_to_sent_mapping,sent_to_edu,i,j)
			if not valid:
				continue
			##d=0, c=0
			max_score=0
			max_id=-1
			for q in range(i,j):  
				valid =  check_validation(G,edu_to_sent_mapping,sent_to_edu,i,q) and check_validation(G,edu_to_sent_mapping,sent_to_edu,q+1,j)
				if not valid:
					continue
				score = weight_matrix[i,q,1,1]+weight_matrix[q+1,j,0,1]+G[j][i]
				if score>max_score:
					max_score=score
					max_id=q
			weight_matrix[i,j,0,0]=max_score
			selection_matrix[i,j,0,0] = max_id
			
			##d=1, c=0
			max_score=0
			max_id=-1
			for q in range(i,j):  
				valid =  check_validation(G,edu_to_sent_mapping,sent_to_edu,i,q) and check_validation(G,edu_to_sent_mapping,sent_to_edu,q+1,j)
				if not valid:
					continue
				score = weight_matrix[i,q,1,1]+weight_matrix[q+1,j,0,1]+G[i][j]

				if score>max_score:
					max_score=score
					max_id=q
			weight_matrix[i,j,1,0]=max_score
			selection_matrix[i,j,1,0] = max_id
			
			
			##d=0, c=1
			max_score=0
			max_id=-1
			for q in range(i,j+1): 
				valid =  check_validation(G,edu_to_sent_mapping,sent_to_edu,i,q) and check_validation(G,edu_to_sent_mapping,sent_to_edu,q,j)
				if not valid:
					continue
				score = weight_matrix[i,q,0,1]+weight_matrix[q,j,0,0]
				if score>max_score:
					max_score=score
					max_id=q
			weight_matrix[i,j,0,1]=max_score
			selection_matrix[i,j,0,1] = max_id
			
			##d=1, c=1
			max_score=0
			max_id=-1
			for q in range(i,j+1):  
				valid =  check_validation(G,edu_to_sent_mapping,sent_to_edu,i,q) and check_validation(G,edu_to_sent_mapping,sent_to_edu,q,j)
				if not valid:
					continue
				score = weight_matrix[i,q,1,0]+weight_matrix[q,j,1,1]
				if score>max_score:
					max_score=score
					max_id=q
			weight_matrix[i,j,1,1]=max_score
			selection_matrix[i,j,1,1] = max_id
	dep_tree=Traceback(selection_matrix,0,num_v-1,1,1)
	return dep_tree  

def Traceback(selection_matrix,i,j,d,c):
	if i==j:
		return {}
	q=int(selection_matrix[i,j,d,c])
	if d==1 and c==1:
		left_result = Traceback(selection_matrix,i,q,1,0)
		right_result= Traceback(selection_matrix,q,j,1,1)
		current_dep=merge_dict(left_result,right_result)
	elif d==0 and c==1:
		left_result = Traceback(selection_matrix,i,q,0,1)
		right_result= Traceback(selection_matrix,q,j,0,0)
		current_dep=merge_dict(left_result,right_result)

	elif d==1 and c==0:
		left_result = Traceback(selection_matrix,i,q,1,1)
		right_result= Traceback(selection_matrix,q+1,j,0,1)
		current_dep=merge_dict(left_result,right_result)
		current_dep=merge_dict(current_dep,{i:[j]})
	elif d==0 and c==0:
		left_result = Traceback(selection_matrix,i,q,1,1)
		right_result= Traceback(selection_matrix,q+1,j,0,1)
		current_dep=merge_dict(left_result,right_result)
		current_dep=merge_dict(current_dep,{j:[i]})

	return current_dep

def merge_dict(dict1,dict2):
	for k in dict2.keys():
		if k in dict1.keys():
			dict1[k].extend(dict2[k])
		else:
			dict1[k] = dict2[k]
	return dict1