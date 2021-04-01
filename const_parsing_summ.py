
from collections import Counter
from random import random
import re
import numpy as np
import torch
import os
import sys


def build_connection_matrix(weight_matrix):
    return torch.triu(weight_matrix)+torch.triu(weight_matrix.transpose(1,0),diagonal=1)

def find_max_idx(connection_matrix):
    max_value=0
    max_idx=-1
    for i in range(connection_matrix.shape[0]-1):
        if connection_matrix[i][i+1]>max_value:
            max_value=connection_matrix[i][i+1]
            max_idx = i
    return max_idx

def find_max_idx_within_range(connection_matrix,sent_start,sent_end):
    max_value=0
    max_idx=-1
    for i in range(sent_start,sent_end):
        if connection_matrix[i][i+1]>max_value:
            max_value=connection_matrix[i][i+1]
            max_idx = i
    return max_idx
    
def rebuild_weight_matrix(weight_matrix,max_idx):
    new_weight_matrix=torch.zeros(weight_matrix.shape[0]-1,weight_matrix.shape[1]-1)
    
    new_weight_matrix[:max_idx,:max_idx] = weight_matrix[:max_idx,:max_idx]
    new_weight_matrix[(max_idx+1):,(max_idx+1):] = weight_matrix[(max_idx+2):,(max_idx+2):]
    new_weight_matrix[:max_idx,(max_idx+1):] = weight_matrix[:max_idx,(max_idx+2):]
    new_weight_matrix[(max_idx+1):,:max_idx] = weight_matrix[(max_idx+2):,:max_idx]
    
    if max_idx!=0:
        new_weight_matrix[max_idx,:max_idx]=weight_matrix[max_idx:(max_idx+2),:max_idx].max(dim=0).values
    if max_idx+2<weight_matrix.shape[0]:
        new_weight_matrix[max_idx,(max_idx+1):]=weight_matrix[max_idx:(max_idx+2),(max_idx+2):].max(dim=0).values
    
    if max_idx!=0:
        new_weight_matrix[:max_idx,max_idx]=weight_matrix[:max_idx,max_idx:(max_idx+2)].max(dim=1).values
    if max_idx+2<weight_matrix.shape[0]:
        new_weight_matrix[(max_idx+1):,max_idx]=weight_matrix[(max_idx+2):,max_idx:(max_idx+2)].max(dim=1).values
    
    new_weight_matrix[max_idx,max_idx]=weight_matrix[max_idx:(max_idx+2),max_idx:(max_idx+2)].sum()/4
    return new_weight_matrix

# def build_const_tree_bottomup_greedy_within_sent(weight_matrix,sent_list):

def generate_new_node(max_idx,weight_matrix,index_mapping):
    result_tree=[]
    important_scores = weight_matrix.sum(0)
    if important_scores[max_idx]>important_scores[max_idx+1]:
        result_tree.append([index_mapping[max_idx],'Neucleus'])
        result_tree.append([index_mapping[max_idx+1],'Satellite'])
    elif important_scores[max_idx]<important_scores[max_idx+1]:
        result_tree.append([index_mapping[max_idx],'Satellite'])
        result_tree.append([index_mapping[max_idx+1],'Neucleus'])
    else:
        result_tree.append([index_mapping[max_idx],'Neucleus'])
        result_tree.append([index_mapping[max_idx+1],'Neucleus'])
    return result_tree

def build_const_tree_bottomup_greedy(weight_matrix,edu_to_sent_mapping):
    sent_to_edu={}
    for edu,sent in enumerate(edu_to_sent_mapping):
        sent_to_edu[sent]=sent_to_edu.get(sent,[])+[edu]
    print(sent_to_edu)
    sent_num=max(sent_to_edu.keys())
    sent_length = [len(sent_to_edu[i])-1 for i in range(sent_num)]
    connection_matrix = build_connection_matrix(weight_matrix)
    index_mapping = {n:(n,n) for n in range(weight_matrix.shape[0])}
    result_tree=[]
    for sent_idx in range(sent_num+1):
#         sent_list = sent_to_edu[sent_idx]
#         print(sent_idx)
        sent_start,sent_end = sent_idx,sent_idx+sent_length[sent_idx]

        while sent_end>sent_start:
            max_idx = find_max_idx_within_range(connection_matrix,sent_start,sent_end)
#             print(max_idx)
            result_tree.extend(generate_new_node(max_idx,weight_matrix,index_mapping))
            weight_matrix = rebuild_weight_matrix(weight_matrix,max_idx)
            connection_matrix = build_connection_matrix(weight_matrix)
            index_mapping[max_idx] = (index_mapping[max_idx][0],index_mapping[max_idx+1][1])
            for i in range(max_idx+1,len(index_mapping.keys())-1):
                index_mapping[i]=index_mapping[i+1]
            del index_mapping[len(index_mapping.keys())-1]
#             print(index_mapping)
            sent_end-=1
#         print(result_tree)
    while connection_matrix.shape[0]>=2:
        max_idx = find_max_idx(connection_matrix)
#         print(max_idx)
        result_tree.extend(generate_new_node(max_idx,weight_matrix,index_mapping))
        weight_matrix = rebuild_weight_matrix(weight_matrix,max_idx)
        connection_matrix = build_connection_matrix(weight_matrix)
        index_mapping[max_idx] = (index_mapping[max_idx][0],index_mapping[max_idx+1][1])
        for i in range(max_idx+1,len(index_mapping.keys())-1):
            index_mapping[i]=index_mapping[i+1]
        del index_mapping[len(index_mapping.keys())-1]
    result_tree.append([index_mapping[0],'Root'])
    return result_tree

def right_branching(length):
    return right_branching_recursion(0,length-1)

def right_branching_recursion(start,end):
    if start==end:
        return [[[start,end],'Satellite']]
    else:
        return [[[start,start],'Satellite']]+right_branching_recursion(start+1,end)+[[[start,end],'Satellite']]
    
def right_branching_recursion_sent(end,subtrees):
    if len(subtrees)==1:
        return subtrees[0]
    else:
        return subtrees[0]+right_branching_recursion_sent(end,subtrees[1:])+[[[subtrees[0][-1][0][0],end],'Satellite']]

def right_branching_with_sent_constraint(length,edu_to_sent_mapping):
    sent_to_edu={}
    for edu,sent in enumerate(edu_to_sent_mapping):
        sent_to_edu[sent]=sent_to_edu.get(sent,[])+[edu]
    sent_num=max(sent_to_edu.keys())
    sent_subtrees=[]
    for s in range(sent_num+1):
        sent_subtrees.append(right_branching_recursion(sent_to_edu[s][0],sent_to_edu[s][-1]))
    return right_branching_recursion_sent(length-1,sent_subtrees)

def right_branching_with_sent_para_constraint(length,edu_to_sent_mapping,sent_to_para_mapping):
    sent_to_edu={}
    for edu,sent in enumerate(edu_to_sent_mapping):
        sent_to_edu[sent]=sent_to_edu.get(sent,[])+[edu]
    para_to_sent={}
    for sent,para in enumerate(sent_to_para_mapping):
        para_to_sent[para]=para_to_sent.get(para,[])+[sent]
        
    sent_num=max(sent_to_edu.keys())
    sent_subtrees=[]
    for s in range(sent_num+1):
        sent_subtrees.append(right_branching_recursion(sent_to_edu[s][0],sent_to_edu[s][-1]))
        
    para_num=max(para_to_sent.keys())    
    para_subtrees=[]
    for p in range(para_num+1):
        start=para_to_sent[p][0]
        end = para_to_sent[p][-1]
        para_subtrees.append(right_branching_recursion_sent(sent_to_edu[end][-1],sent_subtrees[start:end+1]))
    return right_branching_recursion_sent(length-1,para_subtrees)

def cky_local_global_max(weight_matrix):
    importance_vec = weight_matrix.sum(0)
    length = weight_matrix.shape[0]
#     importance_vec = weight_matrix.sum(0)
    score_matrix = torch.zeros(weight_matrix.shape)
    connection_matrix = build_connection_matrix(weight_matrix)
    record = {}
    # Initialize the diagonal - base cases.
    for i in range(weight_matrix.shape[0]):
        record[i]={i:{}}
#         record[i][i]['index_maaping'] = {n:(n,n) for n in range(length)}
#         adjacent_connection = torch.diagonal(weight_matrix,-1)+torch.diagonal(weight_matrix,1)
#         record[i][i]['adjacent_connection'] = adjacent_connection
        record[i][i]['path'] = []
        record[i][i]['span'] = [i,i]
        score_matrix[range(score_matrix.shape[0]),range(score_matrix.shape[1])]= importance_vec

    # Build the matrix recursively.
    for l in range(1,length):
        for i in range(length-l):
            record[i][i+l]={}
            highest_split = -1
            highest_score = 0
            for j in range(1,l+1):
                left = record[i][i+j-1]
                right = record[i+j][i+l]
                left_to_right = weight_matrix[right['span'][0]:right['span'][1]+1,left['span'][0]:left['span'][1]+1].mean()
                right_to_left = weight_matrix[left['span'][0]:left['span'][1]+1,right['span'][0]:right['span'][1]+1].mean()
#                 new_edge = connection_matrix[left['span'][0]:left['span'][1]+1,right['span'][0]:right['span'][1]+1].max()
                new_edge = left_to_right+right_to_left
                score = (score_matrix[i][i+j-1]+score_matrix[i+j][i+l]+new_edge)/2
                if score>=highest_score:
                    highest_score = score
                    highest_split = j
            j=highest_split
            score_matrix[i][i+l]=highest_score
            left = record[i][i+j-1]
            right = record[i+j][i+l]
             ### or mean?
            left_to_right = weight_matrix[right['span'][0]:right['span'][1]+1,left['span'][0]:left['span'][1]+1].max()
            right_to_left = weight_matrix[left['span'][0]:left['span'][1]+1,right['span'][0]:right['span'][1]+1].max()
            if left_to_right>right_to_left:             
                record[i][i+l]['path'] = left['path']+ [[left['span'],'Neucleus']]+right['path'] + [[right['span'],'Satellite']]
            else:             
                record[i][i+l]['path'] = left['path']+ [[left['span'],'Satellite']]+right['path'] + [[right['span'],'Neucleus']]
            record[i][i+l]['span'] = [left['span'][0],right['span'][1]] 
    final_path = record[0][length-1]['path']+[[[0,length-1],'Root']]
    return final_path            

def cky_local_global_max_with_sent_constraint(weight_matrix,edu_to_sent_mapping):
    sent_to_edu={}
    for edu,sent in enumerate(edu_to_sent_mapping):
        sent_to_edu[sent]=sent_to_edu.get(sent,[])+[edu]
    importance_vec = weight_matrix.sum(0)
    length = weight_matrix.shape[0]
#     importance_vec = weight_matrix.sum(0)
    score_matrix = torch.zeros(weight_matrix.shape)
    # connection_matrix = build_connection_matrix(weight_matrix)
    record = {}
    # Initialize the diagonal - base cases.
    for i in range(weight_matrix.shape[0]):
        record[i]={i:{}}
#         record[i][i]['index_maaping'] = {n:(n,n) for n in range(length)}
#         adjacent_connection = torch.diagonal(weight_matrix,-1)+torch.diagonal(weight_matrix,1)
#         record[i][i]['adjacent_connection'] = adjacent_connection
        record[i][i]['path'] = []
        record[i][i]['span'] = [i,i]
        score_matrix[range(score_matrix.shape[0]),range(score_matrix.shape[1])]= importance_vec

    # Build the matrix recursively.
    for l in range(1,length):
        for i in range(length-l):
            record[i][i+l]={}
            highest_split = -1
            highest_score = 0
            for j in range(1,l+1):
                if (not record[i].get(i+j-1,None)) or (not record[i+j].get(i+l,None)):
                    continue
                left = record[i][i+j-1]
                right = record[i+j][i+l]
                
                left_most_edu = left['span'][0]
                right_most_edu = right['span'][1]
                ### if the new constituent cross different sentences
                if edu_to_sent_mapping[left_most_edu]!=edu_to_sent_mapping[right_most_edu]:
                    ### not valid if the left-most sentence not complete
                    if sent_to_edu[edu_to_sent_mapping[left_most_edu]][0]!=left_most_edu:
                        continue
                    ### not valid if the right-most sentence not complete
                    if sent_to_edu[edu_to_sent_mapping[right_most_edu]][-1]!=right_most_edu:
                        continue

                left_to_right = weight_matrix[right['span'][0]:right['span'][1]+1,left['span'][0]:left['span'][1]+1].mean()
                right_to_left = weight_matrix[left['span'][0]:left['span'][1]+1,right['span'][0]:right['span'][1]+1].mean()
#                 new_edge = connection_matrix[left['span'][0]:left['span'][1]+1,right['span'][0]:right['span'][1]+1].max()
                new_edge = left_to_right+right_to_left
                score = (score_matrix[i][i+j-1]+score_matrix[i+j][i+l]+new_edge)/2
                if score>=highest_score:
                    highest_score = score
                    highest_split = j
            if highest_split==-1:
                continue
            j=highest_split
            score_matrix[i][i+l]=highest_score
            left = record[i][i+j-1]
            right = record[i+j][i+l]
             ### or mean?
            left_to_right = weight_matrix[right['span'][0]:right['span'][1]+1,left['span'][0]:left['span'][1]+1].max()
            right_to_left = weight_matrix[left['span'][0]:left['span'][1]+1,right['span'][0]:right['span'][1]+1].max()
            if left_to_right>right_to_left:             
                record[i][i+l]['path'] = left['path']+ [[left['span'],'Neucleus']]+right['path'] + [[right['span'],'Satellite']]
            else:             
                record[i][i+l]['path'] = left['path']+ [[left['span'],'Satellite']]+right['path'] + [[right['span'],'Neucleus']]
            record[i][i+l]['span'] = [left['span'][0],right['span'][1]] 
    final_path = record[0][length-1]['path']+[[[0,length-1],'Root']]
    return final_path   


def cky_local_global_max_with_sent_para_cons(weight_matrix,edu_to_sent_mapping,sent_to_para_mapping,scores=None):
    sent_to_edu={}
    for edu,sent in enumerate(edu_to_sent_mapping):
        sent_to_edu[sent]=sent_to_edu.get(sent,[])+[edu]

    para_to_sent={}
    for sent,para in enumerate(sent_to_para_mapping):
        para_to_sent[para]=para_to_sent.get(para,[])+[sent]

    importance_vec = weight_matrix.sum(0)

    length = weight_matrix.shape[0]
    score_matrix = torch.zeros(weight_matrix.shape)


#     score_matrix = torch.eye(weight_matrix.shape[0])
    # connection_matrix = build_connection_matrix(weight_matrix)
    record = {}
    # Initialize the diagonal - base cases.

    for i in range(weight_matrix.shape[0]):
        record[i]={i:{}}
#         record[i][i]['index_maaping'] = {n:(n,n) for n in range(length)}
#         adjacent_connection = torch.diagonal(weight_matrix,-1)+torch.diagonal(weight_matrix,1)
#         record[i][i]['adjacent_connection'] = adjacent_connection
        record[i][i]['path'] = []
        record[i][i]['span'] = [i,i]
        score_matrix[range(score_matrix.shape[0]),range(score_matrix.shape[1])]= torch.from_numpy(importance_vec)

    # Build the matrix recursively.
    for l in range(1,length):
        for i in range(length-l):
            record[i][i+l]={}
            highest_split = -1
            highest_score = 0
            for j in range(1,l+1):
                if (not record[i].get(i+j-1,None)) or (not record[i+j].get(i+l,None)):
                    continue
                left = record[i][i+j-1]
                right = record[i+j][i+l]

                left_most_edu = left['span'][0]
                right_most_edu = right['span'][1]
                ### if the new constituent cross different sentences
                if edu_to_sent_mapping[left_most_edu]!=edu_to_sent_mapping[right_most_edu]:
                    ### not valid if the left-most sentence not complete
                    if sent_to_edu[edu_to_sent_mapping[left_most_edu]][0]!=left_most_edu:
                        continue
                    ### not valid if the right-most sentence not complete
                    if sent_to_edu[edu_to_sent_mapping[right_most_edu]][-1]!=right_most_edu:
                        continue
                        
                left_most_sent = edu_to_sent_mapping[left['span'][0]]
                right_most_sent = edu_to_sent_mapping[right['span'][1]]
                ### if the new constituent cross different sentences
                if sent_to_para_mapping[left_most_sent]!=sent_to_para_mapping[right_most_sent]:
                    ### not valid if the left-most sentence not complete
                    if para_to_sent[sent_to_para_mapping[left_most_sent]][0]!=left_most_sent:
                        continue
                    ### not valid if the right-most sentence not complete
                    if para_to_sent[sent_to_para_mapping[right_most_sent]][-1]!=right_most_sent:
                        continue         
                
                left_to_right = weight_matrix[right['span'][0]:right['span'][1]+1,left['span'][0]:left['span'][1]+1].mean()
                right_to_left = weight_matrix[left['span'][0]:left['span'][1]+1,right['span'][0]:right['span'][1]+1].mean()
#                 new_edge = connection_matrix[left['span'][0]:left['span'][1]+1,right['span'][0]:right['span'][1]+1].max()
                new_edge = left_to_right+right_to_left
                score = (score_matrix[i][i+j-1]+score_matrix[i+j][i+l]+new_edge)/2
#                 new_edge = left_to_right+right_to_left
                # score = score_matrix[i][i+j-1]*score_matrix[i+j][i+l]*new_edge
                if score>=highest_score:
                    highest_score = score
                    highest_split = j

            if highest_split==-1:
                continue
            j=highest_split
            score_matrix[i][i+l]=highest_score
            left = record[i][i+j-1]
            right = record[i+j][i+l]
             ### or mean?
            left_to_right = weight_matrix[right['span'][0]:right['span'][1]+1,left['span'][0]:left['span'][1]+1].max()
            right_to_left = weight_matrix[left['span'][0]:left['span'][1]+1,right['span'][0]:right['span'][1]+1].max()
#             left_to_right = importance_vec[left['span'][0]:left['span'][1]+1].mean()
#             right_to_left = importance_vec[right['span'][0]:right['span'][1]+1].mean()
#             if abs(left_to_right-right_to_left)<(0.5/weight_matrix.shape[0]):
#                 record[i][i+l]['path'] = left['path']+ [[left['span'],'Nucleus']]+right['path'] + [[right['span'],'Nucleus']]
            if left_to_right>right_to_left:             
                record[i][i+l]['path'] = left['path']+ [[left['span'],'Nucleus']]+right['path'] + [[right['span'],'Satellite']]
            else:             
                record[i][i+l]['path'] = left['path']+ [[left['span'],'Satellite']]+right['path'] + [[right['span'],'Nucleus']]
            if scores is not None:
                left_to_right = scores[left['span'][0]:left['span'][1]+1].max()
                right_to_left = scores[right['span'][0]:right['span'][1]+1].max()
                if left_to_right>right_to_left:             
                    record[i][i+l]['path'] = left['path']+ [[left['span'],'Nucleus']]+right['path'] + [[right['span'],'Satellite']]
                else:             
                    record[i][i+l]['path'] = left['path']+ [[left['span'],'Satellite']]+right['path'] + [[right['span'],'Nucleus']]
            record[i][i+l]['span'] = [left['span'][0],right['span'][1]] 
    final_path = record[0][length-1]['path']+[[[0,length-1],'Root']]
    return final_path 

