from random import random
import pandas as pd
import re
import numpy as np
import os
import sys




def revise_instruction(file):
    with open(file,'r') as of:
        text = of.read()
    all_wrong_leaf = re.findall(r'Root \(leaf (.*?)\)', text)
    if len(all_wrong_leaf)!=0:
        for l in all_wrong_leaf:
            text = text.replace('Root (leaf %s)'%(l),'Root (span %s %s)'%(l,l))
    text=text.split('\n')
    return text


def convert_gt(gt_list):
    result = []
    for s in gt_list:
        s = s.strip()
        m=re.search('\( (?P<ns>\S+) \((?P<cover>((span \d+ \d+)|(leaf \d+)))\)',s)
        if m!=None:
            properties = m.group('cover').split()
            if properties[0]=='leaf':
                cur = [[int(properties[1])-1,int(properties[1])-1],m.group('ns')]
            else:
                cur = [[int(properties[1])-1,int(properties[2])-1],m.group('ns')]
            result.append(cur)
    root_num = count_root(result)
    if root_num>1:
        for i in range(len(result)):
            if result[i][1]=='Root':
                result[i][1]='Nucleus'
        result = [[[0,result[-1][0][1]],'Root']] +result
    return result

def count_root(node_list):
    root_num=0
    for n in node_list:
        if n[1]=='Root':
            root_num+=1
    return root_num

class Node(object):
    def __init__(self, idx, nuclearity,start,end):
        self.idx = idx
        self.nuclearity = nuclearity
        self.parent = None
        self.num_children = 0
        self.children = [None, None]
        self.positional_encoding = None
        self.branch = None
        self.start_idx = int(start)
        self.end_idx = int(end)

    def add_child(self,child, branch):
        child.parent = self
        self.num_children += 1
        child.branch = branch
        self.children[branch] = child

    def set_height(self,height):
        self.height = height

    def set_level(self,level):
        self.level = level


class Dep_Node(object):
    def __init__(self, idx):
        self.idx = idx
    #         self.text = text
        self.sentence = None
        self.parent = None
        self.positional_encoding = None
        self.num_children = 0
        self.children = []

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def remove_child(self, child):
        self.num_children -= 1
        self.children.remove(child)
        
def build_tree(json_data):
    nodes_stack = []
    for node in json_data:
        node_index_tupel = node[0]
        node_index_str = str(node[0])
        print(node_index_str)
        node_nuclearity = node[1]
        # Check if node is a leaf
        if node_index_tupel[0] == node_index_tupel[1]:
            nodes_stack.append(Node(idx = node_index_str, nuclearity = node_nuclearity,start=node_index_tupel[0],end=node_index_tupel[1]))
        # Add children for internal nodes
        else:
            print('here')
            tmp_node = Node(idx = node_index_str, nuclearity = node_nuclearity,start=node_index_tupel[0],end=node_index_tupel[1])
            tmp_node.add_child(nodes_stack.pop(), branch = 1)
            tmp_node.add_child(nodes_stack.pop(), branch = 0)
            nodes_stack.append(tmp_node)
    root_node = Node(idx = 'root', nuclearity = None, start=nodes_stack[0].start_idx,end=nodes_stack[1].end_idx)
    root_node.branch = 0
    root_node.add_child(nodes_stack.pop(), branch = 1)
    root_node.add_child(nodes_stack.pop(), branch = 0)
    return root_node


def const_to_dep_tree(node):
    set_head(node)
    const_leaves = get_leaf_nodes(node)
    dep_nodes = [Dep_Node(l.start_idx) for l in const_leaves]
    for leaf,dep_node in zip(const_leaves,dep_nodes):
        closest_S_ancestor, is_tree_head = find_S_ancestor(leaf)
        if is_tree_head:
            root = dep_node
        else:
            dep_node.parent=dep_nodes[closest_S_ancestor.head]
            dep_nodes[closest_S_ancestor.head].add_child(dep_node)
    for dep_node in dep_nodes:
        # print('current id:%d'%(dep_node.idx))
        child_id = []
        for child in dep_node.children:
            child_id.append(child.idx)
        # print(child_id)
    return dep_nodes

def const_to_dep_tree_li14(node):
    const_leaves = get_leaf_nodes(node)
    dep_tree={}
    for leaf in const_leaves:
        # print(leaf.start_idx)
        # print(leaf.end_idx)
        p = find_my_top_node(leaf)
        if p.nuclearity=='Root':
            root = leaf.start_idx
        else:
            
            head = find_head_edu(p.parent)

            if head not in dep_tree:
                dep_tree[head]=[]
            dep_tree[head].append(leaf.start_idx)
#     print(dep_tree)
    return dep_tree,root

def find_my_top_node(e):
    C=e
    p = C.parent
    nucleus_child = [p.children[i] for i in range(len(p.children)) if p.children[i].nuclearity=='Nucleus']
    while nucleus_child[0]==C and not p.nuclearity=='Root':
        C = p
        p = C.parent
        # print(p.idx)
        nucleus_child = [p.children[i] for i in range(len(p.children)) if p.children[i].nuclearity=='Nucleus']
    if p.nuclearity=='Root'and nucleus_child[0]==C:
        C=p
    return C

def find_head_edu(p):

    while p.num_children!=0:
        nucleus_child = [p.children[i] for i in range(len(p.children)) if p.children[i].nuclearity=='Nucleus']
        p = nucleus_child[0]

    return p.start_idx

def print_tree_preorder(root):
    if root==None:
        return
    print(root.start_idx,root.end_idx, root.nuclearity)
    for child in root.children:
        print_tree_preorder(child)
        
def print_tree_postorder(root):
    if root==None:
        return
    for child in root.children:
        print_tree_postorder(child)
    print(root.idx, root.start_idx,root.end_idx, root.nuclearity)

def return_tree_postorder(root):
    if root==None:
        return []
    l=[]
    for child in root.children:
        l.extend(return_tree_postorder(child))
    l+=[[[root.start_idx,root.end_idx], root.nuclearity]]
    return l


def set_height_bottomup(node):
    if not node:
        return -1
    else:
        h0 = set_height_bottomup(node.children[0])
        h1 = set_height_bottomup(node.children[1])
        current_height = max(h0,h1)+1
        node.set_height(current_height)
        return current_height

def set_level_topdown(node,cur_level):
    if not node:
        return -1
    else:
        node.set_level(cur_level)
        l0=set_level_topdown(node.children[0],cur_level+1)
        l1=set_level_topdown(node.children[1],cur_level+1)
        return max(l0,l1,cur_level)


def build_attention_map_bottomup(node,att_map):
    if not node:
        return att_map
    h = att_map.shape[0]-1
    if node.nuclearity=='Nucleus':
        att_map[h-node.level,node.start_idx:node.end_idx,node.start_idx:node.end_idx]=2
    else:
        att_map[h-node.level,node.start_idx:node.end_idx,node.start_idx:node.end_idx]=1
    attn_map = build_attention_map_bottomup(node.children[0],att_map)
    attn_map = build_attention_map_bottomup(node.children[1],att_map)
    return att_map

def get_importance_score(node,n=0,s=0):
    if node.nuclearity=='Satellite':
        s+=1
    else:
        n+=1
    if node.num_children==0:
        result_dict={}
        result_dict[node.start_idx] = (n/(n+s))
        return result_dict
    else:
        result_dict = {}
        if node.children[0]:
            result_dict.update(get_importance_score(node.children[0],n,s))
        if node.children[1]:
            result_dict.update(get_importance_score(node.children[1],n,s))
        return result_dict


# Generate dependency tree from constituency tree
def map_disco_to_sent(disco_span):
    map_to_sent = [0 for _ in range(len(disco_span))]
    curret_sent = 0
    current_idx = 1
    for idx, disco in enumerate(disco_span):
        if disco[0] == current_idx:
            map_to_sent[idx] = curret_sent
        else:
            curret_sent += 1
            map_to_sent[idx] = curret_sent
        current_idx = disco[1]
    return map_to_sent

def set_head(node):
    if node.num_children==0:
        node.head=node.start_idx
    else:
        for child in node.children:
            set_head(child)
        if node.children[0].nuclearity!='Satellite':
            node.head = node.children[0].head
        else:
            node.head = node.children[1].head

def const_to_dep_tree(node):
    set_head(node)
    const_leaves = get_leaf_nodes(node)
    dep_nodes = [Dep_Node(l.start_idx) for l in const_leaves]
    for leaf,dep_node in zip(const_leaves,dep_nodes):
        closest_S_ancestor, is_tree_head = find_S_ancestor(leaf)
        if is_tree_head:
            root = dep_node
        else:
            dep_node.parent=dep_nodes[closest_S_ancestor.head]
            dep_nodes[closest_S_ancestor.head].add_child(dep_node)
    for dep_node in dep_nodes:
        # print('current id:%d'%(dep_node.idx))
        child_id = []
        for child in dep_node.children:
            child_id.append(child.idx)
        # print(child_id)
    return dep_nodes

def find_S_ancestor(const_node):
    if const_node.nuclearity == None and const_node.parent == None:
        return const_node, True
    if const_node.nuclearity == 'Nucleus' and not const_node.parent == None:
        closest_S_ancestor, is_tree_head = find_S_ancestor(const_node.parent)
    else:
        closest_S_ancestor = const_node.parent
        is_tree_head = False
    return closest_S_ancestor, is_tree_head


def get_leaf_nodes(node):
    if node.num_children == 0:
        return [node]
    else:
        leaves = []
        for child_node in node.children:
            leaves.extend(get_leaf_nodes(child_node))
        return leaves
    
def construct_preorder(node_list,start,end,num_nonbinary=0):
    if start>end:
        return

    node = node_list[start]
    
    node_index_tupel = node[0]
    node_index_str = str(node[0])
    
    node_nuclearity = node[1]
    tmp_node = Node(idx = node_index_str, nuclearity = node_nuclearity,start=node_index_tupel[0],end=node_index_tupel[1])
    if start==end:
        return tmp_node,num_nonbinary
    i=start+1
    splits = []
    splits.append(i)
    while i<=end:
        if node_list[i][0][0]>node_list[splits[-1]][0][1]:
            splits.append(i)
            if node_index_tupel[1]==node_list[i][0][1]:
                break
        i+=1
    end_idx = node_index_tupel[1]
    splits.append(end)   
    # print(splits)
    right,num_nonbinary=construct_preorder(node_list,splits[-2],splits[-1],num_nonbinary)
    if len(splits)>3:
        while len(splits)>3:
            num_nonbinary +=1
#             node_index_tupel = [node_list[splits[-3]][0][0],node_list[splits[-1]][0][1]]
            node_index_tupel = [node_list[splits[-3]][0][0],end_idx]
            node_index_str = str(node_index_tupel)
            if 'Nucleus' in [node_list[splits[-3]][1],node_list[splits[-2]][1]]:
                node_nuclearity = 'Nucleus'
            else:
                node_nuclearity = 'Satellite'
            # print(node_index_tupel)
            tmp_tmp_node = Node(idx = node_index_str, nuclearity = node_nuclearity,start=node_index_tupel[0],end=node_index_tupel[1])
            
            left,num_nonbinary = construct_preorder(node_list,splits[-3],splits[-2]-1,num_nonbinary)
            if left!=None:
                tmp_tmp_node.add_child(left,branch = 0)
            if right!=None:
                tmp_tmp_node.add_child(right,branch = 1)
            right = tmp_tmp_node
            del splits[-1]            
    left,num_nonbinary = construct_preorder(node_list,splits[-3],splits[-2]-1,num_nonbinary)
    if left!=None:
        tmp_node.add_child(left,branch = 0)
    if right!=None:
        tmp_node.add_child(right,branch = 1)
    return tmp_node,num_nonbinary
