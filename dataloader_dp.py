import pandas as pd
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import numpy as np
import torch
import torch.nn.functional as F
import os
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, IterableDataset
import sys

class SummarizationDataset(IterableDataset):
    def __init__(self,inputs_dir,to_shuffle=True,is_test=False, unit='edu', use_bert=False,bert_model=None,attnmap_path=None):
        if isinstance(inputs_dir,list):
            self._input_files = inputs_dir
        else:
            inputs_dir = Path(inputs_dir)
            self._input_files = [path for path in inputs_dir.glob("*.pt")]
        self.shuffle=to_shuffle
        self._input_files = sorted(self._input_files)
        if self.shuffle:
            shuffle(self._input_files)
        self.is_test=is_test
        self.unit = unit
        if use_bert:
            self.bert = bert_model
        self.attnmap_path = attnmap_path
        # self.cur_filenum = 0
        # self._loaddata()


    def _loaddata(self,idx):
        file = self._input_files[idx]
        self.cur_data = torch.load(file)
        if self.shuffle:
            shuffle(self.cur_data)
        if (idx==len(self._input_files)-1) and self.shuffle:
            shuffle(self._input_files)

        # self.cur_filenum+=1
        # self.cur_filenum = self.cur_filenum%len(self._input_files)
    def preprocessing(self,data):
        out = {}
        out['id'] = data['doc_id'].split('.')[0]
        out['src'] = data['src']

        out['d_span'] = data['d_span']
        out['clss'] = np.where(np.array(out['src'])==101)[0].tolist()+[len(data['src'])]
        cur_seg=0
        segs=[]
        for i in range(len(out['clss'])-1):
            segs.extend([cur_seg]*(out['clss'][i+1]-out['clss'][i]))
        out['segs'] = segs

        if(self.is_test):
            out['disco_txt'] = data['disco_txt']
            out['sent_txt'] = data['sent_txt']
            out['disco_dep'] = data['disco_dep']
            out['disco_to_sent'] = self.map_disco_to_sent(out['d_span'])
            out['sent_to_para'] = [0]*len(out['sent_txt'])

            out['tgt_txt'] = '\n'.join(data['tgt_list_str'])

        return out

    def edu_seg_input(self,src,d_span):

        unit_list = [[101]+src[d_span[i][0]:d_span[i][1]]+[102] for i in range(len(d_span))]
        new_src = []
        new_clss=[]
        new_segment=[]
        new_d_span=[]
        for i, unit in enumerate(unit_list):
            if i%2==0:
                new_segment.extend([0]*len(unit))
            else:
                new_segment.extend([1]*len(unit))
            new_clss.append(len(new_src))
            new_src.extend(unit)
            new_d_span.append((new_clss[-1]+1,len(new_src)-1))
        new_clss.append(len(new_src))

        return new_src,new_clss,new_segment,new_d_span

    def map_disco_to_sent(self,disco_span):
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

    def __iter__(self):
        # for i in range(len(self._input_files)):
        if not self.is_test:
            i=0
            while (True):
                self._loaddata(i)
                while len(self.cur_data) !=0:
                    data = self.cur_data.pop()
                    out = self.preprocessing(data)
                    yield out 
                i = (i+1)%(len(self._input_files))

        if self.is_test:
            for i in range(len(self._input_files)):
                self._loaddata(i)
#                 print(self.cur_data)
                while len(self.cur_data) !=0:
                    data = self.cur_data.pop()
                    out = self.preprocessing(data)
                    yield out 

class Batch(object):
    def _pad(self, data, pad_id,width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def _cut(self, data):
        if len(data['src'])>self.max_length:
            # self.max_length = max_length
            data['src'] = data['src'][:self.max_length]
            data['d_span'] = [(d[0],min(d[1],self.max_length)) for d in data['d_span'] if d[0]<self.max_length]
            data['clss'] = [clss for clss in data['clss'] if clss<self.max_length-1] + [self.max_length]
            data['segs'] = data['segs'][:self.max_length]
        return data


    def _build_edu_span(self,d_span_list):
        max_edu = max([len(d_span) for d_span in d_span_list])
        #batch * num_token * num_edu
        edu_span = torch.zeros(self.batch_size,self.max_length,max_edu)
        for i,d_span in enumerate(d_span_list):
            for j, edu in enumerate(d_span):
                edu_span[i,edu[0]:edu[1],j]=1
        return edu_span

    def _build_sent_span(self,clss_list):
        max_sent = max([len(clss) for clss in clss_list])-1
        #batch * num_token * num_edu
        sent_span = torch.zeros(self.batch_size,self.max_length,max_sent)
        for i,clss in enumerate(clss_list):
            for j, sent in enumerate(clss[:-1]):
                sent_span[i,clss[j]+1:clss[j+1],j]=1
        return sent_span


    def __init__(self, batch=None, device=None,  is_test=False, max_length=None, unit='edu'):
        """Create a Batch from a list of examples."""
        if batch is not None:
            self.batch_size = len(batch)
            self.max_length = max([len(x['src']) for x in batch])
            if max_length and (max_length<self.max_length):
                self.max_length=max_length
                batch = [self._cut(d) for d in batch]
            # batch.sort(key=lambda x: len(x['d_span']), reverse=True)
            batch.sort(key=lambda x: len(x['labels']), reverse=True)
            d_span_list = [x['d_span'] for x in batch]
#             clss_list = [x['clss'] for x in batch]
            if unit=='edu':
                edu_span = self._build_edu_span(d_span_list)
            else:
                sent_span = self._build_sent_span(clss_list)

            pre_src = [x['src'] for x in batch]
            pre_d_labels = [x['d_labels'] for x in batch]
            pre_labels = [x['labels'] for x in batch]

            ids = [x['id'] for x in batch]

            src = torch.tensor(self._pad(pre_src, 0))
            d_labels = torch.tensor(self._pad(pre_d_labels, 0)).type(torch.float)
            labels = torch.tensor(self._pad(pre_labels, 0)).type(torch.float)
#             segs = torch.tensor(self._pad(pre_segs, 0))
            # tree_embed = torch.tensor(self._cut_pad_tree_embed(tree_embed))
            mask = (~(src == 0)).type(torch.float)

            #batch*edu_num
            if unit=='edu': 
                edu_mask = (~(torch.sum(edu_span,dim=1) == 0)).type(torch.float)
            else:
                sent_mask = (~(torch.sum(sent_span,dim=1) == 0)).type(torch.float)

            setattr(self, 'src_list', pre_src)
            setattr(self, 'd_span_list', d_span_list)

            if unit=='edu':
                setattr(self, 'unit_mask', edu_mask.to(device))
            else:
                setattr(self, 'unit_mask', sent_mask.to(device))

            setattr(self, 'ids', ids)

            if (is_test):
                if unit=='edu':
                    src_str = [x['disco_txt'] for x in batch]
                    setattr(self, 'unit_txt', src_str)
                    disco_dep= [x['disco_dep'] for x in batch]
                    setattr(self, 'disco_dep', disco_dep)
                    disco_to_sent= [x['disco_to_sent'] for x in batch]
                    setattr(self, 'disco_to_sent', disco_to_sent)
                    sent_to_para= [x['sent_to_para'] for x in batch]
                    setattr(self, 'sent_to_para', sent_to_para)
                else:
                    src_str = [x['sent_txt'] for x in batch]
                    setattr(self, 'unit_txt', src_str)

    def __len__(self):
        return self.batch_size

class SummarizationDataLoader(DataLoader):
    def __init__(self,dataset, batch_size=5,device=-1, max_length = 10000, is_test=False,unit='edu'):
        super(SummarizationDataLoader, self).__init__(
            dataset, batch_size=batch_size,collate_fn =self.collate_fn)
        self.max_length = max_length
        self.is_test=is_test
        self.device=device
        self.unit = unit
    def collate_fn(self,batch):
        # if self.bert_unit_encoder:
        # 	return BatchBERTEncoder(batch, self.device, self.is_test, self.max_length)
        # else:
        return Batch(batch,self.device,self.is_test,self.max_length,self.unit)