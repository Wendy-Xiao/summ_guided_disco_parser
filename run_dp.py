import torch
import os
import sys



def evaluate(model,dataloader,pos_weight,device, hyp_path, ref_path, \
            word_length_limit=80, unit_length_limit=100,use_edu=True,\
            use_mmr=False, lamb=0.6,\
            use_trigram_block=False):
    model.eval()
    all_ids = []
    total_loss = 0
    all_selections = {}
    all_attns=[]
    for i,batch in enumerate(dataloader):
        batch_ids, attn = evaluate_batch(model, batch, device)
        all_ids.extend(batch_ids)
        all_attns.append(attn)
        del attn, loss
        if i%10==0:
            print(i)
    return all_ids,all_attns


def evaluate_batch(model,batch, device):
    out,attn,unit_repre = model(batch,device)
    # batch * length
    batch_ids = batch.ids
    total_num = torch.sum(batch.unit_mask)
    del batch,out
    # print(attn.shape)
    return batch_ids, attn