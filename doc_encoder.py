''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

###################################
#### different attention types ####
###################################
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        # self.mix_weights= nn.Parameter(torch.rand(1))
        # self.mix_weights=nn.Linear(512,1)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
            # attn = attn.masked_fill(mask, 0)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn

###################################
#### position feed forward ####
###################################

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

##########################################################
#### multi head attention with diffent attention type ####
##########################################################

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v


        self.w_vs = nn.Linear(d_model, n_head * d_v)

        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))



        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(p=dropout)


    def forward(self, q, k, v, mask=None, tree_attn=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        residual = q
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        # n_head = n_head_local+n_head_globa
        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk

        output, attn = self.attention(q, k, v,mask=mask)
        output = output.view(n_head, sz_b, len_q, d_v)


        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output,attn

########################
#### Encoder Layer ####
#######################
class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head,d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        enc_output *= non_pad_mask
        # print(enc_output.size()) #b*lv*d_v
        enc_output = self.pos_ffn(enc_output)

        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

##################################
#### Complete Document Encoder####
##################################
class Encoder(nn.Module):
	''' A encoder model with self attention mechanism. '''

	def __init__(
			self,n_layers, n_head,d_k, d_v,
			d_model, d_inner, dropout=0.1):

		super().__init__()


		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])

	def forward(self, enc_output, edu_mask, return_attns=False):

		enc_slf_attn_list = []
		if (enc_output != enc_output).any():
			print('nan at line 91 in EncoderForSumm.py')
		non_pad_mask = edu_mask.unsqueeze(-1)
		slf_attn_mask = (1-edu_mask).unsqueeze(1).expand(-1,edu_mask.size()[1],-1).type(torch.bool)
		for enc_layer in self.layer_stack:
			enc_output, enc_slf_attn = enc_layer(
				enc_output,
				non_pad_mask=non_pad_mask,
				slf_attn_mask=slf_attn_mask)

			if (enc_output != enc_output).any():
				print('nan at line 101 in EncoderForSumm.py')
			if return_attns:
				enc_slf_attn_list += [enc_slf_attn]

		if return_attns:
			return enc_output, enc_slf_attn_list
		return enc_output
