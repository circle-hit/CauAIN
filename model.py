from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import numpy as np, itertools, random, copy, math
from model import SimpleAttention, MatchingAttention, Attention
from dynamic_rnn import DynamicLSTM

class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss

class Cause_Effect_Reasoning(nn.Module):
    def __init__(self, D_h, attention_probs_dropout_prob=0.1):
        super(Cause_Effect_Reasoning, self).__init__()
        self.D_h = D_h
        
        self.weight = nn.Linear(2*D_h, 2*D_h)
        self.query = nn.Linear(2*D_h, 2*D_h)
        self.key = nn.Linear(2*D_h, 2*D_h)
        self.value = nn.Linear(2*D_h, 2*D_h)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def forward(self, inter_features, intra_c_e_mask, inter_c_e_mask, effects_from_self, effects_from_others):
        
        # cause_candidate = self.cause_trans(inter_features):

        query_layer_intra = self.query(inter_features)
        key_layer_intra = self.key(inter_features) + self.weight(effects_from_self.transpose(0, 1))
        value_layer_intra = self.value(inter_features) + self.weight(effects_from_self.transpose(0, 1))
        
        query_layer_inter = self.query(inter_features)
        key_layer_inter = self.key(inter_features) + self.weight(effects_from_others.transpose(0, 1))
        value_layer_inter = self.value(inter_features) + self.weight(effects_from_others.transpose(0, 1))

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores_intra = torch.matmul(query_layer_intra, key_layer_intra.transpose(-1, -2))
        attention_scores_intra = attention_scores_intra / math.sqrt(self.D_h)
        attention_scores_intra = attention_scores_intra * intra_c_e_mask

        attention_scores_inter = torch.matmul(query_layer_inter, key_layer_inter.transpose(-1, -2))
        attention_scores_inter = attention_scores_inter / math.sqrt(self.D_h)
        attention_scores_inter = attention_scores_inter * inter_c_e_mask
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        mask = intra_c_e_mask + inter_c_e_mask
        attention_scores_both = attention_scores_intra + attention_scores_inter

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores_both.masked_fill(mask==0, -1e9))
        attention_probs = self.dropout(attention_probs)
    
        effect_vectors = torch.matmul(attention_probs*intra_c_e_mask, value_layer_intra) + torch.matmul(attention_probs*inter_c_e_mask, value_layer_inter)
        cause_vectors = torch.matmul(attention_probs*intra_c_e_mask, query_layer_intra) + torch.matmul(attention_probs*inter_c_e_mask, query_layer_inter)
        final_features = torch.cat([effect_vectors, cause_vectors], dim=-1)

        return final_features.transpose(0, 1)

class CauAIN(nn.Module):
    def __init__(self, opt, n_classes=7):

        super(CauAIN, self).__init__()

        if opt.norm:
            norm_train = True
            self.norm1a = nn.LayerNorm(opt.roberta_dim, elementwise_affine=norm_train)

        self.opt = opt
        self.linear_in = nn.Linear(opt.roberta_dim, opt.hidden_dim)
        self.global_rnn = DynamicLSTM(opt.hidden_dim, opt.hidden_dim, bidirectional=True, rnn_type=opt.rnn_type)
    
        self.fusion_inter_effects = nn.Linear(3*opt.csk_dim, 2*opt.hidden_dim)
        self.fusion_intra_effects = nn.Linear(3*opt.csk_dim, 2*opt.hidden_dim)
        self.cause_effect_reasoning = Cause_Effect_Reasoning(opt.hidden_dim)

        layers = [nn.Linear(4*opt.hidden_dim, opt.hidden_dim), nn.ReLU()]
        for _ in range(opt.mlp_layers - 1):
            layers += [nn.Linear(opt.hidden_dim, opt.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(opt.hidden_dim, n_classes)]
        self.smax_fc = nn.Sequential(*layers)

    def forward(self, r1, r2, r3, r4, x1, x2, x3, x4, x5, x6, o1, o2, o3, qmask, umask, inter_c_e_mask, intra_c_e_mask, speaker_ids, inter_position_index=None, intra_position_index=None, att2=False, return_hidden=False):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        seq_len, batch, feature_dim = r1.size()

        r = (r1 + r2 + r3 + r4) / 4
        if self.opt.norm:
            r = self.norm1a(r.transpose(0, 1).reshape(-1, feature_dim)).reshape(-1, seq_len, feature_dim).transpose(1, 0)
        r = self.linear_in(r)
        
        text_len = torch.sum(umask != 0, dim=-1).cpu()
        
        inter_features, _ = self.global_rnn(r.transpose(0, 1), text_len)
        inter_features = inter_features.transpose(0, 1)
    
        effects_from_self = self.fusion_intra_effects(torch.cat([x4, x5, x6], dim=-1))
        effects_from_others = self.fusion_inter_effects(torch.cat([o1, o2, o3], dim=-1))
        final_features = self.cause_effect_reasoning(inter_features.transpose(0, 1),  intra_c_e_mask, inter_c_e_mask, effects_from_self, effects_from_others)
        final_features = self.dropout(final_features)
    
        log_prob = F.log_softmax(self.smax_fc(final_features), 2)
        return log_prob