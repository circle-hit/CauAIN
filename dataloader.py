import numpy
import torch
from torch.nn.modules import padding
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd

def pad_matrix(matrix, padding_index=0):
    max_len = max(i.size(0) for i in matrix)
    batch_matrix = []
    for item in matrix:
        item = item.numpy()
        batch_matrix.append(numpy.pad(item, ((0, max_len-len(item)), (0, max_len-len(item))), 'constant', constant_values=(padding_index, padding_index)))
    return batch_matrix

class IEMOCAPRobertaCometDataset(Dataset):

    def __init__(self, split):
        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

        self.speakers, self.labels, self.roberta1, self.roberta2, self.roberta3, self.roberta4, self.inter_cause_effect_masks, self.intra_cause_effect_masks, self.inter_position_index, self.intra_position_index, \
        self.sentences, self.trainIds, self.testIds, self.validIds = pickle.load(open('iemocap/iemocap_features_roberta_cause_effect.pkl', 'rb'), encoding='latin1')

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open('iemocap/iemocap_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        
        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.roberta2[vid]),\
               torch.FloatTensor(self.roberta3[vid]),\
               torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor(self.xIntent[vid]),\
               torch.FloatTensor(self.xAttr[vid]),\
               torch.FloatTensor(self.xNeed[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xEffect[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oEffect[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor([[1,0] if x=='M' else [0,1] for x in self.speakers[vid]]),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]),\
               torch.FloatTensor(self.inter_cause_effect_masks[vid]),\
               torch.FloatTensor(self.intra_cause_effect_masks[vid]),\
               torch.LongTensor(self.inter_position_index[vid]),\
               torch.LongTensor(self.intra_position_index[vid]),\
               vid
               
    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        inter_cause_effect_masks = torch.FloatTensor(pad_matrix(dat[16]))
        intra_cause_effect_masks = torch.FloatTensor(pad_matrix(dat[17]))
        inter_position_index = torch.LongTensor(pad_matrix(dat[18]))
        intra_position_index = torch.LongTensor(pad_matrix(dat[19]))
        return [pad_sequence(dat[i]) if i<14 else pad_sequence(dat[i], True) if i<16 else inter_cause_effect_masks if i < 17 else intra_cause_effect_masks if i<18 else inter_position_index if i < 19 else intra_position_index if i < 20 else dat[i].tolist() for i in dat]
    
class MELDRobertaCometDataset(Dataset):

    def __init__(self, split, classify='emotion'):
        '''
        label index mapping = 
        '''
        self.speakers, self.emotion_labels, self.sentiment_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, self.inter_cause_effect_masks, self.intra_cause_effect_masks, self.inter_position_index, self.intra_position_index,\
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open('meld/meld_features_roberta_cause_effect.pkl', 'rb'), encoding='latin1')  

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open('meld/meld_features_comet.pkl', 'rb'), encoding='latin1')
        
        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.roberta2[vid]),\
               torch.FloatTensor(self.roberta3[vid]),\
               torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor(self.xIntent[vid]),\
               torch.FloatTensor(self.xAttr[vid]),\
               torch.FloatTensor(self.xNeed[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xEffect[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oEffect[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor(self.speakers[vid]),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]),\
               torch.FloatTensor(self.inter_cause_effect_masks[vid]),\
               torch.FloatTensor(self.intra_cause_effect_masks[vid]),\
               torch.LongTensor(self.inter_position_index[vid]),\
               torch.LongTensor(self.intra_position_index[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        inter_cause_effect_masks = torch.FloatTensor(pad_matrix(dat[16]))
        intra_cause_effect_masks = torch.FloatTensor(pad_matrix(dat[17]))
        inter_position_index = torch.LongTensor(pad_matrix(dat[18]))
        intra_position_index = torch.LongTensor(pad_matrix(dat[19]))
        return [pad_sequence(dat[i]) if i<14 else pad_sequence(dat[i], True) if i<16 else inter_cause_effect_masks if i < 17 else intra_cause_effect_masks if i<18 else inter_position_index if i < 19 else intra_position_index for i in dat]

class DailyDialogueRobertaCometDataset(Dataset):

    def __init__(self, split):

        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, self.inter_cause_effect_masks, self.intra_cause_effect_masks, self.inter_position_index, self.intra_position_index, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
        = pickle.load(open('dailydialog/dailydialog_features_roberta_cause_effect.pkl', 'rb'), encoding='latin1')

        self.xIntent, self.xAttr, self.xNeed, self.xWant, self.xEffect, self.xReact, self.oWant, self.oEffect, self.oReact \
        = pickle.load(open('dailydialog/dailydialog_features_comet.pkl', 'rb'), encoding='latin1')

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.roberta1[vid]),\
               torch.FloatTensor(self.roberta2[vid]),\
               torch.FloatTensor(self.roberta3[vid]),\
               torch.FloatTensor(self.roberta4[vid]),\
               torch.FloatTensor(self.xIntent[vid]),\
               torch.FloatTensor(self.xAttr[vid]),\
               torch.FloatTensor(self.xNeed[vid]),\
               torch.FloatTensor(self.xWant[vid]),\
               torch.FloatTensor(self.xEffect[vid]),\
               torch.FloatTensor(self.xReact[vid]),\
               torch.FloatTensor(self.oWant[vid]),\
               torch.FloatTensor(self.oEffect[vid]),\
               torch.FloatTensor(self.oReact[vid]),\
               torch.FloatTensor([[1,0] if x=='0' else [0,1] for x in self.speakers[vid]]),\
               torch.FloatTensor([1]*len(self.labels[vid])),\
               torch.LongTensor(self.labels[vid]),\
               torch.FloatTensor(self.inter_cause_effect_masks[vid]),\
               torch.FloatTensor(self.intra_cause_effect_masks[vid]),\
               torch.LongTensor(self.inter_position_index[vid]),\
               torch.LongTensor(self.intra_position_index[vid]),\
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        inter_cause_effect_masks = torch.FloatTensor(pad_matrix(dat[16]))
        intra_cause_effect_masks = torch.FloatTensor(pad_matrix(dat[17]))
        inter_position_index = torch.LongTensor(pad_matrix(dat[18], padding_index=11))
        intra_position_index = torch.LongTensor(pad_matrix(dat[19], padding_index=11))
        return [pad_sequence(dat[i]) if i<14 else pad_sequence(dat[i], True) if i<16 else inter_cause_effect_masks if i < 17 else intra_cause_effect_masks if i<18 else inter_position_index if i < 19 else intra_position_index if i < 20 else dat[i].tolist() for i in dat]