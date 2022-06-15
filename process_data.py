import pickle
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import numpy
import torch

def get_input_features(ids, split):
    pass

def get_self_speaker_features():
    speakers, labels, roberta1, roberta2, roberta3, roberta4, sentences, trainIds, testIds, validIds  = pickle.load(open('iemocap/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
    self_utt = {}
    totalIds = trainIds + testIds + validIds

    for ids in tqdm(totalIds):
        cur_speaker_list = speakers[ids]
        speaker2id = {s:i for i, s in enumerate(set(cur_speaker_list))}
        cur_r1, cur_r2, cur_r3, cur_r4 = roberta1[ids], roberta2[ids], roberta3[ids], roberta4[ids]
        data = [{'utterance': [], 'index': []}for _ in range(len(set(cur_speaker_list)))]
        assert(len(cur_speaker_list) == len(cur_r1))
        for i in range(len(cur_r1)):
            speaker_id = speaker2id[cur_speaker_list[i]]
            data[speaker_id]['utterance'].append((cur_r1[i] + cur_r2[i] + cur_r3[i]+ cur_r4[i]) / 4)
            data[speaker_id]['index'].append(i)

        self_utt[ids] = data

    pickle.dump([speakers, labels, roberta1, roberta2, roberta3, roberta4, self_utt,\
        sentences, trainIds, testIds, validIds], open('iemocap/iemocap_features_roberta_new.pkl', 'wb'))

def get_cause_effect_masks(dataset='iemocap', position_upper=10):
    if dataset == 'iemocap':
        speakers, labels, roberta1, roberta2, roberta3, roberta4, sentences, trainIds, testIds, validIds  = pickle.load(open('iemocap/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
    elif dataset == 'meld':
        speakers, emotion_labels, sentiment_labels, roberta1, roberta2, roberta3, roberta4, \
        sentences, trainIds, testIds, validIds = pickle.load(open('meld/meld_features_roberta.pkl', 'rb'), encoding='latin1')
    elif dataset == 'emorynlp':
        speakers, emotion_labels, roberta1, roberta2, roberta3, roberta4, \
        sentences, trainIds, testIds, validIds = pickle.load(open('emorynlp/emorynlp_features_roberta.pkl', 'rb'), encoding='latin1')
    else:
        speakers, labels, roberta1, roberta2, roberta3, roberta4, \
        sentences, trainIds, testIds, validIds = pickle.load(open('dailydialog/dailydialog_features_roberta.pkl', 'rb'), encoding='latin1')
    
    inter_cause_effect_masks, intra_cause_effect_masks = {}, {}
    inter_position_index, intra_position_index = {}, {}
    totalIds = trainIds + testIds + validIds
    for ids in tqdm(totalIds):

        if dataset == 'meld' or dataset == 'emorynlp':
            cur_speaker_list = [numpy.argmax(i) for i in speakers[ids]]
        else:
            cur_speaker_list = speakers[ids]

        cur_inter_cause_effect_mask = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_intra_cause_effect_mask = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_inter_position_index = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_intra_position_index = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        
        # cur_inter_cause_effect_mask = numpy.array([[position_upper+1]*len(cur_speaker_list)]*len(cur_speaker_list))
        # cur_intra_cause_effect_mask = numpy.array([[position_upper+1]*len(cur_speaker_list)]*len(cur_speaker_list))
        # cur_inter_position_index = numpy.array([[position_upper+1]*len(cur_speaker_list)]*len(cur_speaker_list))
        # cur_intra_position_index = numpy.array([[position_upper+1]*len(cur_speaker_list)]*len(cur_speaker_list))

        cur_intra_cause_effect_mask[0][0] = 1
        for i in range(1, len(cur_speaker_list)):
            inter_cnt, intra_cnt = 1, 1
            speaker_now = cur_speaker_list[i]
            cur_intra_cause_effect_mask[i][i] = 1
            j = i-1
            while(j >= 0):
                if cur_speaker_list[j] != speaker_now:
                    cur_inter_cause_effect_mask[i][j] = 1
                    if inter_cnt <= position_upper:
                        cur_inter_position_index[i][j] = inter_cnt
                        inter_cnt += 1
                else:
                    cur_intra_cause_effect_mask[i][j] = 1
                    if intra_cnt <= position_upper:
                        cur_intra_position_index[i][j] = intra_cnt
                        intra_cnt += 1
                j -= 1
        inter_cause_effect_masks[ids] = cur_inter_cause_effect_mask
        intra_cause_effect_masks[ids] = cur_intra_cause_effect_mask
        inter_position_index[ids] = cur_inter_position_index
        intra_position_index[ids] = cur_intra_position_index
    
    if dataset == 'iemocap':
        pickle.dump([speakers, labels, roberta1, roberta2, roberta3, roberta4, inter_cause_effect_masks, intra_cause_effect_masks, inter_position_index, intra_position_index, \
            sentences, trainIds, testIds, validIds], open('iemocap/iemocap_features_roberta_cause_effect.pkl', 'wb'))
    elif dataset == 'meld':
        pickle.dump([speakers, emotion_labels, sentiment_labels, roberta1, roberta2, roberta3, roberta4, inter_cause_effect_masks, intra_cause_effect_masks, inter_position_index, intra_position_index, \
            sentences, trainIds, testIds, validIds], open('meld/meld_features_roberta_cause_effect.pkl', 'wb'))
    elif dataset == 'emorynlp':
        pickle.dump([speakers, emotion_labels, roberta1, roberta2, roberta3, roberta4, inter_cause_effect_masks, intra_cause_effect_masks, inter_position_index, intra_position_index, \
            sentences, trainIds, testIds, validIds], open('emorynlp/emorynlp_features_roberta_cause_effect.pkl', 'wb'))
    else:
        pickle.dump([speakers, labels, roberta1, roberta2, roberta3, roberta4, inter_cause_effect_masks, intra_cause_effect_masks, inter_position_index, intra_position_index, \
            sentences, trainIds, testIds, validIds], open('dailydialog/dailydialog_features_roberta_cause_effect.pkl', 'wb'))

def get_cause_effect_masks_with_window_size(dataset='iemocap', window_size=7):
    if dataset == 'iemocap':
        speakers, labels, roberta1, roberta2, roberta3, roberta4, sentences, trainIds, testIds, validIds  = pickle.load(open('iemocap/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
    elif dataset == 'meld':
        speakers, emotion_labels, sentiment_labels, roberta1, roberta2, roberta3, roberta4, \
        sentences, trainIds, testIds, validIds = pickle.load(open('meld/meld_features_roberta.pkl', 'rb'), encoding='latin1')
    elif dataset == 'emorynlp':
        speakers, emotion_labels, roberta1, roberta2, roberta3, roberta4, \
        sentences, trainIds, testIds, validIds = pickle.load(open('emorynlp/emorynlp_features_roberta.pkl', 'rb'), encoding='latin1')
    else:
        speakers, labels, roberta1, roberta2, roberta3, roberta4, \
        sentences, trainIds, testIds, validIds = pickle.load(open('dailydialog/dailydialog_features_roberta.pkl', 'rb'), encoding='latin1')
    
    inter_cause_effect_masks, intra_cause_effect_masks = {}, {}
    inter_position_index, intra_position_index = {}, {}
    totalIds = trainIds + testIds + validIds
    for ids in tqdm(totalIds):
        if dataset == 'meld' or dataset == 'emorynlp':
            cur_speaker_list = [numpy.argmax(i) for i in speakers[ids]]
        else:
            cur_speaker_list = speakers[ids]
        cur_inter_cause_effect_mask = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_intra_cause_effect_mask = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_inter_position_index = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_intra_position_index = numpy.zeros((len(cur_speaker_list), len(cur_speaker_list)))
        cur_intra_cause_effect_mask[0][0] = 1
        for i in range(1, len(cur_speaker_list)):
            inter_cnt, intra_cnt = 0, 0
            speaker_now = cur_speaker_list[i]
            cur_intra_cause_effect_mask[i][i] = 1
            j = i-1
            while(j >= 0):
                if cur_speaker_list[j] != speaker_now:
                    inter_cnt += 1
                    if inter_cnt <= window_size:
                        cur_inter_cause_effect_mask[i][j] = 1
                        cur_inter_position_index[i][j] = inter_cnt
                else:
                    intra_cnt += 1
                    if intra_cnt <= window_size:
                        cur_intra_cause_effect_mask[i][j] = 1
                        cur_intra_position_index[i][j] = intra_cnt
                j -= 1
        inter_cause_effect_masks[ids] = cur_inter_cause_effect_mask
        intra_cause_effect_masks[ids] = cur_intra_cause_effect_mask
        inter_position_index[ids] = cur_inter_position_index
        intra_position_index[ids] = cur_intra_position_index
    
    if dataset == 'iemocap':
        pickle.dump([speakers, labels, roberta1, roberta2, roberta3, roberta4, inter_cause_effect_masks, intra_cause_effect_masks, \
            sentences, trainIds, testIds, validIds], open('iemocap/iemocap_features_roberta_cause_effect_window' + str(window_size) + '.pkl', 'wb'))
    elif dataset == 'meld':
        pickle.dump([speakers, emotion_labels, sentiment_labels, roberta1, roberta2, roberta3, roberta4, inter_cause_effect_masks, intra_cause_effect_masks, inter_position_index, intra_position_index, \
            sentences, trainIds, testIds, validIds], open('meld/meld_features_roberta_cause_effect_window' + str(window_size) + '.pkl', 'wb'))
    elif dataset == 'emorynlp':
        pickle.dump([speakers, emotion_labels, roberta1, roberta2, roberta3, roberta4, inter_cause_effect_masks, intra_cause_effect_masks,\
            sentences, trainIds, testIds, validIds], open('emorynlp/emorynlp_features_roberta_cause_effect_window' + str(window_size) + '.pkl', 'wb'))
    else:
        pickle.dump([speakers, labels, roberta1, roberta2, roberta3, roberta4, inter_cause_effect_masks, intra_cause_effect_masks, \
            sentences, trainIds, testIds, validIds], open('dailydialog/dailydialog_features_roberta_cause_effect_window' + str(window_size) + '.pkl', 'wb'))

def get_emotion_shift_data():
    speakers, labels, roberta1, roberta2, roberta3, roberta4, sentences, trainIds, testIds, validIds  = pickle.load(open('iemocap/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
    masks = {}
    for item in testIds:
        cur_label_list = labels[item]
        cur_speaker_list = speakers[item]
        emotion_shift_mask = numpy.zeros(len(cur_speaker_list))
        A_label_list = []
        B_label_list = []
        for i in range(len(cur_label_list)):
            if cur_speaker_list[i] == 'F':
                A_label_list.append(i)
            else:
                B_label_list.append(i)
        for i in range(1, len(A_label_list)):
            if cur_label_list[A_label_list[i]] != 2 and cur_label_list[A_label_list[i]] != cur_label_list[A_label_list[i-1]]:
                emotion_shift_mask[A_label_list[i]] = 1
        for i in range(1, len(B_label_list)):
            if cur_label_list[B_label_list[i]] != 2 and cur_label_list[B_label_list[i]] != cur_label_list[B_label_list[i-1]]:
                emotion_shift_mask[B_label_list[i]] = 1
        masks[item] = emotion_shift_mask
    pickle.dump([speakers, labels, roberta1, roberta2, roberta3, roberta4, masks, sentences, trainIds, testIds, validIds], open('iemocap/iemocap_features_for_emotion_shift.pkl', 'wb'))


if __name__ == '__main__':
    for dataset in ['iemocap', 'dailydialog', 'meld']:
        get_cause_effect_masks(dataset)
    # for window_size in range(1, 7):
    #     get_cause_effect_masks_with_window_size('meld', window_size=window_size)
    # get_emotion_shift_data()




