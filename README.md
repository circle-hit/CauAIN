# CauAIN

* Code for [IJCAI 2022](https://www.ijcai.org) accepted paper titled "CauAIN: Causal Aware Interaction Network for Emotion Recognition in Conversations"
* Weixiang Zhao, Yanyan Zhao, Xin Lu.

## Requirements
* Python 3.7
* PyTorch 1.8.2
* Transformers 4.12.3
* CUDA 11.1

## Preparation

### Datasets, Utterance Feature and CommonSense Feature
You can download the dataset, extracted utterance feature and commonsense feature we used from:
https://pan.baidu.com/s/1hNuUgUmjdOOQ1cqT75n-ZA  提取码:nwqa
or:
https://drive.google.com/drive/folders/1--yCEESJ1TMReiiiWwTlnNL7H1p-erfq?usp=sharing

and place them into the corresponding folds like iemocap, dailydialog and meld

## Training
You can train the models with the following codes:

For IEMOCAP: 
`python train_iemocap.py`

For MELD: 
`python train_meld.py`

For DailyDialog: 
`python train_dailydialog.py`
