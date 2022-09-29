#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/5 18:35
# @Author  : JJkinging
# @File    : predict.py
import torch
import numpy as np
from model.BERT_BiLSTM_CRF import BERT_BiLSTM_CRF
from scripts.config import Config
from scripts.utils import load_vocab
from seqeval.scheme import IOBES
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
'''用于识别输入的句子（可以换成批量输入）的命名实体
'''
def predict(input_seq, max_length=150):
    '''
    :param input_seq: 输入一句话
    :return:
    '''
    config = Config()
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.label_file)
    tagset_size = len(label_dic)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BERT_BiLSTM_CRF(tagset_size,
                            config.bert_embedding,
                            config.rnn_hidden,
                            config.rnn_layer,
                            config.dropout,
                            config.pretrain_model_name,
                            device).to(device)
    # 原 config.dropout_ratio,config.dropout1
    checkpoint = torch.load(config.checkpoint,map_location='cpu')
    print(config.checkpoint)
    model.load_state_dict(checkpoint["model"])


    # 构造输入
    input_list = []
    for i in range(len(input_seq)):
        input_list.append(input_seq[i])

    if len(input_list) > max_length - 2:
        input_list = input_list[0:(max_length - 2)]
    input_list = ['[CLS]'] + input_list + ['[SEP]']

    input_ids = [int(vocab[word]) if word in vocab else int(vocab['[UNK]']) for word in input_list]
    input_mask = [1] * len(input_ids)

    if len(input_ids) < max_length:
        input_ids.extend([0] * (max_length - len(input_ids)))
        input_mask.extend([0] * (max_length - len(input_mask)))
    assert len(input_ids) == max_length
    assert len(input_mask) == max_length

    # 变为tensor并放到GPU上, 二维, 这里mask在CRF中必须为unit8类型或者bool类型
    input_ids = torch.LongTensor([input_ids]).to(device)
    input_mask = torch.ByteTensor([input_mask]).to(device)

    feats = model(input_ids, input_mask)
    # out_path是一条预测路径（数字列表）, [1:-1]表示去掉一头一尾, <START>和<EOS>标志
    # 进行预测
    out_path = model.predict(feats, input_mask)[0][1:-1]
    out_path = convert_id2tag(out_path)

    return out_path

# 将id转换成tag
def convert_id2tag(input_list):
    config = Config()
    label_dic = load_vocab(config.label_file)
    sentence_tag = []
    for id in input_list:
        for key in label_dic.keys():
            if label_dic[key] == int(id):
                sentence_tag.append(key)

    return sentence_tag

def find_entity(text,tag):
    # 查找所有实体
    dict = []
    for i in range(0, len(tag)):
        c=0
        word = []
        for j in range(0, len(tag[i])):
            if (tag[i][j] != "O" and tag[i][j] != "\n"):
                word_tag = tag[i][j].split('-')
                if (word_tag[0] == 'S'):
                    t = text[i][j]
                    word.append(''.join(t))
                elif (word_tag[0] == "B" or word_tag[0] == "I"):
                    c = c + 1
                elif (word_tag[0] == "E"):
                    t = text[i][j - c:j + 1]
                    word.append(''.join(t))
                    c = 0
        dict.append(word)
    return dict

if __name__ == "__main__":

    #f = open('E:\\nhr\\BERT-Bilstm-CRF-pytorch\\dataset\\result.txt','w',encoding='utf-8')
    # 读取测试集
    with open('../dataset/test.txt','r',encoding='utf-8') as fp:
        word = []
        label = []
        words = []
        labels = []
        result = []

        show=[]
        lines = fp.readlines()
        i=0
        for line in lines:
            i=i+1
            contends = line.strip()
            tokens = contends.split(' ')
            if len(tokens) == 2:
                word.append(tokens[0])
                label.append(tokens[1])
            else:
                if len(contends) == 0 and len(word) > 0:
                    words.append(word)
                    labels.append(label)
                    word = []
                    label = []
        #print(words)
        #print(labels)
    # 对测试集语句进行预测




        for word in words:
            label = predict(word)

            result.append(label)
        for word in words:
            label_copy = predict(word)

            for i in range(len(label_copy)):
                label_copy[i] = word[i] + '-' + label_copy[i]

            show.append(label_copy)

        #print(show)
        f = open('result.txt', 'a+')
        for i in range(len(show)):
            for j in range(len(show[i])):
                f.write(show[i][j])
                f.write(' ')
            f.write('\n')


    # 指标评价
        precision = precision_score(labels,result)
        recall = recall_score(labels, result)
        F1_score = f1_score(labels, result)

        print("p: ", precision)
        print("r: ",recall)
        print("f1: ",F1_score)
        # 评估test true:labels  pred:result
        print(classification_report(labels, result, mode='strict', scheme=IOBES))

    # 查找所有实体
        true_entity = find_entity(words,labels)
        pred_entity = find_entity(words,result)
        # 找差集
        f=open('差集.txt','a+')


        for i in range(0,len(true_entity)):
            s1 = list(set(true_entity[i]).difference(set(pred_entity[i])))
            s2 = list(set(pred_entity[i]).difference(set(true_entity[i])))
            if(len(s1)!= 0 or len(s2) != 0):
                f.write(''.join(words[i])+'\n')
                f.write("未识别正确的原实体： ")
                for m in s1:
                    f.write(m + '   ')
                f.write('\n')
                f.write("识别出的错误实体： ")
                for n in s2:
                    f.write(n + '   ')
                f.write('\n')
                f.write("----------------------------------------"+'\n')


