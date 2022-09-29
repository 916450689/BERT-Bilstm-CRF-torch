#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/3 22:58
# @Author  : JJkinging
# @File    : utils.py
import torch
import os
import time
import torch.nn as nn
from tqdm import tqdm
from scripts.config import Config
from seqeval.scheme import IOBES
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report

'''工具类函数'''

class InputFeature(object):
    def __init__(self, input_id, label_id, input_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask

def load_vocab(vocab_file):
    '''construct word2id or label2id'''
    vocab = {}
    index = 0
    with open(vocab_file, 'r', encoding='utf-8') as fp:
        while True:
            token = fp.readline()
            if not token:
                break
            token = token.strip()  # 删除空白符
            vocab[token] = index
            index += 1
    return vocab

def read_corpus(path, max_length, label_dic, vocab):
    '''
    :param path: 数据集文件路径
    :param max_length: 句子最大长度
    :param label_dic: 标签字典
    :param vocab:
    :return:
    '''
    with open(path, 'r', encoding='utf-8') as fp:
        result = []
        words = []
        labels = []
        for line in fp:
            contends = line.strip()
            tokens = contends.split(' ')
            if len(tokens) == 2:
                words.append(tokens[0])
                labels.append(tokens[1])
            else:
                if len(contends) == 0 and len(words) > 0:
                    if len(words) > max_length - 2:
                        words = words[0:(max_length-2)]
                        labels = labels[0:(max_length-2)]
                    words = ['[CLS]'] + words + ['[SEP]']
                    labels = ['<START>'] + labels + ['<EOS>']
                    input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in words]
                    label_ids = [label_dic[i] for i in labels]
                    input_mask = [1] * len(input_ids)

                    # 填充
                    if len(input_ids) < max_length:
                        input_ids.extend([0]*(max_length-len(input_ids)))
                        label_ids.extend([0]*(max_length-len(label_ids)))
                        input_mask.extend([0]*(max_length-len(input_mask)))

                    assert len(input_ids) == max_length
                    assert len(label_ids) == max_length
                    assert len(input_mask) == max_length

                    feature = InputFeature(input_id=input_ids, label_id=label_ids, input_mask=input_mask)
                    result.append(feature)
                    # 还原words、labels = []
                    words = []
                    labels = []
        return result

def train(model,
          dataloader,
          optimizer,
          max_gradient_norm):
    '''
    Train a model for one epoch on some input data with a given optimizer and
    criterion.
    :param model: A torch module that must be trained on some input data.
    :param dataloader: A DataLoader object to iterate over the training data.
    :param optimizer: A torch optimizer to use for training on the input model.
    :param criterion: A loss criterion to use for training.
    :param max_gradient_norm: Max norm for gradient norm clipping.
    :return:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    '''
    # Switch the model to train mode.
    model.train()
    device = model.device
    epoch_start =time.time()
    batch_time_avg = 0.0
    running_loss = 0.0

    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, batch in enumerate(tqdm_batch_iterator):
        batch_start = time.time()

        # Move input and output data to the GPU if it is used.
        inputs, masks, tags = batch

        inputs = inputs.to(device)
        masks = masks.byte().to(device)
        tags = tags.to(device)

        optimizer.zero_grad()  # 梯度归零
        feats = model(inputs, masks)
        loss = model.loss(feats, tags, masks)
        loss.backward()  # 反向传播

        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)

        optimizer.step()

        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()

        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1),
                    running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)

    return epoch_time, epoch_loss

# 将id转换成tag
def convert_id2tag(input_list):
    config = Config()
    label_dic = load_vocab(config.label_file)
    #print(label_dic)
    result = []
    batch_tag = []
    sentence_tag = []
    for batch in input_list:
        for out_id_list in batch:
            for i in out_id_list:
                for key in label_dic.keys():
                    if label_dic[key] == int(i):
                        sentence_tag.append(key)
            batch_tag.append(sentence_tag)
            sentence_tag = []
        result.append(batch_tag)
        batch_tag = []

    return result



def valid(model,dataloader):
    model.eval()
    device = model.device
    pre_output = []
    true_output = []
    epoch_start = time.time()
    running_loss = 0.0

    with torch.no_grad():
        tqdm_batch_iterator = tqdm(dataloader)
        for _, batch in enumerate(tqdm_batch_iterator):
            inputs, masks, tags = batch

            real_length = torch.sum(masks, dim=1)
            tmp = []
            i = 0
            for line in tags.numpy().tolist():
                tmp.append(line[0: real_length[i]])
                i += 1

            true_output.append(tmp)

            inputs = inputs.to(device)
            masks = masks.byte().to(device)
            tags = tags.to(device)

            feats = model(inputs, masks)
            loss = model.loss(feats, tags, masks)
            out_path = model.predict(feats, masks)
            pre_output.append(out_path)

            running_loss += loss.item()
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)

    # 计算准确率、召回率、F1值
    true_tag = convert_id2tag(true_output)
    pred_tag = convert_id2tag(pre_output)
    sum_precision = 0
    sum_recall = 0
    sum_F1_score = 0
    for i in range(0,len(true_tag)):
        precision = precision_score(true_tag[i], pred_tag[i])
        recall = recall_score(true_tag[i], pred_tag[i])
        F1_score = f1_score(true_tag[i], pred_tag[i])
        sum_precision = sum_precision+precision
        sum_recall = sum_recall + recall
        sum_F1_score = sum_F1_score + F1_score

    estimator = (sum_precision/len(true_tag), sum_recall/len(true_tag), sum_F1_score/len(true_tag))

    true_tag_after = []
    for i in true_tag:
        item = []
        for j in i:
            l = len(j)
            j = j[1:l-1]
            item.append(j)
        true_tag_after.append(item)
    #print(true_tag_after)
    #print(pred_tag)
    #print(classification_report(true_tag_after, pred_tag, mode='strict', scheme=IOBES))
    return epoch_time, epoch_loss, estimator