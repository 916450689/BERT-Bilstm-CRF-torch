#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/4 12:40
# @Author  : JJkinging
# @File    : config.py
class Config(object):
    '''配置类'''

    def __init__(self):
        self.label_file = '../dataset/tag.txt'
        self.train_file = '../dataset/train.txt'
        self.dev_file = '../dataset/dev.txt'
        self.test_file = '../dataset/test.txt'
        self.vocab = '../dataset/bert/vocab.txt'
        self.max_length = 250
        self.use_cuda = False
        self.gpu = 0
        self.batch_size = 8
        self.rnn_hidden = 128
        self.bert_embedding = 768
        self.dropout = 0.5
        self.rnn_layer = 1
        self.lr = 0.00006
        self.lr_decay = 0.00001
        self.weight_decay = 0.00005
        self.checkpoint = '../result/checkpoints/RoBERTa_result/RoBERTa_best_0.9225_18.4679.pth.tar'
        #self.checkpoint = None
        self.epochs = 100
        self.max_grad_norm = 10
        self.target_dir = '../result/checkpoints/RoBERTa_result'
        self.patience = 100
        # 可以换成RoBERTa的中文预训练模型（哈工大提供）,也可修改为其他模型
        #self.pretrain_model_name = 'hfl/chinese-roberta-wwm-ext'
        self.pretrain_model_name = 'D:\\推荐算法\\深度学习\\BERT-Bilstm-CRF-torch\\chinese_roberta_wwm_ext_pytorch'


    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':
    con = Config()
    con.update(gpu=8)
    print(con.gpu)
    print(con)