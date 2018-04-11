# coding: utf-8

import numpy
from collections import defaultdict
import torch
from torch.autograd import Variable
from preprocessing import prepare_data, data_generator


def encode_sentences(curr_model, pair, batch_size=128, test=False):
    """
    Encode sentences into the joint embedding space
    """
    en_feats = numpy.zeros((len(pair[0]), curr_model['options']['dim']), dtype='float32')
    cn_feats = numpy.zeros((len(pair[0]), curr_model['options']['dim']), dtype='float32')

    data_index = prepare_data(pair, curr_model['worddict'], test=test)
    cur = 0
    for en, cn, en_lengths, cn_lengths, en_index, cn_index in data_generator(data_index, batch_size):
        en, cn = curr_model['en_cn_model'].forward(en, en_lengths, en_index, cn, cn_lengths, cn_index)
        en = en.data.cpu().numpy()
        cn = cn.data.cpu().numpy()
        for i in xrange(batch_size):
            if i + cur >= len(pair[0]):
                break
            for j in xrange(curr_model['options']['dim']):
                en_feats[i + cur][j] = en[i][j]
                cn_feats[i + cur][j] = cn[i][j]
        cur += batch_size
    en_feats = Variable(torch.from_numpy(en_feats).cuda())
    cn_feats = Variable(torch.from_numpy(cn_feats).cuda())
    return en_feats, cn_feats

def encode_sentences_np(curr_model, pair, batch_size=128, test=False):
    """
    Encode sentences into the joint embedding space
    """
    en_feats = numpy.zeros((len(pair[0]), curr_model['options']['dim']), dtype='float32')
    cn_feats = numpy.zeros((len(pair[0]), curr_model['options']['dim']), dtype='float32')

    data_index = prepare_data(pair, curr_model['worddict'], test=test)
    cur = 0
    for en, cn, en_lengths, cn_lengths, en_index, cn_index in data_generator(data_index, batch_size):
        en, cn = curr_model['en_cn_model'].forward(en, en_lengths, en_index, cn, cn_lengths, cn_index)
        en = en.data.cpu().numpy()
        cn = cn.data.cpu().numpy()
        for i in xrange(batch_size):
            if i + cur >= len(pair[0]):
                break
            for j in xrange(curr_model['options']['dim']):
                en_feats[i + cur][j] = en[i][j]
                cn_feats[i + cur][j] = cn[i][j]
        cur += batch_size
    # en_feats = Variable(torch.from_numpy(en_feats).cuda())
    # cn_feats = Variable(torch.from_numpy(cn_feats).cuda())
    return en_feats, cn_feats
