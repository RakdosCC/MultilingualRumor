import numpy as np
from torch.autograd import Variable
import math
import torch


# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq_new = seq + [0 for i in xrange(max_length - len(seq))]
    return seq_new


def data_generator(data_pairs, batch_size):
    """
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index.
        batch_size: The size of the batch
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length.
    """
    data_size = len(data_pairs)
    num_batches = math.floor(data_size / batch_size)
    for i in xrange(0, data_size, batch_size):
        if i + batch_size <= data_size:
            batch_data_x = [d[0] for d in data_pairs[i:i + batch_size]]
            batch_data_y = [d[1] for d in data_pairs[i:i + batch_size]]
        else:
            batch_data_x = [d[0] for d in data_pairs[i:data_size]]
            batch_data_y = [d[1] for d in data_pairs[i:data_size]]

        # The lengths for UM_Corpus and labels to be padded to
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])

        # Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        # Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens, x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)
        # Reorder the lengths
        x_sorted_index = list(np.argsort(batch_x_lengths))
        x_reverse_sorted_index = [x for x in reversed(x_sorted_index)]
        x_index =[0 for _ in xrange(batch_size)]
        batch_x_pad_sorted = [batch_x_pad[i] for i in x_reverse_sorted_index]
        for i in xrange(len(x_reverse_sorted_index)):
            x_index[x_reverse_sorted_index[i]]=i

        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens, y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)
        y_sorted_index = list(np.argsort(batch_y_lengths))
        y_reverse_sorted_index = [y for y in reversed(y_sorted_index)]
        y_index = [0 for _ in xrange(batch_size)]
        batch_y_pad_sorted = [batch_y_pad[i] for i in y_reverse_sorted_index]
        for i in xrange(len(y_reverse_sorted_index)):
            y_index[y_reverse_sorted_index[i]]=i
        # Reorder the lengths
        # batch_y_pad_sorted = [batch_y_pad[i] for i in x_reverse_sorted_index]
        # batch_y_lengths_sorted = [batch_y_lengths[i] for i in x_reverse_sorted_index]



        # Generate batch_x and batch_y
        batch_x, batch_y = Variable(torch.LongTensor(batch_x_pad_sorted)), Variable(
            torch.LongTensor(batch_y_pad_sorted))
        # batch_x_lengths, batch_y_lengths = Variable(torch.LongTensor(list(reversed(sorted(batch_x_lengths))))), Variable(
        #     torch.LongTensor(list(reversed(sorted(batch_y_lengths)))))

        batch_x_lengths, batch_y_lengths = list(reversed(sorted(batch_x_lengths))), list(reversed(sorted(batch_y_lengths)))

        x_index, y_index = Variable(
            torch.LongTensor(x_index)),Variable(torch.LongTensor(y_index))
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        # batch_x_lengths=batch_x_lengths.cuda()
        # batch_y_lengths=batch_y_lengths.cuda()
        x_index=x_index.cuda()
        y_index=y_index.cuda()
        # Yield the batch UM_Corpus|
        yield batch_x, batch_y,batch_x_lengths , \
              batch_y_lengths, x_index, y_index


def data_generator_simple(data_pairs, batch_size):
    """
    Input:
        data_pairs: List of pairs, [[data_1,target_1],[data_2,target_2],...], where data_1 and target_1 are id_indexs from 1 to their own vocabulary size. The end of each instance whould end with a EOS_token index.
        batch_size: The size of the batch
    output:
        batch_x: Variable with size: B*Lx
        batch_y: Variable with size: B*Ly
        batch_x_lengths: A list witch contains the length of each source language sentence in the batch
        batch_y_lengths: A list witch contains the length of each target language sentence in the batch
        x_reverse_sorted_index: A list of index that represents the sorted batch with respect to the instance length.
    """
    data_size = len(data_pairs)
    num_batches = math.floor(data_size / batch_size)
    for i in xrange(0, data_size, batch_size):
        if i + batch_size <= data_size:
            batch_data_x = [d[0] for d in data_pairs[i:i + batch_size]]
            batch_data_y = [d[1] for d in data_pairs[i:i + batch_size]]
        else:
            batch_data_x = [d[0] for d in data_pairs[i:data_size]]
            batch_data_y = [d[1] for d in data_pairs[i:data_size]]

        # The lengths for UM_Corpus and labels to be padded to
        x_length = max([len(x) for x in batch_data_x])
        y_length = max([len(y) for y in batch_data_y])

        # Get a list of tokens
        batch_x_pad = []
        batch_x_lengths = []
        batch_y_pad = []
        batch_y_lengths = []

        # Updated batch_x_lengths, batch_x_pad
        for x_tokens in batch_data_x:
            x_l = len(x_tokens)
            x_pad_seq = pad_seq(x_tokens, x_length)
            batch_x_lengths.append(x_l)
            batch_x_pad.append(x_pad_seq)

        for y_tokens in batch_data_y:
            y_l = len(y_tokens)
            y_pad_seq = pad_seq(y_tokens, y_length)
            batch_y_lengths.append(y_l)
            batch_y_pad.append(y_pad_seq)


        # Generate batch_x and batch_y
        batch_x, batch_y = Variable(torch.LongTensor(batch_x_pad)), Variable(
            torch.LongTensor(batch_y_pad))
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        # Yield the batch UM_Corpus|
        yield batch_x, batch_y, batch_x_lengths, batch_y_lengths



def prepare_data(pair, worddict, maxlen=None, n_words=300000, test=False):
    """
    Put UM_Corpus into format useable by the model
    """
    seqs_en = []
    seqs_cn = []
    if test:
        for cc in pair[0]:
            index = []
            for w in cc.split():
                if worddict.get(w) is None or worddict.get(w) >= n_words:
                    index.append(1)
                else:
                    index.append(worddict[w])
            seqs_en.append(index)

        for cc in pair[1]:
            index = []
            for w in cc.split():
                if worddict.get(w) is None or worddict.get(w) >= n_words:
                    index.append(1)
                else:
                    index.append(worddict[w])
            seqs_cn.append(index)
    else:
        for cc in pair[0]:
            seqs_en.append([worddict[w] if worddict[w] < n_words else 1 for w in cc.split()])
        for cc in pair[1]:
            seqs_cn.append([worddict[w] if worddict[w] < n_words else 1 for w in cc.split()])

    return [[s, t] for s, t in zip(seqs_en, seqs_cn)]
