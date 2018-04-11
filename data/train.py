# -*- coding: utf-8 -*-

import torch
import os
import cPickle as pkl
from load_corpus import load_dataset
from vocab import build_dictionary
from preprocessing import prepare_data, data_generator
from torch.autograd import Variable
import time
from model import PairwiseRankingLoss, LIUMCVC_Encoder
import numpy
from tools import encode_sentences
from evaluation import i2t, t2i, devloss
from torch.optim.lr_scheduler import ReduceLROnPlateau

# os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5, 6, 7'
data = 'UM_Corpus'
saveto = 'save/%s' % (data)
run = 'final'


def train(
        margin=0.2,
        dim=300,
        dim_word=300,
        max_epochs=100,
        dispFreq=50,
        validFreq=200,
        grad_clip=2.0,
        maxlen_w=150,
        batch_size=300,
        early_stop=20,
        lrate=0.001,
        reload_=False,
        load_dict=False
):
    # Model options
    model_options = {}
    model_options['UM_Corpus'] = data
    model_options['margin'] = margin
    model_options['dim'] = dim
    model_options['dim_word'] = dim_word
    model_options['max_epochs'] = max_epochs
    model_options['dispFreq'] = dispFreq
    model_options['grad_clip'] = grad_clip
    model_options['maxlen_w'] = maxlen_w
    model_options['batch_size'] = batch_size
    model_options['saveto'] = saveto
    model_options['validFreq'] = validFreq
    model_options['lrate'] = lrate
    model_options['reload_'] = reload_

    print (model_options)

    # reload options
    if reload_ and os.path.exists(saveto):
        print ('reloading...' + saveto)
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    # Load training and development sets
    print ('loading dataset')
    train, dev = load_dataset(data)
    test = load_dataset(data, load_test=True)
    if load_dict:
        with open('%s.dictionary.pkl' % saveto, 'rb') as f:
            worddict = pkl.load(f)
            n_words = len(worddict)
            model_options['n_words'] = len(worddict)
    else:
        # Create and save dictionary
        print ('Create dictionary')
        worddict = build_dictionary(train[0] + train[1] + dev[0] + dev[1])
        n_words = len(worddict)
        model_options['n_words'] = n_words
        print ('Dictionary size: ' + str(n_words))
        with open('%s.dictionary_%s.pkl' % (saveto, run), 'wb') as f:
            pkl.dump(worddict, f)

    # # Inverse dictionary
    # word_idict = dict()
    # for kk, vv in worddict.iteritems():
    #     word_idict[vv] = kk
    # word_idict[0] = '<eos>'
    # word_idict[1] = 'UNK'

    model_options['worddict'] = worddict
    # model_options['word_idict'] = word_idict

    # # Each sentence in the minibatch have same length (for encoder)
    # train_iter = HomogeneousData([train[0], train[1]], batch_size=batch_size, maxlen=maxlen_w)

    share_model = LIUMCVC_Encoder(model_options)
    # gpus = [0, 1, 2, 3]
    # share_model = torch.nn.DataParallel(share_model, device_ids=gpus)
    share_model = share_model.cuda()

    loss_fn = PairwiseRankingLoss(margin=margin)
    loss_fn = loss_fn.cuda()

    params = filter(lambda p: p.requires_grad, share_model.parameters())
    optimizer = torch.optim.Adam(params, lrate)

    # decrease learning rate
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    uidx = 0
    curr = 1e10
    n_samples = 0

    # For Early-stopping
    best_step = 0

    for eidx in xrange(1, max_epochs + 1):

        print('Epoch ', eidx)

        train_data_index = prepare_data(train, worddict, n_words)
        for en, cn, en_lengths, cn_lengths, en_index, cn_index in data_generator(train_data_index, batch_size):
            uidx += 1
            n_samples += len(en)
            en, cn = share_model(en, en_lengths, en_index, cn, cn_lengths, cn_index)

            loss = loss_fn(en, cn)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(params, grad_clip)
            optimizer.step()

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, '\tUpdate ', uidx, '\tCost ', loss.data.cpu().numpy()[0]

            if numpy.mod(uidx, validFreq) == 0:

                print 'Computing results...'
                curr_model = {}
                curr_model['options'] = model_options
                curr_model['worddict'] = worddict
                # curr_model['word_idict'] = word_idict
                curr_model['en_cn_model'] = share_model

                r_time = time.time()
                fen, fcn = encode_sentences(curr_model, dev)
                score = devloss(fen, fcn, margin=margin)
                fen, fcn = encode_sentences(curr_model, test, test=True)
                test_score = devloss(fen, fcn, margin=margin)

                print "Cal Recall@K using %ss" % (time.time() - r_time)

                curr_step = uidx / validFreq

                #scheduler.step(score)

                currscore = score
                print 'loss on dev', score
                print 'loss on test', test_score
                if currscore < curr:
                    curr = currscore
                    # best_r1, best_r5, best_r10, best_medr = r1, r5, r10, medr
                    # best_r1i, best_r5i, best_r10i, best_medri = r1i, r5i, r10i, medri
                    best_step = curr_step

                    # Save model
                    print 'Saving model...',
                    pkl.dump(model_options, open('%s_params_%s.pkl' % (saveto, run), 'wb'))
                    torch.save(share_model.state_dict(), '%s_model_%s.pkl' % (saveto, run))
                    print 'Done'

                if curr_step - best_step > early_stop:
                    print 'Early stopping ...'
                    # print "cn to en: %.1f, %.1f, %.1f, %.1f" % (best_r1, best_r5, best_r10, best_medr)
                    # print "en to cn: %.1f, %.1f, %.1f, %.1f" % (best_r1i, best_r5i, best_r10i, best_medri)
                    return

        print 'Seen %d samples' % n_samples


if __name__ == '__main__':
    train()
