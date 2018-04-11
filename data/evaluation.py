import numpy
import torch
from load_corpus import load_dataset
from tools import encode_sentences
from torch.autograd import Variable
from model import PairwiseRankingLoss


def evalrank(model, data, split='test'):
    """
    Evaluate a trained model on either dev or test
    """

    print ('Loading dataset')
    if split == 'dev':
        _ , X = load_dataset(data)
    else:
        X = load_dataset(data, load_test=True)

    print ('Computing results...')
    en, cn = encode_sentences(model, X, test=True)

    score = devloss(en, cn, margin=model['options']['margin'])

    print(split+' loss: ', score)
    # (r1, r5, r10, medr) = i2t(cn, en)
    # print ("cn to en: %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr))
    # (r1i, r5i, r10i, medri) = t2i(cn, en)
    # print ("en to cn: %.1f, %.1f, %.1f, %.1f" % (r1i, r5i, r10i, medri))


#
def devloss(en, cn, margin=0.2):
    scores = torch.mm(cn, en.transpose(1, 0))
    diagonal = scores.diag()

    # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
    cost_s = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
                       (margin - diagonal).expand_as(scores) + scores)
    # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
    cost_im = torch.max(Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
                        (margin - diagonal).expand_as(scores).transpose(1, 0) + scores)

    for i in xrange(scores.size()[0]):
        cost_s[i, i] = 0
        cost_im[i, i] = 0

    return (cost_s.sum() + cost_im.sum()).data.cpu().numpy()[0]



def i2t(images, captions, npts=None):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        npts = images.size()[0] / 5

    ranks = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].unsqueeze(0)

        # Compute scores
        d = torch.mm(im, captions.t())
        d_sorted, inds = torch.sort(d, descending=True)
        inds = inds.data.squeeze(0).cpu().numpy()

        # Score
        rank = 1e20
        # find the highest ranking
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)


def t2i(images, captions, npts=None, data='f8k'):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts == None:
        npts = images.size()[0] / 5

    ims = torch.cat([images[i].unsqueeze(0) for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index: 5 * index + 5]

        # Compute scores
        d = torch.mm(queries, ims.t())
        for i in range(d.size()[0]):
            d_sorted, inds = torch.sort(d[i], descending=True)
            inds = inds.data.squeeze(0).cpu().numpy()
            ranks[5 * index + i] = numpy.where(inds == index)[0][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)
    medr = numpy.floor(numpy.median(ranks)) + 1
    return (r1, r5, r10, medr)
