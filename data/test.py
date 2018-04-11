import train
from model import LIUMCVC_Encoder
import torch
from evaluation import evalrank
import cPickle as pkl
import torch.nn as nn

data = 'UM_Corpus'
saveto = 'save/%s' % (data)
run='final'

def test(run=run):
    with open('%s_params_%s.pkl' % (saveto, run), 'rb') as f:
        model_options = pkl.load(f)
    with open('%s.dictionary_%s.pkl' % (saveto, run), 'rb') as f:
        worddict = pkl.load(f)
    model = LIUMCVC_Encoder(model_options)
    model.load_state_dict(torch.load('%s_model_%s.pkl' % (saveto, run)))
    model = model.cuda()

    best_model = {}
    best_model['options'] = model_options
    best_model['en_cn_model'] = model
    best_model['worddict'] = worddict

    evalrank(best_model, data, split='dev')
    evalrank(best_model, data, split='test')


if __name__ == "__main__":
    test()
