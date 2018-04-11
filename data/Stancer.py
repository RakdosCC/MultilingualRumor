import torch
import json
from torch.autograd import Variable
from train_fnc import StanceClf
import numpy as np

class stancer:
    def __init__(self, run):
        data = 'FNC_1'
        path = 'save_fnc/%s' % (data)
        model = StanceClf(600, 4)
        model.load_state_dict(torch.load('%s_model_%s.pkl' % (path, run)))
        model = model.cuda()

        self.model = model

    def compute_stance(self, head, body):
        head = Variable(torch.FloatTensor(head)).cuda()
        body = Variable(torch.FloatTensor(body)).cuda()
        out = self.model.forward(head, body)
        return np.exp(out.data.cpu().numpy()).tolist()
