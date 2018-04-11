import numpy as np
import csv
from Share_embedder import embedder
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import xavier_weight
from sklearn.metrics import f1_score
import random

stances = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}


class News(object):
    # object for processing and presenting news to clf

    def __init__(self, stances='FNC_1/train_stances.csv', bodies='FNC_1/train_bodies.csv', embed='final'):
        # process files into arrays, etc
        self.bodies = self.proc_bodies(bodies)
        self.headlines = []
        self.embedder = embedder(embed)
        random.seed(777)
        with open(stances, 'r') as f:
            reader = csv.reader(f)
            reader.next()
            for line in reader:
                if len(line) == 2:
                    hl, bid = line
                    stance = 'unrelated'
                else:
                    hl, bid, stance = line
                self.headlines.append((hl, bid, stance))
        random.shuffle(self.headlines)
        self.n_headlines = len(self.headlines)

    def get_one(self, ridx=None):
        # select a single sample either randomly or by index
        if ridx is None:
            ridx = np.random.randint(0, self.n_headlines)
        head = self.headlines[ridx]
        body = self.bodies[head[1]]

        return head, body

    def sample(self, n=16, ridx=None):
        # select a batch of samples either randomly or by index
        heads = []
        bodies = []
        stances_d = []
        if ridx is not None:
            for r in ridx:
                head, body_text = self.get_one(r)
                head_text, _, stance = head
                heads.append(head_text)
                bodies.append(body_text)
                stances_d.append(stances[stance])
        else:
            for i in xrange(n):
                head, body_text = self.get_one()
                head_text, _, stance = head
                heads.append(head_text)
                bodies.append(body_text)
                stances_d.append(stances[stance])

        heads, bodies = self.embedder.embed(heads, bodies)

        stances_d = Variable(torch.LongTensor(np.asarray(stances_d, dtype='int32'))).cuda()

        return heads, bodies, stances_d

    def validate(self):
        # iterate over the dataset in order
        for i in xrange(len(self.headlines)):
            yield self.sample(ridx=[i])

    def proc_bodies(self, fn):
        # process the bodies csv into arrays
        tmp = {}
        with open(fn, 'r') as f:
            reader = csv.reader(f)
            reader.next()
            for line in reader:
                bid, text = line
                tmp[bid] = text
        return tmp


# Classification model
class StanceClf(torch.nn.Module):
    def __init__(self, hidden_dim, output_size):
        super(StanceClf, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout1 = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_size)
        self.dropout2 = nn.Dropout(p=0.5)
        self.init_weights()

    def forward(self, batch_x, batch_y):
        # combined = torch.cat((inputs, hidden), 1)
        x = torch.cat([batch_x, batch_y], dim=1)
        x = self.dropout1(x)
        x = nn.ReLU()((self.linear1(x)))
        x = nn.ReLU()((self.linear2(x)))
        x = nn.ReLU()((self.linear3(x)))
        x = F.log_softmax(x, dim=1)
        return x

    def init_weights(self):
        xavier_weight(self.linear1.weight)
        self.linear1.bias.data.fill_(0)
        xavier_weight(self.linear2.weight)
        self.linear1.bias.data.fill_(0)
        xavier_weight(self.linear3.weight)
        self.linear1.bias.data.fill_(0)


def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
    category_i = top_i[:, 0].cpu().numpy()  # print(stance.size(0))
    return category_i


def train():
    run = 'final_batch_32'
    data = 'FNC_1'
    saveto = 'save_fnc/%s' % (data)


    # all_categories = [0, 1, 2, 3]
    output_size = 4
    embed = 'final'
    sent_dim = 300
    hidden_dim = sent_dim * 2
    batch_size = 32
    learn_rate = 0.001
    grad_clip = 2.0
    max_epoch = 100
    validFreq = 200
    dispFreq = 50
    early_stop = 20


    model = StanceClf(hidden_dim, output_size)
    model = model.cuda()

    params = filter(lambda p: p.requires_grad, model.parameters())
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)


    # FNC Baseline training set
    news = News(stances='FNC_1/train.csv', bodies='FNC_1/train_bodies.csv', embed=embed)
    # FNC baseline validation set
    val_news = News(stances='FNC_1/test.csv', bodies='FNC_1/train_bodies.csv', embed=embed)

    test_news = News(stances='FNC_1/competition_test_stances.csv',
                     bodies='FNC_1/competition_test_bodies.csv', embed=embed)
    print news.n_headlines
    # min_num=min(val_news.n_headlines, test_news.n_headlines)
    # print min_num
    vh, vb, vs = val_news.sample(1000)
    th, tb, ts = test_news.sample(1000)

    curr = 0
    uidx=0
    # For Early-stopping
    best_step = 0
    print 'number of training samples: ', news.n_headlines
    for iepx in xrange(1, max_epoch+1):
        for _ in xrange(1, news.n_headlines / batch_size + 2):
            uidx+=1
            head, body, stance = news.sample(n=batch_size)

            pred = model(head, body)

            loss = loss_function(pred, stance)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(params, grad_clip)
            optimizer.step()

            if np.mod(uidx, dispFreq) == 0:
                print 'Epoch ', iepx, '\tUpdate ', uidx, '\tCost ', loss.data.cpu().numpy()[0]

            if np.mod(uidx, validFreq) == 0:
                #compute dev
                out = model.forward(vh, vb)
                pred = categoryFromOutput(out)
                score = f1_score(vs.data.cpu().numpy(), pred, average='macro')
                # compute test
                out = model.forward(th, tb)
                pred = categoryFromOutput(out)
                test_score = f1_score(ts.data.cpu().numpy(), pred, average='macro')


                curr_step = uidx / validFreq

                currscore = score
                print 'F1 on dev', score
                print 'F1 on test', test_score
                if currscore > curr:
                    curr = currscore
                    # best_r1, best_r5, best_r10, best_medr = r1, r5, r10, medr
                    # best_r1i, best_r5i, best_r10i, best_medri = r1i, r5i, r10i, medri
                    best_step = curr_step

                    # Save model
                    print 'Saving model...',
                    torch.save(model.state_dict(), '%s_model_%s.pkl' % (saveto, run))
                    print 'Done'

                if curr_step - best_step > early_stop:
                    print 'Early stopping ...'
                    # print "cn to en: %.1f, %.1f, %.1f, %.1f" % (best_r1, best_r5, best_r10, best_medr)
                    # print "en to cn: %.1f, %.1f, %.1f, %.1f" % (best_r1i, best_r5i, best_r10i, best_medri)
                    return


if __name__ == '__main__':
    train()
