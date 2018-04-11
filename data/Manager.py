import json
from urlparse import urlparse
import urllib2
from scipy.spatial.distance import cdist, cosine
from Stancer import stancer
import numpy as np
from numpy import einsum


class manager():
    def __init__(self, twitter='twitter_embed.txt', baidu='baidu_embed.txt',
                 google='google_embed.txt'):
        self.twitter = twitter
        self.baidu = baidu
        self.google = google
        self.stancer = stancer('final')

    def save_twitter_processed(self):
        with open(self.twitter) as f1, open(self.google) as f2:
            dataset = json.load(f1)
            google = json.load(f2)

        for tweet in dataset:
            if tweet['label'] == 'fake':
                tweet['label'] = 1
            elif tweet['label'] == 'real':
                tweet['label'] = 0
            else:
                print('error!')
                return

            google_results = []
            for id in tweet['image_id']:
                google_results += google[id]
            # title
            head = [source['embed'] for source in google_results]
            body = [tweet['embed'] for _ in google_results]

            if not head:
                tweet['agree'] = []
                tweet['dist'] = []
            else:
                tweet['dist'] = ((np.ones(len(head)) - einsum('ij,ij->i', head, body)) * 1).tolist()
                tweet['agree'] = self.stancer.compute_stance(head, body)

            # #body
            # head = [source['bembed'] for source in google_results]
            # body = [tweet['bembed'] for _ in google_results]
            #
            # if not head:
            #     tweet['bagree'] = []
            #     tweet['bdist'] = []
            # else:
            #     tweet['bdist'] = ((np.ones(len(head)) - einsum('ij,ij->i', head, body)) * 1).tolist()
            #     tweet['bagree'] = self.stancer.compute_stance(head, body)

        with open('processed_twitter.txt', 'w') as f:
            json.dump(dataset, f)

    def save_baidu_processed(self):
        with open(self.baidu) as f1, open(self.google) as f2:
            baidu = json.load(f1)
            google = json.load(f2)
            dataset = []

            # build and index Baidu results
            index = {}
            n = 0
            for image_id in baidu:
                for s in baidu[image_id]:
                    if s['url'] not in index and s['annotation'] in [0, 1]:
                        post = {}
                        post['index'] = n
                        index[s['url']] = n
                        n += 1
                        #title
                        head = [source['embed'] for source in google[image_id][:20]]
                        body = [s['embed'] for _ in google[image_id][:20]]

                        if not head:
                            post['agree'] = []
                            post['dist'] = []
                        else:
                            post['dist'] = ((np.ones(len(head)) - einsum('ij,ij->i', head, body)) * 0.5).tolist()
                            post['agree'] = self.stancer.compute_stance(head, body)

                        # #body
                        # head = [source['bembed'] for source in google[image_id][:20]]
                        # body = [s['bembed'] for _ in google[image_id][:20]]
                        #
                        # if not head:
                        #     post['bagree'] = []
                        #     post['bdist'] = []
                        # else:
                        #     post['bdist'] = ((np.ones(len(head)) - einsum('ij,ij->i', head, body)) * 0.5).tolist()
                        #     post['bagree'] = self.stancer.compute_stance(head, body)

                        post['label'] = s['annotation']
                        post['content'] = s['title']
                        event = image_id.split('_')[0]
                        if event in ['sandyA', 'sandyB']:
                            event = 'sandy'
                        post['event'] = event
                        s['event'] = event

                        dataset.append(post)
        # with open(self.baidu, 'w') as f:
        #     json.dump(baidu, f)

        with open('processed_baidu.txt', 'w') as f:
            json.dump(dataset, f)


class WOT():
    def __init__(self):
        self.key = '0c8da812e5dad1c8ff1e0de62a6a8ddadb45fa85'

    def rank(self, link):
        host = urlparse(link).hostname
        if host is None:
            print(link)
            return "NaN"
        url = ' http://api.mywot.com/0.4/public_link_json2?' \
              'hosts=' + host + '/&key' \
                                '=' + self.key
        try:
            response = urllib2.urlopen(url).read()
            res = json.loads(response)
            valueTrust = int(res[host]['0'][0])
            confTrust = int(res[host]['0'][1])
            value = valueTrust * confTrust / float(100)
            return value
        except:
            return "NaN"


if __name__ == '__main__':
    pass
