import json
from Tokenizer import tokenizer

p = tokenizer()



#
# with open('google_results.txt','w') as f:
#     json.dump(google, f)

# with open('baidu_results.txt') as f:
#     baidu = json.load(f)
#
# for id in baidu:
#     for s in baidu[id]:
#         s['title'] = p.tokenize(s['title'])
#
# with open('baidu_results.txt', 'w') as f:
#     json.dump(baidu, f)

def num():
    with open('baidu_results.txt') as f:
        baidu = json.load(f)

    with open('google_results.txt') as f:
        google = json.load(f)

    with open('dataset.txt') as f:
        twitter = json.load(f)

    events = ['sandy', 'boston', 'sochi', 'malaysia', 'bringback', 'columbianChemicals', 'passport', 'elephant',
              'underwater', 'livr', 'pigFish', 'eclipse', 'samurai', 'nepal', 'garissa', 'syrianboy', 'varoufakis']
    k = set([])
    for event in events:
        print event
        num = 0
        fake = 0
        real = 0
        wb = set([])
        for tweet in twitter:
            if tweet['event'] == event:
                num += 1
                if tweet['label'] == 'fake':
                    fake += 1
                else:
                    real += 1
                for id in tweet['image_id']:
                    for s in google[id]:
                        wb.add(s['index'])
                        k.add(s['index'])

        print 'num ', num
        print 'fake', fake
        # print 'real', real
        print 'wb', len(wb)

    print len(k)


if __name__ == '__main__':
    # with open('dataset.txt') as f:
    #     twitter = json.load(f)
    #
    # for tweet in twitter:
    #     tweet['content']=p.tokenize(tweet['content'])
    #
    # with open('dataset.txt','w') as f:
    #     json.dump(twitter, f)
    # num()
    # events = ['sandy', 'boston', 'sochi', 'malaysia', 'bringback', 'columbianChemicals', 'passport', 'elephant',
    #           'underwater', 'livr', 'pigFish', 'eclipse', 'samurai', 'nepal', 'garissa', 'syrianboy', 'varoufakis']
    #
    # with open('processed_baidu.txt') as f:
    #     baidu=json.load(f)
    #
    # for event in events:
    #     real, fake = 0, 0
    #     for s in baidu:
    #         if s['event']==event:
    #             if s['label'] == 0:
    #                 real+=1
    #             elif s['label'] == 1:
    #                 fake+=1
    #     print event
    #     print 'real ', real
    #     print 'fake', fake

    with open('google_results.txt') as f:
        google = json.load(f)

    index = {}
    n = 0
    for id in google:
        for s in google[id]:
            s['title'] = p.tokenize(s['title'])
            if s['url'] not in index:
                index[s['url']] = n
                n += 1
                s['index'] = n
            else:
                s['index'] = index[s['url']]

    print n
