from Share_embedder import embedder
from Manager import manager


if __name__ == '__main__':
    emb = embedder(run='final')
    emb.embed_baidu()
    emb.embed_twitter()
    emb.embed_google()

    # compute distance and stance
    m = manager()
    m.save_twitter_processed()
    m.save_baidu_processed()
