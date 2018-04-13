# Cross-Lingual Cross-Platform Rumor Verification Pivoting on Multimedia Content

# Overview

This repository contains the implementation of methods in "Cross-Lingual Cross-Platform Rumor Verification Pivoting on Multimedia Content".

# Library Dependencies
  - Python 2.7
  - Pytorch
  - scikit-learn
  - Theano
  - Keras (with Theano backend)
  - Pandas
  - ...

# Data
All data used in this project is save in the 'data' folder. The original Twitter dataset from [VMU 2015](https://github.com/MKLab-ITI/image-verification-corpus) is saved as ‘resources/dataset.txt’ in json format. The additional data we collected from Google and Baidu from VMU 2016 is saved as ‘google_results.txt’ and 'baidu_results.txt'.


# Procedure
1.	Download parallel English and Mandarin sentence of news and microblogs from [UM-Corpus](http://nlp2ct.cis.umac.mo/um-corpus/index.html) and save them in a folder named 'UM_Corpus' in 'data'.

2.	Run load_corpus.py in data to split and tokenize the data in UM-Corpus.

3.	Run train.py in data to train the multilingual embedding.

4.	Run train_fnc.py in data to train the agreement classifier with the dataset from [Fake News Challenge](http://www.fakenewschallenge.org/).

5.  Run embed_data.py in data to compute the cosine distance and agreement between target rumors and titles of their related sources.

6.	Run VMU2015.py to reproduce the results in the task and event settings.

7.	Run Transfer_Baidu.py to reproduce the results in transfer learning.
