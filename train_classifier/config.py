# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 10:55:06 2014

@author: kensuke-mi
"""

#! /usr/bin/python
# -*- coding:utf-8 -*-
__date__='2014/3/1';

import argparse, codecs, os, glob, json, sys;
sys.path.append('../');
import return_range, mulan_module, liblinear_module, bigdoc_module;
import feature_create;
import original_dutch_module;
import file_load_module;
from nltk.corpus import stopwords;
from nltk import stem;
from nltk import tokenize; 
from nltk.stem import SnowballStemmer;
stemmer=SnowballStemmer("dutch");
#------------------------------------------------------------
lemmatizer = stem.WordNetLemmatizer();
stopwords = stopwords.words('english');
symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')'];
#------------------------------------------------------------
#option parameter
level=1;
dev_limit=1;
#Idea number of TFIDF
tfidf_idea=2;        
#------------------------------------------------------------
#Initialize dfd_training_map with 23 labels
alphabetTable=[unichr(i) for i in xrange(65, 91)if chr(i) not in [u'I',u'O',u'Y']]
#------------------------------------------------------------
dfd_dir_path='../training_resource/dfd/';
tmi_dir_path='../training_resource/tmi/';
dfd_orig_path='../training_resource/dfd_orig/';
additional_resource_list=['../../american_indian_corpus/tagged_corpus/'];
#------------------------------------------------------------