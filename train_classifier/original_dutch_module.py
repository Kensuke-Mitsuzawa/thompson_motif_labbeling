#! /usr/bin/python
# -*- coding:utf-8 -*-
__date__='2014/1/22';

"""
オランダ語のままの時に，分類性能がどうなるのか．を確かめるために実験をする．
このスクリプトでは，オランダ語文書を読み込む処理をする
"""
import os,sys,codecs;
import construct_bigdoc_or_classifier
import nltk;
#オランダ語で記述されたファイルの置き場
original_dutch_document_dir='../../dutch_folktale_corpus/given_script/top_dutch/top_document_train/';

def tokenize(filepath):
    file_obj=codecs.open(filepath,'r','utf-8'); 
    document_unicode=file_obj.read(); 
    tokenized_document=nltk.tokenize.wordpunct_tokenize(document_unicode);
    file_obj.close(); 
    return tokenized_document;

def make_document_set(file_list,label_map):
    """
    RETURN map label_map {unicode label: list documents [list document [unicode token]]}
    """
    for filepath in file_list:
        #ラベルの分解処理
        label_list=(os.path.basename(filepath)).split('_')[:-1];
        for label in label_list:
            #tokenized_documentはリスト型
            tokenized_document=tokenize(filepath);
            #オランダ語はわからんが，一応すべて小文字化はしておく
            tokenized_document=[t.lower() for t in tokenized_document];
            if label not in label_map:
                label_map[label]=[tokenized_document];
            else:
                label_map[label].append(tokenized_document);
    return label_map;

def main(label_map):
    filelist=construct_bigdoc_or_classifier.make_filelist(original_dutch_document_dir);
    label_map=make_document_set(filelist,label_map);

    return label_map;

if __name__=='__main__':
    label_map={};
    main(label_map);
