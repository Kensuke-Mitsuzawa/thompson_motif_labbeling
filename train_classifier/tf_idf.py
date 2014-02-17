#! /usr/bin/python
# -*- coding:utf-8 -*-
__author__='Kensuke Mitsuzawa';
__data__='2014/1/23';
import nltk, json, codecs, sys, re, os, glob, math;
take_log=True;
debug=False;

#------------------------------------------------------------
#このコードは間違っている．
#tfidfの計算時に，文書集合全体が引数に与えられてしまっている．
#tfidfの引数は文書集合内の１文書dとt in dなので，上の方針が間違い
#が，もしかしてこの方が精度がいい，ということもあり得なくもないので，残しておく
def tf_idf_test(docs):  
    documents=[];
    for one_doc in docs:
        doc=[]; 
        for t in one_doc:
            try:
                doc.append(t.encode('ascii'));
            except UnicodeEncodeError:
                pass;
        documents.append(doc);
    tf_idf_score={};
    tokens = [];  
    for doc in documents:  
        tokens += doc  
    A = nltk.TextCollection(documents)  
    token_types = set(tokens)  
    for token_type in token_types:
        if not A.tf_idf(token_type, tokens)==0: 
            if take_log==True:
                tf_idf_score[token_type]=math.log(A.tf_idf(token_type, tokens));
            else:
                tf_idf_score[token_type]=A.tf_idf(token_type, tokens);
        else:
            tf_idf_score[token_type]=0;
    return tf_idf_score;

#------------------------------------------------------------
#確認用のコード（他人が書いた）
#http://nktmemo.wordpress.com/2013/07/14/tfidf%E3%81%A7%E3%82%AD%E3%83%BC%E3%83%AF%E3%83%BC%E3%83%89%E6%8A%BD%E5%87%BA%E3%83%AD%E3%82%A4%E3%82%BF%E3%83%BC%E3%82%B3%E3%83%BC%E3%83%91%E3%82%B9/
def tf(word, words, lower=False):
    if lower:
        words = [word.lower() for word in words]
        word = word.lower()
    if words:
        return float(words.count(word)) / len(words)
    return 0.0

def df(word, docs, lower=False):
    if lower:
        docs = [[word.lower() for word in doc] for doc in docs]
        word = word.lower()
    return float(sum([word in doc for doc in docs])) / len(docs)

def idf(word, docs, lower=False):
    if lower:
        docs = [[word.lower() for word in doc] for doc in docs]
        word = word.lower()
    df_val = df(word, docs)
    if df_val == 0.0:
        return 0.0
    return math.log(1.0 / df_val)

def tfidf_else(word, doc, docs, lower=False):
    return tf(word, doc, lower=lower) * idf(word, docs, lower=lower)

def main_else(document_set):
    doc=document_set[0];
    all_words=[t for doc in document_set for t in doc];
    for word in all_words:
        w_dt=tfidf_else(word,doc,document_set);
    
#------------------------------------------------------------
#自分で書いたコード
def IDF(t,N,df_map):
    numerator=N;
    denominator=df_map[t];
    IDF_score=numerator/denominator;
    if IDF_score==0:
        pass;
    else:
        IDF_score=math.log(IDF_score);
    return IDF_score;

def TF(t,d):
    denominator=len(d);
    numerator=float(d.count(t));
    return numerator/denominator;

def TF_nishimura(t,d):
    denominator=len([sub_document for sub_document in d]);
    frequency_of_unique_d=0; 
    for sub_document in d:
        if t in sub_document:
            frequency_of_unique_d+=1;
    TF_nishimura_score=float(frequency_of_unique_d)/denominator;
    
    return TF_nishimura_score;

def df_mine(documents):
    doc_num_having_query={};
    for label_documents in documents:
        for doc in label_documents:
            for token in doc:
                if token in doc_num_having_query:
                    doc_num_having_query[token]+=1;
                else:
                    doc_num_having_query[token]=1;
    return doc_num_having_query;

def tfidf_mine(t,d,docs,N,df_map):
    w_dt=TF(t,d)*IDF(t,N,df_map);
    return w_dt;

def main(document_set,document_index,N,df_map):
    """
    RETURN map w_dt_map {unicode token: float w_dt}
    """
    w_dt_map={};
    d=document_set[document_index]
    for t in df_map:
        w_dt=tfidf_mine(t,d,document_set,N,df_map);
        if take_log==True:
            if w_dt==0:
                w_dt_map[t]=0.0;
            else:
                w_dt_map[t]=math.log(w_dt);
        else:
            w_dt_map[t]=w_dt;
    return w_dt_map;

def tf_idf_interface(document_set):
    """
    RETURN list w_dt_maps_list [map w_dt_map {unicode token: float w_dt} ]
    """
    w_dt_maps_list=[];
    N=len(document_set);
    df_map=df_mine(document_set);
    for document_index in range(0, len(document_set)):
        w_dt_map=main(document_set,document_index,N,df_map);
        w_dt_maps_list.append(w_dt_map);
    return w_dt_maps_list;

def tfidf_nishimura(t,d,document_set,N,df_map):
    idf_score=IDF(t,N,df_map);
    tf_nishimura_score=TF_nishimura(t,d);
    w_dt=tf_nishimura_score*idf_score;
    
    if debug==True:
        print 'token:',t;
        #print 'd:',d;
        print 'TF score:',tf_nishimura_score;
        print 'IDF score:',idf_score;
        print '-'*30;
    return w_dt;

def main_nishimura(document_set_for_tf,document_set,document_index,N,df_map):
    """
    INPUT: list document_set[list document [list sub-document[unicode token]]]
    RETURN map w_dt_map {unicode token: float w_dt}
    """
    w_dt_map={};
    d_for_tf=document_set_for_tf[document_index];
    for t in df_map:
        w_dt=tfidf_nishimura(t,d_for_tf,document_set,N,df_map);
        if take_log==True:
            if w_dt==0:
                w_dt_map[t]=0.0;
            else:
                w_dt_map[t]=math.log(w_dt);
        else:
            w_dt_map[t]=w_dt;
    return w_dt_map;

def tf_idf_nishimura_interface(document_set_for_tf):
    """
    RETURN list w_dt_maps_list [map w_dt_map {unicode token: float w_dt} ]
    """
    #IDF計算は通常と同じなので，次元数を下げておく
    document_set=[t for sub_document in document_set_for_tf for t in sub_document];
    df_map=df_mine(document_set);
    N=len(document_set);
    w_dt_maps_list=[];
    iter_number=0;
    for label_document in document_set_for_tf:
        iter_number+=1;
    for document_index in range(0,iter_number):
        w_dt_map=main_nishimura(document_set_for_tf,document_set,document_index,N,df_map);
        w_dt_maps_list.append(w_dt_map);
    
    return w_dt_maps_list;

if __name__=='__main__':
    document_set=["The demands from Beijing have resulted in tensions with Japan and the United States.",
                  "On Saturday, United, American and Delta airlines told CNN that its pilots were following Washington's advice and complying with Beijing's 'air defense identification zone."];
    document_set_for_nishimura=[
        ["At that time Tunisians, full of high hopes and incited by the degradation of their living conditions, took to the streets and demonstrated peacefully for real change in their country. For us, it wasn't an Arab Spring but a Dignity Revolution.","But how have things changed since then? Are there real efforts to bring democracy? Are we really experiencing a democratic transition?"],
        ["As Russian forces hunt for a black widow suicide bomber who may have infiltrated Sochi, the mounting threats of a terrorist attack at the games has weighed heavily on the minds of would-be spectators.","Dan Fredricks said it will be bittersweet not being able to watch his son compete."]
    ]
    document_set=[sentence.split() for sentence in document_set];
    document_set_for_nishimura=[[sub_document.split() for sub_document in document] for document in document_set_for_nishimura];
   
     
    main_else(document_set);
    N=len(document_set);
    df_map=df_mine(document_set);
    print main(document_set,0,N,df_map);

    main_nishimura(document_set_for_nishimura,0)
