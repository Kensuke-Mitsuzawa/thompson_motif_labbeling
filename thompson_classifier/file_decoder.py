#! /usr/bin/python
# -*- coding:utf-8 -*-
__date__='2014/03/01';

import subprocess, argparse, os, glob, sys, pickle, json, codecs;
sys.path.append('../get_thompson_motif/');
import call_liblinear,call_mulan;
import document_based as doc_based;
import sentence_based as sent_based;
from nltk import tokenize;
from nltk import stem;
from nltk.corpus import stopwords;
lemmatizer = stem.WordNetLemmatizer();
stopwords = stopwords.words('english');
symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')']

def load_feature_map(feature_map_path):
    with codecs.open(feature_map_path, 'r', 'utf-8') as f:
        feature_map=json.load(f);
    return feature_map;

def load_files(classifier_dir, suffix=u''):
    classifier_path_list=[];
    for root, dirs, files in os.walk(classifier_dir):
        for f in glob.glob(os.path.join(root, '*'+suffix)):
            classifier_path_list.append(f);
    return classifier_path_list;


def unify_stack(tokens_stack, motif_stack):
    return (motif_stack, tokens_stack);

def conv_to_featurespace(test_data_list, feature_map_character, feature_map_numeric, tfidf_score_map, args):
    """
    testデータを素性空間に変換する．
    ARGSのうち，test_data_listは tuple (list [unigram モチーフ], list [unigram token])
    RETURN (list [unigram モチーフ], list [(int 素性番号, float ベクトルの大きさ)])
    """
    doc_in_feature_space=[];
    for token in test_data_list[1]:
        if args.tfidf==True:
            if token in tfidf_score_map:
                tfidf_score=tfidf_score_map[token];
                for feature_candidate in feature_map_character[token]:
                    feature_number=feature_map_numeric[feature_candidate];
                    doc_in_feature_space.append((feature_number, tfidf_score));
        elif args.tfidf==False:
            #unigramの場合
            if token in feature_map_character:
                for feature_candidate in feature_map_character[token]:
                    feature_number=feature_map_numeric[feature_candidate];
                    doc_in_feature_space.append((feature_number, 1));
    return (test_data_list[0], doc_in_feature_space);

def out_libsvm_format_sentence(sentences_in_document, feature_map_character, feature_map_numeric, feature_show, tfidf_flag):
    """
    文ごとに作成したインスタンスをファイルに書き出す
    OUT:libsvm_formatのファイル
    """
    out_file=codecs.open('./test.data', 'w', 'utf-8');
    for sentence in sentences_in_document:
        #======================================== 
        #一文ごとにインスタンスの書き出し処理
        one_instance_stack=[];
        for token in sentence:
            if tfidf_flag==True:
                if token in feature_map_character:
                    feature_character=feature_map_character[token][0];
                    tfidf_score=feature_character.split(u'_')[2];
                    token=feature_character.split(u'_')[1];
                    feature_number=feature_map_numeric[feature_character];
                    if feature_show==True:
                        print token, tfidf_score;
                    one_instance_stack.append((feature_number, tfidf_score));
            elif tfidf_flag==False: 
                if token in feature_map_character:
                    for feature_character in feature_map_character[token]:
                        if feature_show==True:
                            print feature_character;
                        one_instance_stack.append((feature_map_numeric[feature_character], 1));
        #----------------------------------------  
        #ファイルへの書き出し
        one_instance_stack.sort();
        if not one_instance_stack==[]:
            out_file.write('+1 ');
            for tuple_item in list(set(one_instance_stack)):
                if not tuple_item==None:
                    out_file.write(u'{}:{} '.format(tuple_item[0], tuple_item[1]))
            out_file.write(u'\n');
        #----------------------------------------  
        #======================================== 
    out_file.close();
    if feature_show==True:
        print u'-'*30;

def eval_with_arow():
    model_dir_path='../get_thompson_motif/classifier/libsvm_format/';
    model_path_list=load_files(model_dir_path, '.model') 
    result_map={};
    for model_file in model_path_list:
        alphabet_label=os.path.basename(model_file)[0];
        p=subprocess.Popen(['arow_test', 'test.data', model_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE);
        for line in p.stdout.readlines():
            result_map[alphabet_label]=line.split(u' ')[0];
    return result_map;

def predict_labels(test_matrix, classifier_path_list):
    result_map={};
    for classifier_index, classifier_path in enumerate(classifier_path_list):
        with open(classifier_path, 'r') as f:
            classifier_for_1st=pickle.load(f);
        label_name=(os.path.basename(classifier_path)).split('_')[0];
        predict_label=classifier_for_1st.predict(test_matrix);
        result_map[label_name]=predict_label[0]; 
    return result_map;

def eval_on_single_file(test_corpus_dir,feature_map_numeric, feature_map_character, tfidf_score_map,classifier_path_list, args):
    """
    RETURN void    
    """
    stop=args.stop;
    feature_show=args.feature_show;
    tfidf_flag=args.tfidf;
    exno=args.experiment_no;
    #------------------------------------------------------------     
    test_filepath=test_corpus_dir;
    #------------------------------------------------------------ 
    if args.persian_test==True:
        tokens_stack, motif_stack=file_loader(test_filepath, stop);
    elif args.dutch_test==True:
        tokens_stack, motif_stack=file_loader_dutch(test_filepath, stop);
    test_data_list=unify_stack(tokens_stack, motif_stack);
    #------------------------------------------------------------ 
    if args.arow==True:
        #この関数はまだ未改修
        out_libsvm_format(tokens_stack, feature_map_character, feature_map_numeric);
        result_map=eval_with_arow();
    elif args.liblinear==True:
        out_libsvm_format(tokens_stack, feature_map_character, feature_map_numeric, feature_show, tfidf_flag);
        result_map=call_liblinear.eval_with_liblinear(exno);         
    elif args.mulan==True:  
        call_mulan.out_mulan_file(test_data_list, feature_map_character, feature_map_numeric, tfidf_score_map, args);
        model_type='RAkEL';
        arff_train='../get_thompson_motif/classifier/mulan/exno{}.arff'.format(exno);
        modelsavepath='../get_thompson_motif/classifier/mulan/exno{}.model'.format(exno);
        arff_test='./arff_and_xml/test_{}.arff'.format(args.experiment_no);
        xml_file='./arff_and_xml/test_{}.xml'.format(args.experiment_no);
        call_mulan.mulan_command(model_type, arff_train, xml_file, 
                                 arff_test, modelsavepath, 
                                 args.experiment_no, args.reduce_method);
        sys.exit("Stil not implemented");
    else:
        #この関数はまだ未改修
        test_matrix=construct_input_matrix(tokens_stack, feature_map_character, feature_map_numeric);
        result_map=predict_labels(test_matrix, classifier_path_list);
    #------------------------------------------------------------ 
    
    gold_map={};
    list_gold_cap_result=[];
    for gold_motif in motif_stack:
        alphabet_label=gold_motif[0];
        gold_map[alphabet_label]=1;
    gold_cap_result=0;
    for result_label in result_map:
        if result_label in gold_map and result_map[result_label]==1:
            gold_cap_result+=1;
            list_gold_cap_result.append(result_label);
    print '-'*30;
    print 'RESULT\nresult of classifiers:{}\ngold:{}\ncorrect estimation:{}\n'.format(result_map,
                                                                                      gold_map,
                                                                                      list_gold_cap_result);
    if 1 in result_map.values():
        precision=float(gold_cap_result) / len([label for label in result_map.values() if label==1]);
    else:
        precision=0;
    recall=float(gold_cap_result) / len(gold_map);
    if not precision==0 and not recall==0:
        F=float(2*precision*recall)/(precision+recall);
    else:
        F=0;
    print 'Precision:{}\nRecall:{}\nF:{}'.format(precision, recall, F);     

def make_goldmap(motif_stack):
    gold_map={};
    for gold_motif in motif_stack:
        alphabet_label=gold_motif[0];
        gold_map[alphabet_label]=1;

    return gold_map;

def get_the_num_of_1_classifier(result_map):
    """
    1を返す分類器の数を数える
    RETURN: int num_of_1_classifer
    """
    num_of_1_classifer=len([label for label in result_map if result_map[label]==1]);
    return num_of_1_classifer

def calc_h_loss(result_map, motif_stack, h_loss_sum):
    y_delta_z=0;
    for p_l in result_map:
        z_i=result_map[p_l];
        if z_i==1 and p_l in motif_stack:
            y_delta_z+=0;
        elif z_i==1 and p_l not in motif_stack:
            y_delta_z+=1;
        elif z_i==0 and p_l in motif_stack:
            y_delta_z+=1;
        elif z_i==0 and p_l not in motif_stack:
            y_delta_z+=0;
        else:
            print 'Invalid:This messae means, this function has bugs'
    
    try:
        h_loss_sum+=float(y_delta_z)/len(result_map);
    except ZeroDivisionError:
        h_loss_sum=0;

    return h_loss_sum
    
def calc_subset_acc(result_map, gold_map, subset_acc_sum):
    result_only_1={};
    for label in result_map:
        if result_map[label]==1:
            result_only_1[label]=1;

    if gold_map==result_only_1:
        subset_acc_sum+=1;
        return subset_acc_sum;
    else:
        return subset_acc_sum;

def calc_p_r_f(result_map, motif_stack, ex_p_sum, ex_r_sum, ex_f_sum, acc_sum):
    num_of_cap=0;
    len_z_i=0;
    for p_l in result_map:
        z_i=result_map[p_l];        
        if z_i==1 and p_l in motif_stack:
            num_of_cap+=1;
        if z_i==1:
            len_z_i+=1;
    
    try:
        ex_p_sum+=float(num_of_cap)/len_z_i;
        print 'P',float(num_of_cap)/len_z_i
    except ZeroDivisionError:
        ex_p_sum+=0;
    try:
        ex_f_sum+=(2*float(num_of_cap))/(len_z_i+len(motif_stack));
        print 'F',(2*float(num_of_cap))/(len_z_i+len(motif_stack));
    except ZeroDivisionError:
        ex_f_sum+=0;
    try:
        acc_sum+=float(num_of_cap)/(max(len_z_i, len(motif_stack)));
    except ZeroDivisionError:
        acc_sum+=0;
    try:
        ex_r_sum+=float(num_of_cap)/len(motif_stack);
        print 'R',float(num_of_cap)/len(motif_stack);
    except ZeroDivisionError:
        ex_r_sum+=0;

    return ex_p_sum, ex_r_sum, ex_f_sum, acc_sum;

def eval_on_multi_file(test_corpus_dir, feature_map_character, feature_map_numeric, feature_show, tfidf_score_map, args):
    if args.liblinear==True:
        doc_based.multipule_eval_for_liblinear(test_corpus_dir, feature_map_character, feature_map_numeric, feature_show, args);
    elif args.arow==True:
        sent_based.multipule_eval_for_arow(test_corpus_dir, feature_map_character, feature_map_numeric, feature_show, args); 
    elif args.logistic==True:
        sent_based.multipule_eval_for_logistic(test_corpus_dir, feature_map_character, feature_map_numeric, feature_show, args); 
    """    
    elif args.mulan==True:
        test_data_list_multi=[];
        #------------------------------------------------------------
        for test_file in load_files(test_corpus_dir):
            #------------------------------------------------------------   
            if args.persian_test==True:
                tokens_stack, motif_stack=file_loader(test_file, stop);
            elif args.dutch_test==True:
                tokens_stack, motif_stack=file_loader_dutch(test_file, stop);
            #------------------------------------------------------------
            test_data_tuple=unify_stack(tokens_stack, motif_stack);
            test_data_list_multi.append(test_data_tuple);
        #------------------------------------------------------------
        call_mulan.out_mulan_file(test_data_list_multi, feature_map_character, feature_map_numeric, tfidf_score_map, args);    
        model_type=args.mulan_reduce_method;
        arff_train='../get_thompson_motif/classifier/mulan/exno{}.arff'.format(exno);
        modelsavepath='../get_thompson_motif/classifier/mulan/exno{}.model'.format(exno);
        arff_test='./arff_and_xml/test_{}.arff'.format(args.experiment_no);
        xml_file='./arff_and_xml/test_{}.xml'.format(args.experiment_no);
        call_mulan.mulan_command(model_type, arff_train, xml_file, 
                                 arff_test, modelsavepath, args.experiment_no);
                                 """

def level_1(single_file_eval, feature_map_character, feature_map_numeric, test_corpus_dir,
            stop, arow, liblinear, feature_show, tfidf_flag, tfidf_score_map, args):
                
    if single_file_eval==True:
        eval_on_single_file(test_corpus_dir,feature_map_numeric,feature_map_character,tfidf_score_map,args);
    else:
        eval_on_multi_file(test_corpus_dir,feature_map_character,feature_map_numeric,feature_show,tfidf_score_map,args);

def main(level, single_file_eval, input_path, stop, arow, liblinear, feature_show, args):
    exno=str(args.experiment_no);
    tfidf_flag=args.tfidf;
    feature_map_character_path='../classifier/feature_map_character/feature_map_character_1st.json.'+exno;
    feature_map_numeric_path='../classifier/feature_map_numeric/feature_map_numeric_1st.json.'+exno;
    tfidf_score_map_path='../classifier/tfidf_weight/tfidf_word_weight.json.'+exno;
    if args.persian_test==True:
        test_corpus_dir='../test_resource/ifn/'
    elif args.dutch_test==True:
        test_corpus_dir='../test_resource/dfd/';
    elif args.dutch_orig_test==True:
        test_corpus_dir='../test_resource/dfd_orig/';
    
    if single_file_eval==True:
        test_corpus_dir=input_path;
    
    feature_map_character=load_feature_map(feature_map_character_path);
    feature_map_numeric=load_feature_map(feature_map_numeric_path)
    tfidf_score_map=load_feature_map(tfidf_score_map_path);

    if level==1:
        level_1(single_file_eval, feature_map_character, feature_map_numeric, test_corpus_dir,
                stop, arow, liblinear, feature_show, tfidf_flag, tfidf_score_map, args);

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='');
    parser.add_argument('-single_eval', '--single_eval',
                        help='if added, this script evaluates just for file which you specify in -i flag',
                        action='store_true', default=False);
    parser.add_argument('-level', 
                        '--level', 
                        help='level of thompson tree for classification', 
                        default=1, 
                        required=False);
    parser.add_argument('-dutch_test', action='store_true', help='evaluation on dutch test data');
    parser.add_argument('-persian_test', action='store_true', help='evaluation on persian test data');
    parser.add_argument('-dutch_orig_test', action='store_true', help='evaluation on dutch_orig test data');
    parser.add_argument('-i', help='If single mode, add input file path', default=False);
    parser.add_argument('-feature_show', help='If added, show features', action='store_true');
    parser.add_argument('-stop', help='If added, eliminate stopwords from test file', action='store_true');
    parser.add_argument('-arow', help='If added, arow classifies', action='store_true');
    parser.add_argument('-logistic', help='If added, logistic classifies is used', action='store_true');
    parser.add_argument('-threshold', help='The threshold for arow mode and logistic mode', default=0.5);
    parser.add_argument('-tfidf', help='If added, use TFIDF feature', action='store_true');
    parser.add_argument('-liblinear',help='If added, liblinear classifies', action='store_true');
    parser.add_argument('-mulan', '--mulan', action='store_true', help='Use Mulan multilabel classifier');
    parser.add_argument('-mulan_reduce_method', help='reduce method for dimetion. This method must be same as reduce method of training');
    parser.add_argument('-exno', '--experiment_no', default=0);
    parser.add_argument('-save_performance', '--save_performance', help='To save the classification result', action='store_true');
    args=parser.parse_args();
    #------------------------------------------------------------ 
    if args.dutch_test==False and args.persian_test==False and args.dutch_orig_test==False:
        sys.exit("-persian_test and dutch_test and -dutch_orig_test are all False. Choose one of them")
    if args.liblinear==False and args.logistic==False and args.arow==False and args.mulan==False:
        sys.exit("decoding tools are all False. Choose one of them")
    
    print 'model number {} is specified'.format(args.experiment_no);
    #------------------------------------------------------------ 
    
    main(args.level, args.single_eval, args.i, args.stop, args.arow, args.liblinear, args.feature_show, args);
