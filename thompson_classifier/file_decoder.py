#! /usr/bin/python
# -*- coding:utf-8 -*-
__date__='2014/02/18';

import re, subprocess, argparse, os, glob, sys, pickle, json, codecs;
sys.path.append('../get_thompson_motif/');
import call_liblinear, call_mulan;
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

def file_loader(file_path, eliminate_stop=True):
    """
    Read and construct tokens list from test file.
    ARGS: file_path(file which you want to classify)
    """
    #queryファイルの読み込み
    line_flag=False;
    motif_flag=False;
    motif_stack=[];
    line_stack=[];
    with codecs.open(file_path, 'r', 'utf-8') as lines:
        for line in lines:
            if line==u'\n':
                continue;
            if line==u'#motif\n':
                motif_flag=True;
                continue;
            elif line==u'#text\n':
                motif_flag=False;
                line_flag=True;
                continue;
            if motif_flag==True and line_flag==False:
                motif_stack.append(line.strip());
            if line_flag==True and motif_flag==False:
                line_stack.append(line.strip());
    tokens_stack=[tokenize.wordpunct_tokenize(line) for line in line_stack]
    tokens_stack=[[t.lower() for t in l] for l in tokens_stack]
    if eliminate_stop==True: 
        tokens_stack=[[t for t in l if t not in stopwords and t not in symbols] for l in tokens_stack]
    #配列を二次元から一次元に落とす．ついでにlemmatizeも行う．
    tokens_stack=[lemmatizer.lemmatize(t) for line in tokens_stack for t in line];
    return tokens_stack, motif_stack;

def file_loader_dutch(file_path, eliminate_stop=True):
    filename=os.path.basename(file_path);
    label=re.sub(ur'(\w_)\d+\.tok\.seg\.en', ur'\1', filename);
    motif_stack=label.split(u'_');
    motif_stack=[m for m in motif_stack if not m==u''];
    file_obj=codecs.open(file_path, 'r', 'utf-8');
    document=tokenize.wordpunct_tokenize(file_obj.read());
    file_obj.close();
    tokens_stack=[t.lower() for t in document];
    if eliminate_stop==True:
        tokens_stack=[lemmatizer.lemmatize(t) for t in tokens_stack if lemmatizer.lemmatize(t) not in stopwords and lemmatizer.lemmatize(t) not in symbols];
    else:
        tokens_stack=[lemmatizer.lemmatize(t) for t in tokens_stack];        
    return tokens_stack, motif_stack;

def file_loader_dutch_orig(file_path, eliminate_stop=True):
    """
    オランダ語原文のテストファイルを読み込む
    RETURN1:list tokens_stack [unicode token]
    RETURN2: list motif_stack [unicode motif_number]
    """
    filename=os.path.basename(file_path);
    motif_stack=filename.split(u'_');
    motif_stack=[m for m in motif_stack if not m==u''];
    file_obj=codecs.open(file_path, 'r', 'utf-8');
    document=tokenize.wordpunct_tokenize(file_obj.read());
    file_obj.close();
    tokens_stack=[t.lower() for t in document];
    return tokens_stack, motif_stack;

def file_loader_sentence(test_file, eliminate_stop=True):
    """
    文ごとにインスタンスを評価できる形式に返す．戻り値は二次元リスト 
    ARGS: 省略
    RETURN1: list sentences_in_document [list tokens_in_sentence [unicode token]]
    RETURN2: list motif_stack [unicode motif_number]
    """
    #queryファイルの読み込み
    line_flag=False;
    motif_flag=False;
    motif_stack=[];
    line_stack=[];
    with codecs.open(test_file, 'r', 'utf-8') as lines:
        for line in lines:
            if line==u'\n':
                continue;
            if line==u'#motif\n':
                motif_flag=True;
                continue;
            elif line==u'#text\n':
                motif_flag=False;
                line_flag=True;
                continue;
            if motif_flag==True and line_flag==False:
                motif_stack.append(line.strip());
            if line_flag==True and motif_flag==False:
                line_stack.append(line.strip());
    
    tokens_stack=[tokenize.wordpunct_tokenize(line) for line in line_stack]
    sentences_in_document=[[t.lower() for t in l] for l in tokens_stack]
    if eliminate_stop==True: 
        sentences_in_document=[[t for t in l if t not in stopwords and t not in symbols] for l in sentences_in_document];

    return sentences_in_document, motif_stack;

def file_loader_dutch_sentence(file_path, eliminate_stop=True):
    """
    文ごとにインスタンスを評価できる形式に返す．戻り値は二次元リスト 
    ARGS: 省略
    RETURN1: list sentences_in_document [list tokens_in_sentence [unicode token]]
    RETURN2: list motif_stack [unicode motif_number]
    """
    filename=os.path.basename(file_path);
    label=re.sub(ur'(\w_)\d+\.tok\.seg\.en', ur'\1', filename);
    motif_stack=label.split(u'_');
    motif_stack=[m for m in motif_stack if not m==u''];
    file_obj=codecs.open(file_path, 'r', 'utf-8');
    sentences_in_document=(file_obj.read()).split(u'\n');
    tokens_sentences=[tokenize.wordpunct_tokenize(sentence) for sentence in sentences_in_document];
    tokens_sentences=[[t.lower() for t in sentence] for sentence in tokens_sentences];
    file_obj.close();
    if eliminate_stop==True:
        tokens_sentences=[[t for t in sentence if lemmatizer.lemmatize(t) not in stopwords and lemmatizer.lemmatize(t) not in symbols] for sentence in tokens_sentences];
    
    return tokens_sentences, motif_stack;

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

def out_libsvm_format(tokens_stack, feature_map_character, feature_map_numeric, feature_show, tfidf_flag):
    out_file=codecs.open('./test.data', 'w', 'utf-8');
    one_instance_stack=[];
    for token in tokens_stack:
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
    one_instance_stack.sort();
    out_file.write('+1 ');
    one_instance_stack=[out_file.write(u'{}:{} '.format(tuple_item[0], tuple_item[1]))\
                        for tuple_item in list(set(one_instance_stack))];
    #out_file.write(u'+1 {}\n'.format(u' '.join(one_instance_stack)));
    out_file.close();
    if feature_show==True:
        print u'-'*30;

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

def construct_input_matrix(tokens_stack, feature_map):
    feature_space=len(feature_map);
    test_matrix=lil_matrix((feature_space, 1));
    #sample_index is same as column index in test_matrix
    for one_instance in tokens_stack:
        for token in one_instance:
            #convert to fearure number from token
            if token in feature_map:
                feature_number=feature_map[token];
                test_matrix[feature_number, 0]=1;
    test_matrix=test_matrix.T;
    return test_matrix;

def predict_labels(test_matrix, classifier_path_list):
    result_map={};
    for classifier_index, classifier_path in enumerate(classifier_path_list):
        with open(classifier_path, 'r') as f:
            classifier_for_1st=pickle.load(f);
        label_name=(os.path.basename(classifier_path)).split('_')[0];
        predict_label=classifier_for_1st.predict(test_matrix);
        result_map[label_name]=predict_label[0]; 
    return result_map;

def get_accuracy(num_docs_having_motif, num_of_correct_decision):
    print '-'*30;
    print 'A way to get accuracy is: num. of correct decision of label classifier / num. of docs which have label';
    print 'num. docs having motif\n{}'.format(num_docs_having_motif)
    print 'num. of correct decision\n{}'.format(num_of_correct_decision);
    for label in num_docs_having_motif:
        if label in num_docs_having_motif and label in num_of_correct_decision:
            accuracy=float(num_of_correct_decision[label])/num_docs_having_motif[label];
            print 'Accuracy of classifier_{} is {}'.format(label, accuracy);
        else:
            print 'Acuracy of classifier_{} is {}'.format(label, 0);

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

def calc_p_r_f_old(num_docs_having_motif, result_map, motif_stack, num_of_correct_decision, precision_sum, recall_sum, F_sum):
    #古い方の評価関数
    gold_map={};
    list_gold_cap_result=[];
    for gold_motif in motif_stack:
        alphabet_label=gold_motif[0];
        gold_map[alphabet_label]=1;
        #ラベルを持っている文書数を保存しておく
        if gold_motif[0] in num_docs_having_motif:
            num_docs_having_motif[gold_motif[0]]+=1;
        else:
            num_docs_having_motif[gold_motif[0]]=1;
    gold_cap_result=0;
    for result_label in result_map:
        #result_labelがgold_mapの中に存在していて，かつ，result_map[result_label]==1のとき，分類器が正しい判断をできているはず
        if result_label in gold_map and result_map[result_label]==1:
            gold_cap_result+=1;
            list_gold_cap_result.append(result_label);
            #accuracyを計算するため
            if result_label in num_of_correct_decision:
                num_of_correct_decision[result_label]+=1;
            else:
                num_of_correct_decision[result_label]=1;

    if 1 in result_map.values():
        precision=float(gold_cap_result) / len([label for label in result_map.values() if label==1]);
    else:
        precision=0;
    recall=float(gold_cap_result) / len(gold_map);
    precision_sum+=precision; recall_sum+=recall;
    if not precision==0 and not recall==0:
        F_sum+=float(2*precision*recall)/(precision+recall);
    else:
        F_sum+=0;
    print u'Performance for this file p:{} r:{}'.format(precision, recall);
    print 'RESULT\nresult of classifiers:{}\ngold:{}\ncorrect estimation:{}\n'.format(result_map,
                                                                                      gold_map,
                                                                                      list_gold_cap_result);
    
    return num_docs_having_motif,num_of_correct_decision, precision_sum, recall_sum, F_sum, gold_map;                                                                                      
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
    except ZeroDivisionError:
        ex_p_sum+=0;
    try:
        ex_f_sum+=(2*float(num_of_cap))/(len_z_i+len(motif_stack));
    except ZeroDivisionError:
        ex_f_sum+=0;
    try:
        acc_sum+=float(num_of_cap)/(max(len_z_i, len(motif_stack)));
    except ZeroDivisionError:
        acc_sum+=0;
    try:
        ex_r_sum+=float(num_of_cap)/len(motif_stack);
    except ZeroDivisionError:
        ex_r_sum+=0;

    return ex_p_sum, ex_r_sum, ex_f_sum, acc_sum;

def multipule_eval_for_liblinear(test_corpus_dir, feature_map_character, feature_map_numeric, feature_show, args):
    num_docs_having_motif={};
    tfidf_flag=args.tfidf;
    exno=args.experiment_no;
    
    #分類器が正しい判断をした回数を保存する．つまりCAP(gold motif tag, candidate by classifier)
    num_of_correct_decision={};
    precision_sum=0; recall_sum=0; F_sum=0; h_loss_sum=0; subset_acc_sum=0;
    ex_p_sum=0; ex_r_sum=0; ex_f_sum=0; acc_sum=0;
    classifier_return_1_sum=0;
    for test_file in load_files(test_corpus_dir):
    #============================================================    
        try:
            with codecs.open(test_file,'r','utf-8') as json_content:
                test_file_map=json.load(json_content);
            
            tokens_stack_list=test_file_map['doc_str'];
            motif_stack=test_file_map['labels'];

            tokens_stack=[t for l in tokens_stack_list for t in l]; 
            out_libsvm_format(tokens_stack, feature_map_character, feature_map_numeric, feature_show, tfidf_flag);
            result_map=call_liblinear.eval_with_liblinear(exno);
            
            num_docs_having_motif,num_of_correct_decision, precision_sum, recall_sum, F_sum, gold_map=calc_p_r_f_old(num_docs_having_motif,result_map,motif_stack,num_of_correct_decision,precision_sum,recall_sum,F_sum);

            h_loss_sum=calc_h_loss(result_map, gold_map, h_loss_sum);
            subset_acc_sum=calc_subset_acc(result_map, gold_map, subset_acc_sum);
            ex_p_sum, ex_r_sum, ex_f_sum, acc_sum=calc_p_r_f(result_map, gold_map, ex_p_sum, ex_r_sum, ex_f_sum, acc_sum);
            classifier_return_1_sum+=get_the_num_of_1_classifier(result_map);
        #------------------------------------------------------------   
        except ValueError:
            print 'File decode error in some reason'
    #============================================================    
    num_of_files=len(load_files(test_corpus_dir));
    h_loss=h_loss_sum/num_of_files;
    subset_acc=float(subset_acc_sum)/num_of_files;
    ex_p=ex_p_sum/num_of_files;
    ex_r=ex_r_sum/num_of_files;
    ex_f=ex_f_sum/num_of_files;
    acc=acc_sum/num_of_files;
    classifier_return_1=float(classifier_return_1_sum)/num_of_files;
    precision_ave=precision_sum/len(load_files(test_corpus_dir));
    recall_ave=recall_sum/len(load_files(test_corpus_dir));
    F_ave=F_sum/len(load_files(test_corpus_dir));
   
    print '-'*30;
    print 'RESULT for {} files classification'.format(len(load_files(test_corpus_dir)));
    """
    print 'Average_precision:{}\nAverage_recall:{}\nAverage_F:{}'.format(precision_ave,
                                                                         recall_ave,
                                                                         F_ave);
    """

    print 'Hamming Loss:{}'.format(h_loss);
    print 'Subset Accuracy(classification accuracy):{}'.format(subset_acc);
    print 'example-based precision:{} example-based recall:{} example-based F:{} accuracy:{}'.format(ex_p, ex_r, ex_f, acc)
    print 'Average number of classifier which returns 1:{}'.format(classifier_return_1);
    #get_accuracy(num_docs_having_motif, num_of_correct_decision);

def multipule_eval_for_logistic(test_corpus_dir, feature_map_character, feature_map_numeric, feature_show, args):
    """
    liblinearのlogisticで作成したモデル（一文ごとにラベルを判断）を行う
    一回でも+1が発生すれば，文書にラベルが付与されたと見なす
    """
    env='pine'; 
    if env=='pine':
        #change below by an environment
        libsvm_wrapper_path='/home/kensuke-mi/opt/libsvm-3.17/python/';
    elif env=='local':
        libsvm_wrapper_path='/Users/kensuke-mi/opt/libsvm-3.17/python/';
        liblinear_wrapper_path='/Users/kensuke-mi/opt/liblinear-1.94/python/';
        sys.path.append(liblinear_wrapper_path);
    sys.path.append(libsvm_wrapper_path);
    import liblinearutil;
    import svmutil;
    
    if args.save_performance==True:
        performance_out=codecs.open('./performance_result.'+args.experiment_no,'w','utf-8');
        performance_out.write(args.experiment_no+u'\n');
        performance_out.write(u'-'*30+u'\n');

    #確信度の閾値
    threshold=float(args.threshold);
    #確信度を表示するか？オプション
    show_confidence=False;
    #確信度の平均値
    average_confidence=0;
    #+1のインスタンス数
    times_plus_1_ins=0;

    num_docs_having_motif={};
    stop=args.stop;
    tfidf_flag=args.tfidf;
    exno=args.experiment_no;
    
    model_dir_path='../get_thompson_motif/classifier/logistic_2nd/';
    model_path_list=load_files(model_dir_path, 'logistic.'+exno); 
    #分類器が正しい判断をした回数を保存する．つまりCAP(gold motif tag, candidate by classifier)
    num_of_correct_decision={};
    precision_sum=0; recall_sum=0; F_sum=0; h_loss_sum=0; subset_acc_sum=0;
    ex_p_sum=0; ex_r_sum=0; ex_f_sum=0; acc_sum=0;
    classifier_return_1_sum=0;
    for test_file in load_files(test_corpus_dir):
    #============================================================    
        result_map={};   
        gold_map={};
        #------------------------------------------------------------   
        if args.persian_test==True:
            #文ごとにインスタンスの作成
            sentences_in_document, motif_stack=file_loader_sentence(test_file, stop);
        elif args.dutch_test==True:
            sentences_in_document, motif_stack=file_loader_dutch_sentence(test_file, stop);
        #------------------------------------------------------------   
        out_libsvm_format_sentence(sentences_in_document, feature_map_character, feature_map_numeric, feature_show, tfidf_flag);
        test_y,test_x=svmutil.svm_read_problem('test.data');
        #------------------------------------------------------------   
        for model_file in model_path_list:
            decision_flag=False;
            alphabet_label=unicode(os.path.basename(model_file)[0], 'utf-8');
            result_map[alphabet_label]=0; 
          
            model=liblinearutil.load_model(model_file);
            p_label,p_acc,p_val=liblinearutil.predict(test_y,test_x,model,'-b 1');  
          
            for index,result_label in enumerate(p_label):
                if result_label==1.0:
                    decision_flag=True;
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
                if decision_flag==True and p_val[index][0] > threshold:
                    result_map[alphabet_label]=1;
                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        #------------------------------------------------------------   
        for gold_motif in motif_stack:
            alphabet_label=gold_motif[0];
            gold_map[alphabet_label]=1;
        #------------------------------------------------------------ 
        gold_cap_result={};
        for label in result_map:
            if result_map[label]==1 and label in gold_map:
                gold_cap_result[label]=1;
        #------------------------------------------------------------ 
        try:
            average=average_confidence/times_plus_1_ins; 
        except ZeroDivisionError:
            average=0;

        print '-'*30;
        print 'Filename:{}\nEstimated:{}\nGold:{}\nCorrect Estimation:{}'.format(test_file,result_map,gold_map,gold_cap_result);
        print 'average confidence is {}'.format(average);
        print '-'*30;
        #------------------------------------------------------------ 
        h_loss_sum=calc_h_loss(result_map, gold_map, h_loss_sum);
        subset_acc_sum=calc_subset_acc(result_map, gold_map, subset_acc_sum);
        ex_p_sum, ex_r_sum, ex_f_sum, acc_sum=calc_p_r_f(result_map, gold_map, ex_p_sum, ex_r_sum, ex_f_sum, acc_sum);
        classifier_return_1_sum+=get_the_num_of_1_classifier(result_map);
    #============================================================    
    num_of_files=len(load_files(test_corpus_dir));
    h_loss=h_loss_sum/num_of_files;
    subset_acc=float(subset_acc_sum)/num_of_files;
    ex_p=ex_p_sum/num_of_files;
    ex_r=ex_r_sum/num_of_files;
    ex_f=ex_f_sum/num_of_files;
    acc=acc_sum/num_of_files;
    classifier_return_1=float(classifier_return_1_sum)/num_of_files;
    precision_ave=precision_sum/len(load_files(test_corpus_dir));
    recall_ave=recall_sum/len(load_files(test_corpus_dir));
    F_ave=F_sum/len(load_files(test_corpus_dir));
    print '-'*30;
    print 'RESULT for {} files classification'.format(len(load_files(test_corpus_dir)));

    hamming_format=u'Hamming Loss:{}'.format(h_loss);
    subset_format=u'Subset Accuracy(classification accuracy):{}'.format(subset_acc);
    else_format=u'example-based precision:{} example-based recall:{} example-based F:{} accuracy:{}'.format(ex_p, ex_r, ex_f, acc)
    classifier_format=u'Ave. number of classifier which returns 1:{}'.format(classifier_return_1);
    print hamming_format;
    print subset_format;
    print else_format;
    print classifier_format

    if args.save_performance==True:
        performance_out.write(hamming_format+u'\n');
        performance_out.write(subset_format+u'\n');
        performance_out.write(else_format+u'\n');
        performance_out.write(classifier_format+u'\n');
        performance_out.close();


def multipule_eval_for_arow(test_corpus_dir, feature_map_character, feature_map_numeric, feature_show, args):
    """
    AROWで作成したモデル（一文ごとにラベルを判断）を行う
    一回でも+1が発生すれば，文書にラベルが付与されたと見なす
    """
    if args.save_performance==True:
        performance_out=codecs.open('./performance_result.'+args.experiment_no,'w','utf-8');
        performance_out.write(args.experiment_no+u'\n');
        performance_out.write(u'-'*30+u'\n');

    #確信度の閾値
    threshold=0;
    #確信度を表示するか？オプション
    show_confidence=False;
    #確信度の平均値
    average_confidence=0;
    #+1のインスタンス数
    times_plus_1_ins=0;

    num_docs_having_motif={};
    stop=args.stop;
    tfidf_flag=args.tfidf;
    exno=args.experiment_no;
    
    model_dir_path='../get_thompson_motif/classifier/arow/';
    model_path_list=load_files(model_dir_path, 'arowmodel2nd.'+exno); 
    #分類器が正しい判断をした回数を保存する．つまりCAP(gold motif tag, candidate by classifier)
    num_of_correct_decision={};
    precision_sum=0; recall_sum=0; F_sum=0; h_loss_sum=0; subset_acc_sum=0;
    ex_p_sum=0; ex_r_sum=0; ex_f_sum=0; acc_sum=0;
    classifier_return_1_sum=0;
    for test_file in load_files(test_corpus_dir):
    #============================================================    
        result_map={};   
        gold_map={};
        #------------------------------------------------------------   
        if args.persian_test==True:
            #文ごとにインスタンスの作成
            sentences_in_document, motif_stack=file_loader_sentence(test_file, stop);
        elif args.dutch_test==True:
            sentences_in_document, motif_stack=file_loader_dutch_sentence(test_file, stop);
        #------------------------------------------------------------   
        out_libsvm_format_sentence(sentences_in_document, feature_map_character, feature_map_numeric, feature_show, tfidf_flag);
        #------------------------------------------------------------   
        for model_file in model_path_list:
            decision_flag=False;
            alphabet_label=unicode(os.path.basename(model_file)[0], 'utf-8');
            result_map[alphabet_label]=0; 
            p=subprocess.Popen(['arow_test', 'test.data', model_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE);
            for line in p.stdout.readlines():
                #if not label
                if re.search(ur'-1\s.+', line):
                    pass;
                elif re.search(ur'\+1\s(.+)', line):
                    if show_confidence==True:
                        print line;
                
                    label, score=line.split();
                    score=float(score);
                    average_confidence+=score;
                    times_plus_1_ins+=1;
                    decision_flag=True; 
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
            if decision_flag==True and score > threshold:
                result_map[alphabet_label]=1;
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        #------------------------------------------------------------   
        for gold_motif in motif_stack:
            alphabet_label=gold_motif[0];
            gold_map[alphabet_label]=1;
        #------------------------------------------------------------ 
        gold_cap_result={};
        for label in result_map:
            if result_map[label]==1 and label in gold_map:
                gold_cap_result[label]=1;
        #------------------------------------------------------------ 
        try:
            average=average_confidence/times_plus_1_ins; 
        except ZeroDivisionError:
            average=0;

        print '-'*30;
        print 'Filename:{}\nEstimated:{}\nGold:{}\nCorrect Estimation:{}'.format(test_file,result_map,gold_map,gold_cap_result);
        print 'average confidence is {}'.format(average);
        print '-'*30;
        #------------------------------------------------------------ 
        h_loss_sum=calc_h_loss(result_map, gold_map, h_loss_sum);
        subset_acc_sum=calc_subset_acc(result_map, gold_map, subset_acc_sum);
        ex_p_sum, ex_r_sum, ex_f_sum, acc_sum=calc_p_r_f(result_map, gold_map, ex_p_sum, ex_r_sum, ex_f_sum, acc_sum);
        classifier_return_1_sum+=get_the_num_of_1_classifier(result_map);
    #============================================================    
    num_of_files=len(load_files(test_corpus_dir));
    h_loss=h_loss_sum/num_of_files;
    subset_acc=float(subset_acc_sum)/num_of_files;
    ex_p=ex_p_sum/num_of_files;
    ex_r=ex_r_sum/num_of_files;
    ex_f=ex_f_sum/num_of_files;
    acc=acc_sum/num_of_files;
    classifier_return_1=float(classifier_return_1_sum)/num_of_files;
    precision_ave=precision_sum/len(load_files(test_corpus_dir));
    recall_ave=recall_sum/len(load_files(test_corpus_dir));
    F_ave=F_sum/len(load_files(test_corpus_dir));
    print '-'*30;
    print 'RESULT for {} files classification'.format(len(load_files(test_corpus_dir)));

    hamming_format=u'Hamming Loss:{}'.format(h_loss);
    subset_format=u'Subset Accuracy(classification accuracy):{}'.format(subset_acc);
    else_format=u'example-based precision:{} example-based recall:{} example-based F:{} accuracy:{}'.format(ex_p, ex_r, ex_f, acc)
    classifier_format=u'Ave. number of classifier which returns 1:{}'.format(classifier_return_1);
    print hamming_format;
    print subset_format;
    print else_format;
    print classifier_format

    if args.save_performance==True:
        performance_out.write(hamming_format+u'\n');
        performance_out.write(subset_format+u'\n');
        performance_out.write(else_format+u'\n');
        performance_out.write(classifier_format+u'\n');
        performance_out.close();

def eval_on_multi_file(test_corpus_dir, feature_map_character, feature_map_numeric, feature_show, tfidf_score_map, args):
    exno=args.experiment_no;
    stop=args.stop;

    if args.liblinear==True:
        multipule_eval_for_liblinear(test_corpus_dir, feature_map_character, feature_map_numeric, feature_show, args);
    elif args.arow==True:
        multipule_eval_for_arow(test_corpus_dir, feature_map_character, feature_map_numeric, feature_show, args); 
    elif args.logistic==True:
        multipule_eval_for_logistic(test_corpus_dir, feature_map_character, feature_map_numeric, feature_show, args); 
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

def level_1(single_file_eval, feature_map_character, feature_map_numeric, test_corpus_dir,
            stop, arow, liblinear, feature_show, tfidf_flag, tfidf_score_map, args):
                
    classifier_dir='../get_thompson_motif/classifier/1st_layer/';
    classifier_path_list=load_files(classifier_dir, '.pickle');
    if single_file_eval==True:
        eval_on_single_file(test_corpus_dir,feature_map_numeric, feature_map_character, tfidf_score_map,classifier_path_list, args);
    else:
        eval_on_multi_file(test_corpus_dir, feature_map_character, feature_map_numeric, feature_show, tfidf_score_map, args);

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
    #TODO dutch_folktale_corpus_orig用のjsonを作って配置しておくこと
    elif args.dutch_orig_test==True:
        test_corpus_dir='../dutch_folktale_corpus/given_script/top_dutch/top_document_test/';
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
    main(args.level, args.single_eval, args.i, args.stop, args.arow, args.liblinear, args.feature_show, args);
