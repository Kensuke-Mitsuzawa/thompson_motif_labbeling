# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 19:06:02 2014

@author: kensuke-mi
@date: 2014/03/01
"""
import codecs,subprocess,json,os,sys,re;
import file_decoder as f_d;
import config as conf;

def label_converter(label,args):
    """
    RETURN: list return_list [unicode correct_label_for_level] 
    """
    if label==None:
        pass;
    else:
        if args.level==1:
            if re.search(ur'\w\d+',label):
                modified_label=label[0]; 
                return modified_label
            else:
                return label

def file_loader_sentence(test_file,args):
    """
    文ごとにインスタンスを評価できる形式に返す．戻り値は二次元リスト
    ARGS: 省略
    RETURN: list sentences_in_document [list tokens_in_sentence [unicode token]]
    RETURN2: map gold_label_map {int line_index : list gold_label [unicode label]}
    """
    sentence_in_document=[];
    gold_label_map={};    
    with codecs.open(test_file,'r','utf-8') as json_content:
        test_file_map=json.load(json_content);

    doc_str=test_file_map['doc_str'];
    
    for line_index,sentence_label_tuple in enumerate(doc_str):
        sentence=sentence_label_tuple[0];
        label=sentence_label_tuple[1];
        label=label_converter(label,args);
        #もし，ラベル付きの文のみで評価をしたければ，ここに下のif文を追加       
        #if not label==None:
        sentence_in_document.append(sentence);
        gold_label_map[line_index]=[label]

    return sentence_in_document,gold_label_map;

def arow_output_analysis(p,alphabet_label,label_index_map,args):
    """
    Arowの出力を分析する
    分類時の確信度が閾値を超えていた場合のみ，採用
    RETURN: label_index_map {int line_index : list label_of_model [unicode label]}
    """       
    for line_index, line in enumerate(p.stdout.readlines()):        
        label, score=line.split();
        score=float(score);        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        if score > args.threshold:
            if line_index not in label_index_map:
                label_index_map[line_index]=[alphabet_label];
            else:
                label_index_map[line_index].append(alphabet_label);
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return label_index_map;

def call_arowmodel(model_path_list,args):
    """
    RETURN: map label_index_map {int line_index : list label_of_mode_that_returns_True [unicode label]}
    """
    label_index_map={};
    for model_file in model_path_list:
        alphabet_label=unicode(os.path.basename(model_file)[0], 'utf-8');
        #call arow            
        p=subprocess.Popen(['arow_test', 'test.data', model_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE);
        #出力から確信度以上の行のみを取り出す        
        label_index_map=arow_output_analysis(p,alphabet_label,label_index_map,args);

    return label_index_map;

def multipule_eval_for_arow(test_corpus_dir, feature_map_character, feature_map_numeric, feature_show, args):
    """
    AROWで作成したモデル（一文ごとにラベルを判断）を行う
    そもそもテストの対象は，ラベルがついている文のみ
    """
    if args.save_performance==True:
        performance_out=codecs.open('./performance_result.'+args.experiment_no,'w','utf-8');
        performance_out.write(args.experiment_no+u'\n');
        performance_out.write(u'-'*30+u'\n');

    tfidf_flag=args.tfidf;
    exno=args.experiment_no;
    if args.dutch_test==True:
        sys.exit("[Warning!] sentence based prediction can not predict for DFD")
    
    model_dir_path=conf.arow_model_dir_path;
    model_path_list=f_d.load_files(model_dir_path, '.arowmodel2nd.'+exno);
    #If path to model is wrong. End
    if model_path_list==[]:
        sys.exit("I tried open Arow models, but failded. Arow model path {} is wrong. Check the path to model".format(model_dir_path)); 

    precision_sum=0; recall_sum=0; F_sum=0; h_loss_sum=0; subset_acc_sum=0;
    ex_p_sum=0; ex_r_sum=0; ex_f_sum=0; acc_sum=0;
    classifier_return_1_sum=0;
    for test_file in f_d.load_files(test_corpus_dir):
    #============================================================    
        sentences_in_document,gold_label_map=file_loader_sentence(test_file,args);
   
        f_d.out_libsvm_format_sentence(sentences_in_document, feature_map_character, feature_map_numeric, feature_show, tfidf_flag);

        label_index_map=call_arowmodel(model_path_list,args);
        #------------------------------------------------------------
        print label_index_map, gold_label_map;
        
        #AROWの出力値が確率じゃなくって，ただのスコアになってる！？
        """        
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
        """

def logistic_output_analysis(p_label,p_val,alphabet_label,label_index_map,args):
    """
    RETURN: map label_index_map {int line_index : list label_of_mode_that_returns_True [unicode label]}
    """
    for instance_index, instance_result in enumerate(p_val):
        prob_positive=instance_result[0];
        prob_negative=instance_result[1];
        if p_label[instance_index]==1.0 and prob_positive > float(args.threshold):
            if instance_index in label_index_map:
                label_index_map[instance_index].append(alphabet_label)
            else:
                label_index_map[instance_index]=[alphabet_label];

    return label_index_map;

def call_logisticmodel(model_path_list,args):
    """
    RETURN: map label_index_map {int line_index : list label_of_mode_that_returns_True [unicode label]}
    """
    label_index_map={};
    test_y,test_x=conf.svmutil.svm_read_problem('test.data');
    for model_file in model_path_list:
        alphabet_label=unicode(os.path.basename(model_file)[0], 'utf-8');
        #call liblinear logistic            
        model=conf.liblinearutil.load_model(model_file);
        p_label,p_acc,p_val=conf.liblinearutil.predict(test_y,test_x,model,'-b 1');  
        #出力から確信度以上の行のみを取り出す        
        label_index_map=logistic_output_analysis(p_label,p_val,alphabet_label,label_index_map,args);

    return label_index_map;

def multipule_eval_for_logistic(test_corpus_dir, feature_map_character, feature_map_numeric, feature_show, args):
    """
    liblinearのlogisticで作成したモデル（一文ごとにラベルを判断）を行う
    一回でも+1が発生すれば，文書にラベルが付与されたと見なす
    """
    if args.save_performance==True:
        performance_out=codecs.open('./performance_result.'+args.experiment_no,'w','utf-8');
        performance_out.write(args.experiment_no+u'\n');
        performance_out.write(u'-'*30+u'\n');

    tfidf_flag=args.tfidf;
    exno=args.experiment_no;
    if args.dutch_test==True:
        sys.exit("[Warning!] sentence based prediction can not predict for DFD")
    
    model_dir_path=conf.logistic_model_dir_path
    model_path_list=f_d.load_files(model_dir_path,'.logistic.'+exno);
    #If path to model is wrong. End
    if model_path_list==[]:
        sys.exit("I tried open Arow models, but failded. Arow model path {} is wrong. Check the path to model".format(model_dir_path)); 

    precision_sum=0; recall_sum=0; F_sum=0; h_loss_sum=0; subset_acc_sum=0;
    classifier_return_1_sum=0;
    num_of_files=len(f_d.load_files(test_corpus_dir))
    for test_file in f_d.load_files(test_corpus_dir):
    #============================================================    
        sentences_in_document,gold_label_map=file_loader_sentence(test_file,args);
   
        f_d.out_libsvm_format_sentence(sentences_in_document, feature_map_character, feature_map_numeric, feature_show, tfidf_flag);

        label_index_map=call_logisticmodel(model_path_list,args)

        precision_of_doc,recall_of_doc,f_of_doc=evaluation_function_per_sentence(label_index_map,gold_label_map);
        precision_sum+=precision_of_doc;
        recall_sum+=recall_of_doc;
        F_sum+=f_of_doc;
    
        print precision_sum
        print recall_sum
        print F_sum
    #============================================================    
    evaluation_function_per_doc(precision_sum,recall_sum,F_sum,num_of_files);

def evaluation_function_per_sentence(label_index_map,gold_label_map):
    """
    文ごとの評価ではNoneもひとつのラベルとみなす
    そうなしないと，predictedはラベルを推薦してるけど，goldがNone，いう場合に評価ができないから．
    """
    precision_of_sentence=0;
    recall_of_sentence=0;
    f_of_sentence=0;
    #gold_label_mapにはかならずすべてのline indexが存在する
    for line_index_in_gold, gold_label_of_sentence in gold_label_map.items():
        #gold_label_of_sentence=gold_label_map[line_index_in_predicted];
        if line_index_in_gold in label_index_map:
            predicted_label_set=label_index_map[line_index_in_gold]
        else:
            predicted_label_set=[None];

        num_of_cap_predicted_gold=len([predicted_label for predicted_label in predicted_label_set if predicted_label in gold_label_of_sentence]) 
        
        
        num_of_predicted_label_set=len(predicted_label_set);
        num_of_gold_label_set=len(gold_label_of_sentence);
        precision_of_sentence+=float(num_of_cap_predicted_gold)/num_of_predicted_label_set;
        recall_of_sentence+=float(num_of_cap_predicted_gold)/num_of_gold_label_set;
        f_of_sentence+=(2*float(num_of_cap_predicted_gold))/(num_of_predicted_label_set+num_of_gold_label_set)
        #Noneはラベルなし．とかんがえると下のようになるが．．．
        """
        if predicted_label_set==[None]:
            num_of_predicted_label_set=0;
        else:
            num_of_predicted_label_set=len(predicted_label_set);
        
        if gold_label_of_sentence==[None]: 
            num_of_gold_label_set=0;
        else:
            num_of_gold_label_set=len(gold_label_of_sentence);

        if num_of_predicted_label_set==0:
            precision_of_sentence=0;
        else:
            precision_of_sentence+=float(num_of_cap_predicted_gold)/num_of_predicted_label_set;
           
        if num_of_gold_label_set==0:
            recall_of_sentence=0;
        else:
            recall_of_sentence+=float(num_of_cap_predicted_gold)/num_of_gold_label_set;

        if num_of_predicted_label_set==0 and num_of_gold_label_set==0:
            f_of_sentence=0;
        else:
            f_of_sentence+=(2*float(num_of_cap_predicted_gold))/(num_of_predicted_label_set+num_of_gold_label_set)
        """

        print predicted_label_set
        print gold_label_of_sentence


        print precision_of_sentence
        print recall_of_sentence
        print f_of_sentence

        num_of_sentence_in_doc=len(gold_label_map);

    precision_of_doc=precision_of_sentence/num_of_sentence_in_doc;
    recall_of_doc=recall_of_sentence/num_of_sentence_in_doc;
    f_of_doc=f_of_sentence/num_of_sentence_in_doc;

    return precision_of_doc,recall_of_doc,f_of_doc

def evaluation_function_per_doc(precision_sum,recall_sum,F_sum,num_of_files):
    precision=precision_sum/num_of_files;
    recall=recall_sum/num_of_files;
    f=F_sum/num_of_files;

    print precision,recall,f

    """
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
        """
