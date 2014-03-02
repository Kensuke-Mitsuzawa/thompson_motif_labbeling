# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 18:58:04 2014

@author: kensuke-mi
"""
__date__='2014/03/01'

import codecs,json
import file_decoder as f_d;
import call_liblinear;

def file_loader_json_doc(test_file):
    """
    jsonファイルからデータを読み込む
    インスタンスの生成単位は，「文書」
    RETURN1: list motif_stack [unicode motif_label]
    RETURN2: list tokens_stack [unicode token]
    """
    with codecs.open(test_file,'r','utf-8') as json_content:
        test_file_map=json.load(json_content);
    
    tokens_stack_list=test_file_map['doc_str'];
    motif_stack=test_file_map['labels'];

    tokens_stack=[t for l in tokens_stack_list for t in l];    

    return motif_stack,tokens_stack;

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


def multipule_eval_for_liblinear(test_corpus_dir, feature_map_character, feature_map_numeric, feature_show, args):
    num_docs_having_motif={};
    tfidf_flag=args.tfidf;
    exno=args.experiment_no;
    
    #分類器が正しい判断をした回数を保存する．つまりCAP(gold motif tag, candidate by classifier)
    num_of_correct_decision={};
    precision_sum=0; recall_sum=0; F_sum=0; h_loss_sum=0; subset_acc_sum=0;
    ex_p_sum=0; ex_r_sum=0; ex_f_sum=0; acc_sum=0;
    classifier_return_1_sum=0;
    for test_file in f_d.load_files(test_corpus_dir):
    #============================================================    
        try:
            motif_stack,tokens_stack=f_d.file_loader_json_doc(test_file);

            out_libsvm_format(tokens_stack, feature_map_character, feature_map_numeric, feature_show, tfidf_flag);
            result_map=call_liblinear.eval_with_liblinear(exno);
            
            gold_map=f_d.make_goldmap(num_docs_having_motif,result_map,motif_stack,num_of_correct_decision,precision_sum,recall_sum,F_sum);

            h_loss_sum=f_d.calc_h_loss(result_map, gold_map, h_loss_sum);
            subset_acc_sum=f_d.calc_subset_acc(result_map, gold_map, subset_acc_sum);
            ex_p_sum, ex_r_sum, ex_f_sum, acc_sum=f_d.calc_p_r_f(result_map, gold_map, ex_p_sum, ex_r_sum, ex_f_sum, acc_sum);
            
            print 'Predict Result',result_map
            print 'Gold',gold_map
            print 'Union',[label for label,value in result_map.items() if label in gold_map and value==1]
            print u'-'*40

            classifier_return_1_sum+=f_d.get_the_num_of_1_classifier(result_map);
        #------------------------------------------------------------   
        except ValueError:
            print 'File decode error in some reason'
    #============================================================    
    num_of_files=len(f_d.load_files(test_corpus_dir));
    h_loss=h_loss_sum/num_of_files;
    subset_acc=float(subset_acc_sum)/num_of_files;
    ex_p=ex_p_sum/num_of_files;
    ex_r=ex_r_sum/num_of_files;
    ex_f=ex_f_sum/num_of_files;
    acc=acc_sum/num_of_files;
    classifier_return_1=float(classifier_return_1_sum)/num_of_files;
   
    print '-'*30;
    print 'RESULT for {} files classification'.format(len(f_d.load_files(test_corpus_dir)));
    print 'Hamming Loss:{}'.format(h_loss);
    print 'Subset Accuracy(classification accuracy):{}'.format(subset_acc);
    print 'example-based precision:{} example-based recall:{} example-based F:{} accuracy:{}'.format(ex_p, ex_r, ex_f, acc)
    print 'Average number of classifier which returns 1:{}'.format(classifier_return_1);