#! /usr/bin/python
# -*- coding:utf-8 -*-
import os, sys, subprocess, codecs, re;
import file_decoder
__date__='2013/12/19';

memory_option=u'-Xmx10000m'
def out_mulan_file(test_data_list_multi, feature_map_character, feature_map_numeric, tfidf_score_map, args):
    #------------------------------------------------------------ 
    motif_vector=[unichr(i) for i in xrange(65,65+26)];
    motif_vector.remove(u'O'); motif_vector.remove(u'I');
    feature_space=len(feature_map_numeric);
    test_data_multi_in_feature_space=[];    
    #------------------------------------------------------------ 
    for test_data_list in test_data_list_multi:
        test_data_multi_in_feature_space.append(file_decoder.conv_to_featurespace(test_data_list, feature_map_character,feature_map_numeric, tfidf_score_map, args)); 
    #------------------------------------------------------------
    #arffファイルのheader部分を作成
    #xmlファイルも同時に作成
    file_contents_stack=[];
    xml_contents_stack=[];
    file_contents_stack.append(u'@relation hoge\n\n');
    xml_contents_stack.append(u'<?xml version="1.0" encoding="utf-8"?>\n<labels xmlns="http://mulan.sourceforge.net/labels">\n')
    for feature_tuple in sorted(feature_map_numeric.items(), key=lambda x:x[1]):
        file_contents_stack.append(u'@attribute {} numeric\n'.format(feature_tuple[1]));
    for motif_name in motif_vector:
        file_contents_stack.append(u'@attribute {} {{0,1}}\n'.format(motif_name));
        xml_contents_stack.append(u'<label name="{}"></label>\n'.format(motif_name));
    xml_contents_stack.append(u'</labels>');
    file_contents_stack.append(u'\n\n');
    #------------------------------------------------------------
    motif_vector_stack=[];    
    #arffファイルのデータ部分を作成
    file_contents_stack.append(u'@data\n');
    for one_instance in test_data_multi_in_feature_space:
        feature_space_for_one_instance=[0]*feature_space;
        motif_vector_numeric=[0]*len(motif_vector);
        motif_prefix=[item[0] for item in one_instance[0]];
        for m in motif_prefix:        
            tmp=motif_vector.index(m);
            motif_vector_numeric[tmp]=1;
        for feature_number_tuple in one_instance[1]:
            feature_space_for_one_instance[feature_number_tuple[0]-1]=feature_number_tuple[1];
        feature_space_for_one_instance=[str(item) for item in feature_space_for_one_instance];
        motif_vector_str=[str(item) for item in motif_vector_numeric];
        motif_vector_stack.append(motif_vector_str);        
        file_contents_stack.append(u','.join(feature_space_for_one_instance)\
                                   +u','+u','.join(motif_vector_str)+u'\n');
    file_contents_stack.append(u'\n');
    #------------------------------------------------------------
    with codecs.open('./arff_and_xml/test_{}.arff'.format(args.experiment_no), 
                     'w', 'utf-8') as f:
        f.writelines(file_contents_stack);
    with codecs.open('./arff_and_xml/test_{}.xml'.format(args.experiment_no), 
                     'w', 'utf-8') as f:
        f.writelines(xml_contents_stack);

def mulan_command(model_type, arff_train, xml_file, arff_test, modelsavepath, exno):
    args=('java {} -jar ./mulan_interface/load_model_and_eval_test.jar -train_arff {} -xml {} \
            -eval_arff {} -reduce True -reduce_method {}\
            -modelfilepath {}'.format(memory_option,
                                      arff_train,
                                       xml_file,
                                       arff_test,
                                       model_type,
                                       modelsavepath
                                      )).split();

    print 'Input command is following:{}'.format(u' '.join(args));
    subproc_args = {'stdin': subprocess.PIPE,
                    'stdout': subprocess.PIPE,
                    'stderr': subprocess.STDOUT,
                    'close_fds' : True,}
    try:
        p = subprocess.Popen(args, **subproc_args)  # 辞書から引数指定
    except OSError:
        print "Failed to execute command: %s" % args[0];
        sys.exit(1);
    out_line_stack=[];    
    output=p.stdout;
    for line in output:
        print line;
        out_line_stack.append(line);
    process_mulan_output(out_line_stack, exno);

def process_mulan_output(out_line_stack, exno):
    hamming_value='None';
    subsetacc_value='None';
    ex_based_p_value='None';
    ex_based_r_value='None';
    ex_based_f_value='None';
    r_avg_p_value='None';
    cov_value='None';
    one_err_value='None';
    r_loss_value='None';
    average_p_value='None';

    hamming_pattern=ur'Hamming Loss: ';
    subsetacc_pattern=ur'Subset Accuracy: ';
    ex_based_p=ur'Example-Based Precision: ';
    ex_based_r=ur'Example-Based Recall: ';
    ex_based_f=ur'Example-Based F Measure: ';
    r_avr_p=ur'Average Precision: ';
    cov=ur'Coverage: ';
    one_err=ur'OneError: ';
    r_loss=ur'Ranking Loss: ';
    average_p=ur'Average Precision:\s(.+)$'; 

    for line in out_line_stack:
        line=line.strip();
        if re.search(hamming_pattern, line):
            hamming_value=re.sub(hamming_pattern, ur'', line);
        if re.search(subsetacc_pattern, line):
            subsetacc_value=re.sub(subsetacc_pattern, ur'', line);
        if re.search(ex_based_p, line):
            ex_based_p_value=re.sub(ex_based_p, ur'', line);
        if re.search(ex_based_r, line):
            ex_based_r_value=re.sub(ex_based_r, ur'', line);
        if re.search(ex_based_f, line):
            ex_based_f_value=re.sub(ex_based_f, ur'', line);
        if re.search(r_avr_p, line):
            r_avg_p_value=re.sub(r_avr_p, ur'', line);
        if re.search(cov, line):
            cov_value=re.sub(cov, ur'', line);
        if re.search(one_err, line):
            one_err_value=re.sub(one_err, ur'', line);
        if re.search(r_loss, line):
            r_loss_value=re.sub(r_loss, ur'', line);
        if re.search(ur'^Mean Average Precision: .+$', line):
            average_p_value=re.sub(ur'^Mean Average Precision:\s(.+)$', ur'\1', line);
   
    header=u'exno\thamming loss\tsubset accuracy\texamplebased precision\texamplebased recall\texamplebased F\t1-Error loss\tcoverage\tRanking Error loss\taverage precision\n';
    header_flag=False;
    if os.path.exists('./result/result.tsv')==True:
        header_flag=True;
    else:
        with codecs.open('./result/result.tsv', 'w', 'utf-8') as f:
            f.write(header);

    with codecs.open('./result/result.tsv', 'a', 'utf-8') as f:
        out_tsv_format=u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(exno,
                                                                      hamming_value,
                                                                      subsetacc_value,
                                                                      ex_based_p_value,
                                                                      ex_based_r_value,
                                                                      ex_based_f_value,
                                                                      one_err_value,
                                                                      cov_value,
                                                                      r_loss_value,
                                                                      average_p_value);
        f.write(out_tsv_format);

if __name__=='__main__':
    """
    mulan_command('RAkEL', '../get_thompson_motif/classifier/mulan/exno31.arff',
                  '../get_thompson_motif/classifier/mulan/exno31.xml',
                  './arff_and_xml/test_31.arff',
                  '../get_thompson_motif/classifier/mulan/exno31.model',
                  str(31));
    """
    stack=[];
    with codecs.open(sys.argv[1], 'r', 'utf-8') as lines:
        for line in lines:
            stack.append(line);
    process_mulan_output(stack, sys.argv[2]);
