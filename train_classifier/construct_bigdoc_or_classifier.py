#! /usr/bin/python
# -*- coding:utf-8 -*-
__date__='2014/3/1';

import argparse,codecs,os,glob,json,sys;
sys.path.append('../');
import config as cfg;
import return_range, mulan_module, liblinear_module, bigdoc_module;
import feature_create;
import file_load_module;
from nltk.corpus import stopwords;
from nltk import tokenize; 
from nltk.stem import SnowballStemmer;
stemmer=SnowballStemmer("dutch");
#------------------------------------------------------------
#Idea number of TFIDF
tfidf_idea=2;        
#------------------------------------------------------------

def make_filelist(dir_path):
    file_list=[];
    for root, dirs, files in os.walk(dir_path):
        for f in glob.glob(os.path.join(root, '*')):
            file_list.append(f);
    return file_list;

def make_feature_set(feature_map, tokens_set_stack,args):
    """
    素性関数を作り出す（要はただのmap）
    RETURN: map feature_map {unicode token: list feature_list [unicode feature]}
    """
    stop=args.stop;

    for token_instance in tokens_set_stack:
        for token in token_instance:
            #ドメインごとの素性を登録
            #character_feature=u'{}_unigram'.format(token);
            character_feature=u'{}_BOW'.format(token);            
            if stop==True and token not in cfg.stopwords and token not in cfg.symbols:
                if token in feature_map and character_feature not in feature_map[token]:
                    feature_map[token].append(character_feature);
                elif token in feature_map and character_feature in feature_map[token]:
                    pass; 
                else:
                    feature_map[token]=[character_feature];
            elif stop==False:
                if token in feature_map and character_feature not in feature_map[token]:
                    feature_map[token].append(character_feature);
                elif token in feature_map and character_feature in feature_map[token]:
                    pass; 
                else:
                    feature_map[token]=[character_feature];

    return feature_map;

def make_numerical_feature(feature_map_character):
    feature_map_numeric={};
    feature_num_max=1;
    for token_key in feature_map_character:
        for feature in feature_map_character[token_key]:
            feature_map_numeric[feature]=feature_num_max; 
            feature_num_max+=1;
    return feature_map_numeric;

def construct_classifier_for_1st_layer(all_thompson_tree,args):
    """
    document based classifier
    """
    exno=str(args.experiment_no);
    training_map={};
    motif_vector=[unichr(i) for i in xrange(65, 91)if chr(i) not in [u'I',u'O',u'Y']]
    for motif_label in motif_vector:
        training_map[motif_label]=[];
        training_map['NOT_'+motif_label]=[];
    tfidf_score_map={};
    feature_map_character={};
    #============================================================ 
    #追加のデータ・セットがあった場合にこのコードを使うようにするうまい仕組みを考えないとね
    for dataset_dirpath in cfg.additional_resource_list:
        training_map=file_load_module.load_resource_general_doc_based(dataset_dirpath,training_map,args);
    #============================================================ 
    if args.dutch==True:
        training_map=file_load_module.load_dfd(training_map,args);
        print 'loaded DFD data'
    #============================================================ 
    if args.thompson==True:
        training_map=file_load_module.load_tmi(training_map,args);
        print 'loaded TMI data';
    #============================================================ 
    if args.dutch_original==True:
        training_map=file_load_module.load_dfd_orig(training_map,args);
        print 'loaded DFD(dutch) data'
    #============================================================ 
    #Use TFIDF feature. There's some feature selection way. This way is controlled by flag
    if args.tfidf==True:
        print 'TDIDF idea number {}'.format(tfidf_idea);
        if tfidf_idea==1:
            feature_map_character, tfidf_score_map=feature_create.create_tfidf_feat_idea1(training_map, 
                                                                                          feature_map_character,
                                                                                          args);
        elif tfidf_idea in [2,3,4,5]:                                                       
            feature_map_character, tfidf_score_map=feature_create.create_tfidf_feat_idea2_4(training_map,
                                                                                            feature_map_character,
                                                                                            tfidf_idea, args);
    else:
        # If feature is Bag-of-words
        for label in training_map:
            tokens_set_stack=training_map[label];
            feature_map_character=make_feature_set(feature_map_character,tokens_set_stack,args);
    #============================================================  
    #作成した素性辞書をjsonに出力(TFIDF)が空の時は空の辞書が出力される
    with codecs.open('../classifier/tfidf_weight/tfidf_word_weight.json.'+exno,'w','utf-8') as f:
        json.dump(tfidf_score_map, f, indent=4, ensure_ascii=False);
    with codecs.open('../classifier/feature_map_character/feature_map_character_1st.json.'+exno,'w','utf-8') as f:
        json.dump(feature_map_character, f, indent=4, ensure_ascii=False);
    #ここで文字情報の素性関数を数字情報の素性関数に変換する
    feature_map_numeric=make_numerical_feature(feature_map_character);
    
    with codecs.open('../classifier/feature_map_numeric/feature_map_numeric_1st.json.'+exno,'w','utf-8') as f:
        json.dump(feature_map_numeric, f, indent=4, ensure_ascii=False);

    feature_space=len(feature_map_numeric);
    print u'The number of feature is {}'.format(feature_space)
   
    if args.training=='liblinear':
        #liblinearを使ったモデル作成
        liblinear_module.out_to_libsvm_format(training_map, 
                            feature_map_numeric, 
                            feature_map_character,
                            args.tfidf,
                            tfidf_score_map,
                            exno, tfidf_idea, args);
                    
    elif args.training=='mulan':
        dutch_dir_path='../../dutch_folktale_corpus/dutch_folktale_database_translated_kevin_system/translated_train/'
        #mulanを使ったモデル作成
        #training_mapは使えないので新たにデータ構造の再構築をする（もったいないけど）
        #thompson木は元々マルチラベルでもなんでもないので，使わない
        training_data_list=create_multilabel_datastructure(dutch_dir_path, args); 
        if args.thompson==True:
            training_data_list=create_multilabel_datastructure_single(training_data_list,
                                                   thompson_training_map,
                                                   args);
        mulan_module.out_to_mulan_format(training_data_list, 
                            feature_map_numeric, 
                            feature_map_character,
                            tfidf, tfidf_score_map,
                            feature_space, 
                            motif_vector, tfidf_idea, args);

def create_multilabel_datastructure_single(training_data_list, thompson_training_map, args):
    """
    mulan用に訓練用のデータを作成する．
    PARAM dir_path:訓練データがあるディレクトリパス args:argumentparserの引数
    RETURN 二次元配列  list [tuple (list [unicode ラベル列], list [unicode token])] 
    """
    for label in thompson_training_map:
        for one_instance in thompson_training_map[label]:
            #tokens_in_label=[t for doc in thompson_training_map[label] for t in doc];
            training_data_list.append( (label, one_instance) );
    return training_data_list;

def create_multilabel_datastructure(dir_path, args):
    """
    mulan用に訓練用のデータを作成する．
    PARAM dir_path:訓練データがあるディレクトリパス args:argumentparserの引数
    RETURN 二次元配列  list [tuple (list [unicode ラベル列], list [unicode token])] 
    """
    training_data_list=[];
    level=args.level;
    for fileindex, filepath in enumerate(make_filelist(dir_path)):
        if level==1:
            alphabet_label_list=(os.path.basename(filepath)).split('_')[:-1];
        elif level==2:
            alphabet_label=(os.path.basename(filepath))[0];
        file_obj=codecs.open(filepath, 'r', 'utf-8');
        tokens_in_label=tokenize.wordpunct_tokenize(file_obj.read());
        file_obj.close();
        lemmatized_tokens_in_label=[cfg.lemmatizer.lemmatize(t.lower()) for t in tokens_in_label];
        if args.stop==True:
            lemmatized_tokens_in_label=\
                    [t for t in lemmatized_tokens_in_label if t not in stopwords and t not in cfg.symbols];
        if level==1:
            #ラベル列，token列のタプルにして追加
            training_data_list.append(([alphabet_label.upper() for alphabet_label in alphabet_label_list],
                                      lemmatized_tokens_in_label));
        #level2のことは後で考慮すればよい
        """
        elif level==2:
            alphabet_label=alphabet_label.upper();
            if alphabet_label in training_map:
                dutch_training_map[alphabet_label].append(lemmatized_tokens_in_label);
            else:
                dutch_training_map[alphabet_label]=[lemmatized_tokens_in_label];
        if dev_mode==True and fileindex==dev_limit:
            break;
            """
    return training_data_list;

def construct_classifier_for_1st_sent_based(all_thompson_tree,args):
    """
    文ごとにラベル付けをする分類器の構築
    必ず，trainingにthompsonを利用する
    追加のデータ・セットがある場合はconfig.pyのadditional_resource_general_dfdに記載
    #TODO せっかくsemi-supervisedの利点があるのだから，ラベルなしの文書をtrainingに利用できる仕組みを整えたらどうか？
    """
    exno=str(args.experiment_no);
    
    sentence_labeled_map={};    
    training_map={};
    tfidf_score_map={};
    feature_map_character={};
    #============================================================ 
    if args.thompson==True:
        sentence_labeled_map=file_load_module.load_tmi(sentence_labeled_map,args);
    else:
        sys.exit('You must specify -thompson to construct sentence based classifier');
    #============================================================ 
    #追加のデータ・セットがあった場合にこのコードを使うようにするうまい仕組みを考えないとね
    #Read additional dataset which labeled per sentences. Loaded sentences are added to sentence_labeled_map
    for dataset_dirpath in cfg.additional_resource_list:
        sentence_labeled_map=file_load_module.load_resource_general_sent_based(dataset_dirpath,sentence_labeled_map,args);        
    #============================================================ 
    if args.dutch==True:
        #文単位でデータを読み込みして，training_mapに保存
        #load DFD dataset.
        training_map=file_load_module.load_dfd(training_map,args);
    #============================================================ 
    if args.dutch_original==True:
        #training_map=load_dfd_orig(training_map,args);
        sys.exit("under const")
    #============================================================ 
    #これはあまりよくない条件なので，あとで書き換えておくこと
    if not args.tfidf==True:
        # If feature is Bag-of-words
        for label,tokens_set_stack in sentence_labeled_map.items():
            feature_map_character=make_feature_set(feature_map_character,tokens_set_stack,args);
        for label,tokens_set_stack in training_map.items():
            feature_map_character=make_feature_set(feature_map_character,tokens_set_stack,args);
    #============================================================     
    #作成した素性辞書をjsonに出力(TFIDF)が空の時は空の辞書が出力される
    with codecs.open('../classifier/tfidf_weight/tfidf_word_weight.json.'+exno,'w','utf-8') as f:
        json.dump(tfidf_score_map, f, indent=4, ensure_ascii=False);
    with codecs.open('../classifier/feature_map_character/feature_map_character_1st.json.'+exno,'w','utf-8') as f:
        json.dump(feature_map_character, f, indent=4, ensure_ascii=False);
    #ここで文字情報の素性関数を数字情報の素性関数に変換する
    feature_map_numeric=make_numerical_feature(feature_map_character);
    
    with codecs.open('../classifier/feature_map_numeric/feature_map_numeric_1st.json.'+exno,'w','utf-8') as f:
        json.dump(feature_map_numeric, f, indent=4, ensure_ascii=False);

    feature_space=len(feature_map_numeric);
    print u'The number of feature is {}'.format(feature_space)
   
    if args.training=='arow':
        liblinear_module.out_to_libsvm_format_arow(training_map,
                            sentence_labeled_map,
                            feature_map_numeric, 
                            feature_map_character,
                            args.tfidf,
                            tfidf_score_map,
                            exno, tfidf_idea, args);
    elif args.training=='logistic':
        liblinear_module.out_to_libsvm_format_logistic(training_map, 
                            sentence_labeled_map,
                            feature_map_numeric, 
                            feature_map_character,
                            args.tfidf,
                            tfidf_score_map,
                            exno, tfidf_idea, args);

def doc_or_sent_based(all_thompson_tree,args):
    """    
    call different function by args.ins_range
    """
    if args.ins_range=='document':
        construct_classifier_for_1st_layer(all_thompson_tree,args)
    
    elif args.ins_range=='sentence':
        construct_classifier_for_1st_sent_based(all_thompson_tree,args)

def main(level, mode, all_thompson_tree, stop, dutch, thompson, tfidf, exno, args):
    #result_stack=return_range.find_sub_tree(input_motif_no, all_thompson_tree) 
    #print 'The non-terminal nodes to reach {} is {}'.format(input_motif_no, result_stack);
    if mode=='big':
        bigdoc_module.big_doc_main(all_thompson_tree, args);   
    elif mode=='class':
        if level==1:
            doc_or_sent_based(all_thompson_tree,args);
        elif level==2:
            sys.exit('Under construction now')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='');
    parser.add_argument('-level', '--level',
                        help='level which you want to construct big doc.', default=1)
    #----------------------------------------------------------- 
    parser.add_argument('-mode', '--mode',
                        help='classification problem(class) or big-document(big)', required=True);
    parser.add_argument('-stop',
                        help='If added, stop words are eliminated from training file', action='store_true');
    #----------------------------------------------------------- 
    parser.add_argument('-dutch', 
                        help='If added, document from dutch folktale database is added to training corpus', 
                        action='store_true');
    parser.add_argument('-dutch_original', 
                        help='If added, document from original dutch document is added to training corpus', 
                        action='store_true');
    parser.add_argument('-AIC',
                        help='If added, American Indian folktales corpus is used as training',
                        action='store_true');
    parser.add_argument('-thompson', 
                        help='If added, outline from thompson tree is added to training corpus', 
                        action='store_true');
    parser.add_argument('-big_TMI',
                        help='if added TMI definition sentences are concatenated into big document',
                        action='store_true');
    #----------------------------------------------------------- 
    parser.add_argument('-ins_range',
                        help='select a range of one instance. "document" or "sentence"',
                        default='document');
    parser.add_argument('-tfidf',
                        help='If added, tfidf is used for feature scoring instead of unigram feature', 
                        action='store_true');
    parser.add_argument('-tfidf_type',
                        help='choose tyidf type, normal or nishimura.Default is normal', 
                        default='normal');
    parser.add_argument('-normalization_term','--n_t',
                        help='2 for L2 normalization or 5 for L1 normalization', 
                        default='2');
    parser.add_argument('-exno', '--experiment_no',
                        help='save in different file',
                        default=0);
    parser.add_argument('-arow_thres', '--arow_thres',
                        help='The threshold to controll semi-supervised training with arow classifier',
                        default=0.0);
    parser.add_argument('-dev', '--dev',
                        help='developping mode',
                        action='store_true');
    parser.add_argument('-training_amount', '--training_amount',
                        help='The ratio of training amount',
                        default=0.95);
    #parser.add_argument('-easy_domain2', '--easy_domain2',
    #                    help='use easy domain idea2, which is domain adaptation of labels',
    #                    action='store_true');
    parser.add_argument('-training', help='which training tool? liblinear or mulan or arow?');
    parser.add_argument('-mulan_model', help='which model in mulan library.\
                        RAkEL, RAkELd, MLCSSP, HOMER, HMC, ClusteringBased, Ensemble etc.',
                        default=u'');
    parser.add_argument('-reduce_method', help='which method use to reduce feature dimention?\
                        labelpower, copy, binary',
                        default='binary');
    parser.add_argument('-save_exno', help='not in use', default=u'');
    args=parser.parse_args();
    dir_path='../parsed_json/'
    #------------------------------------------------------------    
    print 'Normalization term:{}'.format(args.n_t);
    #------------------------------------------------------------    
    if float(args.training_amount)>=1.0:
        sys.exit('[Warning] -training_amount must be between 0-1(Not including 1)');
    #------------------------------------------------------------    
    """
    if args.easy_domain==True:
        if not (args.dutch==True and args.thompson==True):
            sys.exit('[Warning] You specified easy_domain mode. But there is only one domain');
            """
    #------------------------------------------------------------
    if args.training=='mulan' and args.mulan_model==u'':
        sys.exit('[Warning] mulan model is not choosen');
    #------------------------------------------------------------    
    if args.dutch==True and args.dutch_original==True:
        sys.exit('[Warning] Must not specify both of -dutch:translated dutch and -dutch_original');
    #------------------------------------------------------------
    if args.mode=='class':
        if args.training==u'mulan': 
            pass;
        elif args.training==u'liblinear':
            pass;
        elif args.training==u'arow':
            pass;
        elif args.training==u'logistic':
            pass;
        else:
            sys.exit('[Warning] training tool is not choosen(mulan or liblinear)');
    #------------------------------------------------------------
    if args.ins_range=='document':
        if args.training=='arow':
            sys.exit('arow is only for sentence based');
        elif args.training=='logistic':
            sys.exit('logistic is only for sentence based');
    elif args.ins_range=='sentence':
        if args.training=='liblinear':
            sys.exit('liblin. is only for doc. based');
        elif args.training=='mulan':
            sys.exit('mulan is only for doc. based')
    #------------------------------------------------------------
    # I frequently forget add -stop flag, thus added this option
    if args.stop==False:
        print 'Do you really trainig without eliminating stopwords?';
        decision=raw_input('If you eliminate stopwords, type Y. If not, type N\n'); 
        if decision=='Y':
            args.stop=True;
        else:
            pass;
    #------------------------------------------------------------
    all_thompson_tree=return_range.load_all_thompson_tree(dir_path);
    result_stack=main(args.level, 
                      args.mode, 
                      all_thompson_tree, 
                      args.stop, 
                      args.dutch, 
                      args.thompson, 
                      args.tfidf,
                      args.experiment_no,
                      args);