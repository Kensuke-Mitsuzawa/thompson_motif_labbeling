#! /usr/bin/python
# -*- coding:utf-8 -*-
import sys,json,codecs,config;
import construct_bigdoc_or_classifier as cbc;
#------------------------------------------------------------

def cleanup_class_stack(doc,args):
    """
    preprocessing for TMI sentences
    RETURN:list tokens_set_stack [list sentence [unicode token]]
    """
    tokens_set_stack=[];    
    for sent in doc:
        #tokens=tokenize.wordpunct_tokenize(cleaned_sentence);
        tokens_s=[config.lemmatizer.lemmatize(t.lower()) for t in sent]
        if args.stop==True:
            tokens_set_stack.append([t for t in tokens_s if t not in config.stopwords and t not in config.symbols]);
        else:
            tokens_set_stack.append(tokens_s);

    return tokens_set_stack;

def generate_document_instances(doc,filepath,alphabetTable,alphabet_label_list,dutch_training_map,args):
    """
    文書単位で訓練事例を作成する
    RETURN:map dutch_training_map {unicode label:list [document [unicode token]]}
    """
    #convert 2-dim list to 1-dim list
    tokens_in_doc=[t for s in doc for t in s];
    #lemmatize all tokens
    lemmatized_tokens_in_label=[config.lemmatizer.lemmatize(t.lower()) for t in tokens_in_doc];
    #NOT label list
    not_label_list=[a for a in alphabetTable if a not in alphabet_label_list];
   
    if args.stop==True:
        lemmatized_tokens_in_label=[t for t in lemmatized_tokens_in_label if t not in config.stopwords and t not in config.symbols];
    if config.level==1:
        for target_label in alphabet_label_list:
            dutch_training_map[target_label].append(lemmatized_tokens_in_label); 
        for not_target_label in not_label_list:
            dutch_training_map['NOT_'+not_target_label].append(lemmatized_tokens_in_label);
           
    elif config.level==2:
        alphabet_label=alphabet_label.upper();
        if alphabet_label in dutch_training_map:
            dutch_training_map[alphabet_label].append(lemmatized_tokens_in_label);
        else:
            dutch_training_map[alphabet_label]=[lemmatized_tokens_in_label];
    return dutch_training_map;

def generate_sentence_instances(doc,filepath,alphabetTable,alphabet_label_list,dutch_training_map,args):
    """
    文単位で訓練事例を作成する．とは言ったものの，文ごとにラベルがついているわけではないので，とりあえず，文書についているラベルを使って，training_mapに振り分ける
    RETURN dutch_training_map: map {unicode key : list documents_in_label [list tokens_sentence [unicode token]] }
    """
    #NOT label list
    not_label_list=[a for a in alphabetTable if a not in alphabet_label_list];
    
    for tokens_sentence in doc:
        if args.stop==True:
            tokens_sentence=[t for t in tokens_sentence if t not in config.symbols and t not in config.stopwords];
        #lemmatize all tokens
        tokens_sentence=[config.lemmatizer.lemmatize(t.lower()) for t in tokens_sentence];
       
        if not tokens_sentence==[]:
            for alphabet_label in alphabet_label_list:
                dutch_training_map[alphabet_label].append(tokens_sentence);
            for not_target_label in not_label_list:
                dutch_training_map['NOT_'+not_target_label].append(tokens_sentence);
    return dutch_training_map;
    
def load_dfd(training_map,args):
    """
    Load DFD dataset from preprocessed json file.
    DFD file path is specified by dfd_dir_path
    RETURN: map training_map {unicode label: list documents_in_label [list document [unicode token]]}
    """
    dfd_training_map={};
    #------------------------------------------------------------ 
    if config.level==1:
        #Initialize dfd_training_map with 23 labels
        #alphabetTable=[unichr(i) for i in xrange(65, 91)if chr(i) not in [u'I',u'O',u'Y']]
        for alphabet in config.alphabetTable:
            dfd_training_map[alphabet]=[];
            dfd_training_map[u'NOT_'+alphabet]=[];
    elif config.level==2:
        sys.eixt('not implemented yet');
    #------------------------------------------------------------ 
    dfd_f_list=cbc.make_filelist(config.dfd_dir_path);
    #------------------------------------------------------------ 
    for fileindex,filepath in enumerate(dfd_f_list):
        with codecs.open(filepath,'r','utf-8') as f:
            file_obj=json.load(f);
            
        alphabet_label_list=file_obj['labels'];
        doc_and_label=file_obj['doc_str'];
        doc=[sentence_label_tuple[0] for sentence_label_tuple in doc_and_label]

        if args.ins_range=='document':
            dfd_training_map=generate_document_instances(doc,filepath,config.alphabetTable,alphabet_label_list,dfd_training_map,args);
        elif args.ins_range=='sentence':
            #arowを用いた半教師あり学習のために文ごとの事例作成を行う
            dfd_training_map=generate_sentence_instances(doc,filepath,config.alphabetTable,alphabet_label_list,dfd_training_map,args);                

        if args.dev==True and fileindex==config.dev_limit:
            break;
    #------------------------------------------------------------ 
    for label in dfd_training_map:
        if label in training_map:
            training_map[label]+=dfd_training_map[label];
        else:
            training_map[label]=dfd_training_map[label];
    #------------------------------------------------------------ 
    return training_map;

def load_dfd_orig(training_map,args):
    """
    RETURN:map training_map {unicode label: list training_sets_in_label[list training_data[unicode token]]}
    """ 
    dfd_orig_training_map={};
    #------------------------------------------------------------ 
    if config.level==1:
        for alphabet in config.alphabetTable:
            dfd_orig_training_map[alphabet]=[];
            dfd_orig_training_map[u'NOT_'+alphabet]=[];
    elif config.level==2:
        sys.exit('under construction');
    #------------------------------------------------------------ 
    
    dfd_f_list=cbc.make_filelist(config.dfd_orig_path);
    #------------------------------------------------------------ 
    for fileindex,filepath in enumerate(dfd_f_list):
        if fileindex % 100==0:
            print 'Done {} th file.Total {}'.format(fileindex,len(dfd_f_list))

        with codecs.open(filepath,'r','utf-8') as f:
            file_obj=json.load(f);
        
        alphabet_label_list=file_obj['labels'];
        not_label_list=[l for l in config.alphabetTable if l not in alphabet_label_list];
        doc=file_obj['doc_str'];
        
        #convert from 2-dim list to 1-dim list  
        tokens_in_doc=[t for sent in doc for t in sent];
        #dutch stemming
        tokens_in_doc=[config.stemmer.stem(t) for t in tokens_in_doc]; 
        
        for target_label in alphabet_label_list:
            dfd_orig_training_map[target_label].append(tokens_in_doc);
        for not_label in not_label_list:
            dfd_orig_training_map['NOT_'+not_label].append(tokens_in_doc);
    #------------------------------------------------------------ 
    for label,content in dfd_orig_training_map.items():
        training_map[label]=content; 
    #------------------------------------------------------------ 

    return training_map;

def load_tmi(training_map,args):
    """
    Load TMI training data from specified json dir path.
    TMI dir path is specified by tmi_dir_path.
    RETURN:map training_map {unicode label: list training_sets_in_label[list training_data[unicode token]]}
    """
    tmi_training_map={};
    #------------------------------------------------------------ 
    if config.level==1:
        for alphabet in config.alphabetTable:
            tmi_training_map[alphabet]=[];
            tmi_training_map[u'NOT_'+alphabet]=[];
    elif config.level==2:
        sys.exit('under construction');
    #------------------------------------------------------------ 
    tmi_f_list=cbc.make_filelist(config.tmi_dir_path);
    #------------------------------------------------------------ 
    for fileindex,filepath in enumerate(tmi_f_list):
        with codecs.open(filepath,'r','utf-8') as f:
            file_obj=json.load(f);
        
        alphabet_label_list=file_obj['labels'];
        not_label_list=[l for l in config.alphabetTable if l not in alphabet_label_list];
        doc_and_label=file_obj['doc_str'];
        doc=[sentence_label_tuple[0] for sentence_label_tuple in doc_and_label];

        if args.big_TMI==False:
            tokens_set_stack=cleanup_class_stack(doc,args);
        elif args.big_TMI==True:
            tokens_set_stack=cleanup_class_stack(doc,args);
            tokens_set_stack=[t for one_definition in tokens_set_stack for t in one_definition];

        for target_label in alphabet_label_list: 
            tmi_training_map[target_label]+=(tokens_set_stack);
        for not_target_label in not_label_list:
            tmi_training_map['NOT_'+not_target_label]+=(tokens_set_stack);
        
        if args.dev==True and fileindex==config.dev_limit:
            break;
    #------------------------------------------------------------ 
    for key, value in tmi_training_map.items():
        if key in training_map:    
            training_map[key]+=value;
        else:
            training_map[key]=value;
    #------------------------------------------------------------ 

    return training_map;

def label_converter(label):
    #if level==1:
    label=label.strip();
    target_label=label[0]
    return target_label; 

def load_resource_general_doc_based(dataset_dirpath,training_map,args):
    """
    Load dataset from specified preprocessed json file.
    file path is specified by additional_resource_list 
    RETURN: map training_map {unicode label: list document[list [unicode token]]}
    """
    additional_training_map={};
    #------------------------------------------------------------ 
    if config.level==1:
        for alphabet in config.alphabetTable:
            additional_training_map[alphabet]=[];
            additional_training_map[u'NOT_'+alphabet]=[];
    elif config.level==2:
        sys.eixt('not implemented yet');
    #------------------------------------------------------------ 
    f_list=cbc.make_filelist(dataset_dirpath);
    #------------------------------------------------------------ 
    for fileindex,filepath in enumerate(f_list):
        with codecs.open(filepath,'r','utf-8') as f:
            file_obj=json.load(f);
        
        alphabet_label_list=file_obj['labels'];
        alphabet_label_list=[label_converter(label) for label in alphabet_label_list];
        print filepath
        print alphabet_label_list
        doc_and_label=file_obj['doc_str'];
        doc=[sentence_label_tuple[0] for sentence_label_tuple in doc_and_label];

        if args.ins_range=='document':
            additional_training_map=generate_document_instances(doc,filepath,config.alphabetTable,alphabet_label_list,additional_training_map,args);
        """
        elif args.ins_range=='sentence':
            #arowを用いた半教師あり学習のために文ごとの事例作成を行う
            additional_training_map=generate_sentence_instances(doc,filepath,alphabetTable,alphabet_label_list,dfd_training_map,args);                
            """
        if args.dev==True and fileindex==config.dev_limit:
            break;
    #------------------------------------------------------------ 
    for label in additional_training_map:
        if label in training_map:
            training_map[label]+=additional_training_map[label];
        else:
            training_map[label]=additional_training_map[label];
    #------------------------------------------------------------ 
    
    return training_map;

def load_resource_general_sent_based(dataset_dirpath,sentence_labeled_map,args):
    """
    INPUT: map sentence_labeled_map {unicode label : list training_data [list tokens_in_sentence [unicode token]]}
    RETURN: map sentence_labeled_map {unicode label : list training_data [list tokens_in_sentence [unicode token]]}
    """        
    training_filelist=cbc.make_filelist(dataset_dirpath);
    for training_f in training_filelist:
        with codecs.open(training_f,'r','utf-8') as json_content:
            training_document_data=json.load(json_content);
        
        if training_document_data['instance_range']=='sentence':
            for sentence_label_tuple in training_document_data['doc_str']:            
                tmi_label_list=sentence_label_tuple[1];
                if not tmi_label_list==None:
                    if config.level==1:
                        label_1st_list=[(l.strip())[0] for l in tmi_label_list];
                        not_label_list=[a for a in config.alphabetTable if a not in label_1st_list];
                    
                    tokens_in_sentence=[config.lemmatizer.lemmatize(t.lower()) for t in sentence_label_tuple[0]];
                    if args.stop==True:
                        tokens_in_sentence=[t for t in tokens_in_sentence if t not in config.stopwords and t not in config.symbols];
    
                    for label_1st in label_1st_list:
                        sentence_labeled_map[label_1st].append(tokens_in_sentence);
                    for not_label in not_label_list:
                        sentence_labeled_map['NOT_'+not_label].append(tokens_in_sentence);
                        
        elif training_document_data['instance_range']=='document':
            print '[Warning!] This dataset is not sentence based training data.'
            sys.exit();
        
        else:
            print '[Warning!] This dataset is neither "document" or "sentence". instance_range must be either "document" or "sentence". Please check it.'
            sys.exit();
                    
    return sentence_labeled_map;