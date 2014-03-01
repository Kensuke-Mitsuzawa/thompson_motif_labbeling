#! /usr/bin/python
# -*- coding:utf-8 -*-
__date__='2014/3/1'

import sys,json,re,codecs,os,glob;
from nltk.corpus import stopwords;
from nltk import stem;
from nltk import tokenize; 
import nltk;
import return_range;

lemmatizer = stem.WordNetLemmatizer();
stopwords = stopwords.words('english');
symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')'];

#specify raw document dirpath here
raw_doc_path='../../dutch_folktale_corpus/dutch_folktale_database_translated_kevin_system/translated_test/'
#specify raw document dirpath here
raw_doc_path_2='../parsed_json/'
#specify IFN raw document dirpath here
#raw_doc_path_ifn='../../corpus_dir/translated_google/'
raw_doc_path_ifn='../../corpus_dir/translated_google_new_format/'
#specify DFD original document dirpath here
raw_doc_path_dfd_orig='../../dutch_folktale_corpus/given_script/top_dutch/top_document_test/';

#specify json document dirpath here
json_doc_path='../test_resource/dfd/'
#specify json document dirpath here
json_doc_path_2='../training_resource/tmi/'
#specify json document of IFN dirpath here
json_doc_path_ifn='../test_resource/ifn/'
#specify json document save path of DFD original
json_doc_path_dfd_orig='../test_resource/dfd_orig/'

def make_filelist(dir_path):
    file_list=[];
    for root, dirs, files in os.walk(dir_path):
        for f in glob.glob(os.path.join(root, '*')):
            file_list.append(f);
    return file_list;
    
def generate_sentence_instances(file_obj):
    """
    文単位で訓練事例を作成する
    RETURN list document [tuple sentence_and_label (list sentence [unicode token], label unicode)]
    """
    document=[];
    sentences=file_obj.readlines();
    file_obj.close();
    for s in sentences:
        tokens_sentence=tokenize.wordpunct_tokenize(s);
        document.append( (tokens_sentence,None) ); 
    
    return document;    

def json_converter_1(raw_doc_list):
    """
    ファイル名がラベルになっている場合のみ有効
    ファイル名の形式 ラベル_ラベル2_ラベル3
    文書単位の場合のみ有効．つまりDFDに対して有効
    This function converts raw document into json format.
    The File format must be underbar concatenated TMIlabels like TMIlabel1_TMIlabel2_TMIlabel3_...
    RETURN None
    OutJsonFormat map json_f_str {'labels':list [unicode label],'doc_str':list document[list sentence[unicode token]]} 
    """
    for f in raw_doc_list:
        json_f_str={}; 
        labels=[];        
        filename=os.path.basename(f);
        [labels.append(l) for l in filename.split('_') if re.search(r'[A-Z]',l)];

        f_obj=codecs.open(f,'r','utf-8');
        doc_str=generate_sentence_instances(f_obj);        

        json_f_str['labels']=labels;
        json_f_str['doc_str']=doc_str;
        json_f_str['instance_range']='document'

        with codecs.open(json_doc_path+filename,'w','utf-8') as f:
            json.dump(json_f_str,f,ensure_ascii='False',indent=4);

def json_converter_2():
    """
    json_converter_2はTMI tree用    
    """            
    all_thompson_tree=return_range.load_all_thompson_tree(raw_doc_path_2);

    for key_index, key_1st in enumerate(all_thompson_tree):
        json_f_str={};        
        document=[];
        parent_node=key_1st;

        class_training_stack=construct_class_training_1st(parent_node, all_thompson_tree);
        for label_description_tuple in class_training_stack:
            description=label_description_tuple[1];
            s=tokenize.wordpunct_tokenize(description);
            document.append((s,parent_node));
        
        json_f_str['labels']=[parent_node];
        json_f_str['doc_str']=document;
        json_f_str['instance_range']='sentence'

        filename=parent_node+'.json';
        with codecs.open(json_doc_path_2+filename,'w','utf-8') as f:
            json.dump(json_f_str,f,ensure_ascii='False',indent=4);
        """        
        tokens_set_stack=cleanup_class_stack(class_training_stack, stop);
        
        print u'-'*30;
        print u'Training instances for {} from thompson tree:{}'.format(key_1st,len(tokens_set_stack));
        num_of_training_instance+=len(tokens_set_stack);
        #------------------------------------------------------------ 
        #作成した文書ごとのtokenをtrainingファイルを管理するmapに追加
        #TFIDFがTrueだろうが，Falseだろうが関係なく，ここは実行される
        if key_1st in thompson_training_map:
            thompson_training_map[key_1st]+=tokens_set_stack;
        else:
            thompson_training_map[key_1st]=tokens_set_stack;
        if dev_mode==True and key_index==dev_limit:
            break;
    #------------------------------------------------------------ 
    #素性をunigram素性にする
    if tfidf==False:
        for label in thompson_training_map:
            tokens_set_stack=thompson_training_map[label];
            #文字情報の素性関数を作成する
            feature_map_character=make_feature_set(feature_map_character,
                                                   label, tokens_set_stack, 'thompson', stop, args);
                                                   
    """

def json_converter_ifn_body(file_path):
    """
    Raw into json converter for IFN 
    Read and construct tokens list from test file.
    ARGS: file_path(file which you want to classify)
    """
    #queryファイルの読み込み
    line_flag=False;
    motif_flag=False;
    attach_flag=False;
    motif_stack=[];
    line_stack=[];
    with codecs.open(file_path, 'r', 'utf-8') as lines:
        for line_no,line in enumerate(lines):
            line_number=line_no+1;
            
            if line==u'\n':
                continue;
            if re.search(ur'#motif',line):#line==u'#motif\n':
                motif_flag=True;
                continue;
            elif line==u'#text\n':
                motif_flag=False;
                line_flag=True;
                continue;
            if motif_flag==True and line_flag==False:
                #get TMI label and its range tuple (TMI label, start_position, end_position)
                label, start_position, end_position=line.strip().split(u'\t') 
                label_range_tuple=(label,int(start_position),int(end_position))
                motif_stack.append(label_range_tuple);

            if line_flag==True and motif_flag==False:
                for label_range_tuple in motif_stack:
                    if line_number==label_range_tuple[1]:
                        label=label_range_tuple[0];
                        attach_flag=True;

                    if line_number==label_range_tuple[2]:
                        label=None;
                        attach_flag=False;
               
                tokens_in_line=tokenize.wordpunct_tokenize(line.strip());
                tokens_in_line=[t.lower() for t in tokens_in_line];
                if attach_flag==True:
                    sentence_label_tuple=(line.strip(), label);
                    line_stack.append(sentence_label_tuple);
                else:
                    sentence_label_tuple=(line.strip(),None);
                    line_stack.append(sentence_label_tuple)

    #tokens_stack=[tokenize.wordpunct_tokenize(line) for line in line_stack]
    #tokens_stack=[[t.lower() for t in l] for l in tokens_stack]
    #ここではstopwordsの除去はしない
    #if eliminate_stop==True: 
    #    tokens_stack=[[t for t in l if t not in stopwords and t not in symbols] for l in tokens_stack]
    #配列を二次元から一次元に落とす．ついでにlemmatizeも行う．
    #tokens_stack=[lemmatizer.lemmatize(t) for line in tokens_stack for t in line];
    return line_stack,motif_stack;

def json_converter_ifn_head():
    file_list=make_filelist(raw_doc_path_ifn);
    for f_path in file_list:
        filename=os.path.basename(f_path);
        
        json_f_str={};
        tokens_stack,motif_stack=json_converter_ifn_body(f_path)
        motifs_in_doc=[each_tuple[0] for each_tuple in motif_stack];
        
        json_f_str['labels']=motifs_in_doc;
        json_f_str['doc_str']=tokens_stack;
        json_f_str['instance_range']='sentence'
        
        with codecs.open(json_doc_path_ifn+filename,'w','utf-8') as f:
            json.dump(json_f_str,f,ensure_ascii='False',indent=4);

def def_tokenize(filepath):
    file_obj=codecs.open(filepath,'r','utf-8'); 
    document_unicode=file_obj.read(); 
    tokenized_document=nltk.tokenize.wordpunct_tokenize(document_unicode);
    file_obj.close(); 
    return tokenized_document;

def json_converter_dfd_orig():
    file_list=make_filelist(raw_doc_path_dfd_orig);
    for filepath in file_list:
        dfd_orig_one_doc_map={}
        #ラベルの分解処理
        label_list=(os.path.basename(filepath)).split('_')[:-1];
        #tokenized_documentはリスト型
        tokenized_document=def_tokenize(filepath);
        #オランダ語はわからんが，一応すべて小文字化はしておく
        #さらにモチーフラベルのタプルを作成(全部Noneだけど)
        tokenized_document=[t.lower() for t in tokenized_document];
        
        dfd_orig_one_doc_map['labels']=label_list;
        dfd_orig_one_doc_map['doc_str']=(tokenized_document,None);
        dfd_orig_one_doc_map['instance_range']='document'

        with codecs.open(json_doc_path_dfd_orig+os.path.basename(filepath),'w','utf-8') as json_content:
            json.dump(dfd_orig_one_doc_map,json_content,ensure_ascii=False,indent=4);

def construct_class_training_1st(parent_node, all_thompson_tree):
    """
    一層目を指定した時に，一層目の各ラベルに属する単語から訓練事例を作って返す
    RETURN: list label_description_set [tuple (unicode label, unicode description)]
    """
    class_training_stack=[];
    for child_tree_key in all_thompson_tree[parent_node]:
        for grandchild_tree_key in all_thompson_tree[parent_node][child_tree_key]:
            if re.search(ur'[A-Z]_\d+_\d+_\w+', grandchild_tree_key):
                for child_of_grandchild in all_thompson_tree[parent_node][child_tree_key][grandchild_tree_key]:
                    target_subtree_map=all_thompson_tree\
                        [parent_node][child_tree_key][grandchild_tree_key][child_of_grandchild];
                    class_training_stack=\
                        extract_leaf_content_for_class_training(parent_node,
                                                                target_subtree_map,
                                                                     class_training_stack);

            elif re.search(ur'\d+', grandchild_tree_key):
                target_subtree_map=all_thompson_tree[parent_node][child_tree_key][grandchild_tree_key];
                class_training_stack=extract_leaf_content_for_class_training(parent_node,
                                                                             target_subtree_map,
                                                                                  class_training_stack)
    return class_training_stack;    
    
def extract_leaf_content_for_class_training(class_lebel, target_subtree_map, class_training_stack):
    if not target_subtree_map['child']==[]:
        for child_of_grandchild in target_subtree_map['child']:
            if not child_of_grandchild[1]==None:
                class_training_stack.append((class_lebel,
                                           child_of_grandchild[1].replace(u'\r\n', u'.').strip())); 
    else:
        outline_text=target_subtree_map['content'][1];
        if not outline_text==None:
            class_training_stack.append((class_lebel,
                                         outline_text.replace(u'\r\n', u'.').strip()));

    return class_training_stack;
    
if __name__=='__main__':
    raw_doc_list=make_filelist(raw_doc_path);
    json_converter_1(raw_doc_list); 
    json_converter_2();
    json_converter_ifn_head();
    json_converter_dfd_orig();
