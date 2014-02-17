#! /usr/bin/python
#-*- coding:utf-8 -*-
__date__='2014/01/09'
import re, sys, codecs, json;

def load_model(path_to_model):
    weight_line_flag=False;
    feature_number=1;
    weight_map={};
    with codecs.open(path_to_model, 'r', 'utf-8') as lines:
        for line in lines:
            if line==u'w\n':
                weight_line_flag=True;
                continue;
            if weight_line_flag==True:
                weight_map[feature_number]=line.strip();
                feature_number+=1;
    return weight_map;

def load_featdict(path_to_featdict):
    with codecs.open(path_to_featdict, 'r', 'utf-8') as f:
        feature_map_character=json.load(f);
    return feature_map_character;

def load_featdict_numeric(path_to_numeric_dict):
    with codecs.open(path_to_numeric_dict, 'r', 'utf-8') as f:
        feature_map_numeric=json.load(f);
    return feature_map_numeric;

def checker(feature_map_character, feature_map_numeric, weight_map):
    while True:
        word=raw_input('Word to check the weight. To finish weight search, input "Q"\n');
        query=unicode(word, 'utf-8');
        if query=='Q':
            return True;
        else:
            if query in feature_map_character:
                print 'feature:weight for query {}'.format(query);
                features_character=feature_map_character[query];
                for feature_character in features_character:
                    feature_numeric=feature_map_numeric[feature_character];
                    print feature_numeric;
                    weight_value=weight_map[feature_numeric];
                    print '{}:{}'.format(feature_character, weight_value);
            else:
                print 'No weight for input query';

def sort_weight(feature_map_character, feature_map_numeric, weight_map):
    sorted_weight_f=codecs.open('./sorted_weight', 'w', 'utf-8')
    weight_number_dict={};
    for k in weight_map:
        if weight_map[k] not in weight_number_dict:
            weight_number_dict[weight_map[k]]=[k]
        else:
            weight_number_dict[weight_map[k]].append(k);
   
    number_character_dict={};
    for k in feature_map_numeric:
        if feature_map_numeric[k] not in number_character_dict:
            number_character_dict[feature_map_numeric[k]]=k;

    for k, v in sorted(weight_map.items(), key=lambda x:x[1]):
        features_number=weight_number_dict[v];
        for feature_number in features_number:
            feature_character=number_character_dict[feature_number];
            out_format='{}\t{}\t{}\t{}\n'.format(v, k, feature_number, feature_character); 
            sorted_weight_f.write(out_format);
    sorted_weight_f.close();

if __name__=='__main__':
    exno=sys.argv[1];
    label=sys.argv[2];
    
    path_to_model='./classifier/liblinear/'+label+'.liblin.model.'+exno;
    path_to_featdict='./classifier/feature_map_character/feature_map_character_1st.json.'+exno;
    path_to_numeric_dict='./classifier/feature_map_numeric/feature_map_numeric_1st.json.'+exno;

    feature_map_character=load_featdict(path_to_featdict);
    feature_map_numeric=load_featdict_numeric(path_to_numeric_dict);
    weight_map=load_model(path_to_model);

    while True:
        mode=raw_input('Choose search mode. 1 for sort mode, 2 for query search mode. To finish, input "Q"\n');
        if mode=='2':
            checker(feature_map_character, feature_map_numeric, weight_map);
        elif mode=='1':
            sort_weight(feature_map_character, feature_map_numeric, weight_map);
        elif mode=='Q':
            sys.exit('Bye Bye');
