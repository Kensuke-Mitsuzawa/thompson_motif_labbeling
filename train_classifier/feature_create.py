# -*- coding: utf-8 -*-
"""
__author__: kensuke-mi
__date__="2014/2/12"
素性作成方法をアイディアごとに，関数へ切り分けしている
"""
import tf_idf;
from nltk import stem;
from nltk.corpus import stopwords;
lemmatizer = stem.WordNetLemmatizer();
stopwords = stopwords.words('english');
symbols = ["'", '"', '`', '.', ',', '-', '!', '?', ':', ';', '(', ')'];

def create_tfidf_feat_idea1(training_map, feature_map_character, args):
    """
    TFIDFにて素性選択をするidea-1
    文書集合全体で素性選択を行う．
    疑似文書（ラベル）の作成→ラベル文書ごとにTFIDF計算→全文書でのスコアを足す→全単語のスコアを足す→
    閾値を設定する→素性選択
    """
    import math;
    stop=args.stop;
    tfidf_type=args.tfidf_type;
    L2_flag=True;
    num_of_discarded_feat=0;
    #TFIDF空間にペルシア語コーパスの語も含めるか？
    persian_flag=False;    
    #単語スコアを保存しておくmap
    word_score_map={};
    #------------------------------------------------------------
    print 'TFIDF(Idea-1) score calculating'
    print 'L2 flag:{} Persian flag:{}'.format(L2_flag, persian_flag);
    #------------------------------------------------------------
    #訓練コーパスからラベルごとに疑似文書を作成する
    #TFIDF計算の入力にはlist documents [list document [ unicode token ] ]が必要
    if tfidf_type=='normal':
        print 'use tfidf(normal)'
        all_training_instances=[];
        tmp_document_controll_map={};
        for subdata in training_map:
            for k, v in sorted(training_map[subdata].items()):
                label_document=[t for doc in v for t in doc];
                if k in tmp_document_controll_map:
                    tmp_document_controll_map[k]+=label_document;
                else:
                    tmp_document_controll_map[k]=label_document
     
        for k, v in sorted(tmp_document_controll_map.items()):
            all_training_instances.append(tmp_document_controll_map[k]);
        w_dt_maps_list=tf_idf.tf_idf_interface(all_training_instances);
    #------------------------------------------------------------
    elif args.tfidf_type=='nishimura':
        #TFIDF計算の入力にはlist documents [list document [ list sub-document [unicode token] ] ]が必要
        #IDFの計算にはlist documents[unicode token]が必要だが，これは関数内で自動で変換してくれる
        print 'use tfidf(nishimura)'
        all_training_instances=[];
        tmp_document_controll_map={};
        for subdata in training_map:
            for k, v in sorted(training_map[subdata].items()):
                if k in tmp_document_controll_map:
                    tmp_document_controll_map[k]+=training_map[subdata][k];
                else:
                    tmp_document_controll_map[k]=training_map[subdata][k];
        for k, v in sorted(tmp_document_controll_map.items()):         
            all_training_instances.append(tmp_document_controll_map[k]);
        w_dt_maps_list=tf_idf.tf_idf_nishimura_interface(all_training_instances);
        all_training_instances=[document for sub_document in all_training_instances for document in sub_document]
    #------------------------------------------------------------
    #全文書での重みスコアを足す
    for document_index,w_dt_map in enumerate(w_dt_maps_list):
        for t in w_dt_map:
            if t not in word_score_map:
                word_score_map[t]=w_dt_map[t];
            else:
                word_score_map[t]+=w_dt_map[t];
    #全文書の重みスコアにL2正則化をかけて，閾値を算出 
    if L2_flag==True:
        #L2正則化をかける        
        weight_sum=0;    
        for key in word_score_map:
            weight=word_score_map[key];
            weight_sum+=(weight)**2;
        L2_norm=math.sqrt(weight_sum);
        L2_normalized_map={};    
        L2_weightsum=0;
        for key in word_score_map:
            normalized_score=word_score_map[key]/L2_norm;
            L2_normalized_map[key]=normalized_score;        
            #L2で正規化した重みの和を求める
            L2_weightsum+=normalized_score;
        #足切りスコアの算出
        L2_average=L2_weightsum/len(L2_normalized_map);
        for doc in all_training_instances:
            for t in doc:
                if t in L2_normalized_map:
                    if L2_normalized_map[t] < L2_average:
                        #足切り
                        num_of_discarded_feat+=1;
                    else:
                        weight_format=u'{}_{}_{}'.format('normal', t, L2_normalized_map[t]);
                        if t not in feature_map_character:
                            feature_map_character[t]=[weight_format];
                        elif weight_format not in feature_map_character[t]:
                            feature_map_character[t].append(weight_format);
    #閾値で足切りしない場合
    else:
        for t in word_score_map:
            weight_format=u'{}_{}_{}'.format('normal', t, word_score_map[t]);
    
    print 'The number of discarded features :{}'.format(num_of_discarded_feat);
    #コード書き換えに伴い変数名が変わったので，一時的な措置
    tfidf_score_map=word_score_map;

    return feature_map_character, tfidf_score_map;
    
def create_tfidf_feat_idea2_4(training_map, feature_map_character, tfidf_idea, args):
    easy_domain_flag=args.easy_domain2;
    if tfidf_idea==2:
        #素性ベクトルを定数倍しない
        constant=False;
        #g_ ではじまる素性は作らない
        easy_domain_flag=False;
    elif tfidf_idea==3:
        #素性ベクトルを定数倍する
        constant=True;
        #g_ で始まる素性は作らない
        easy_domain_flag=False;
    elif tfidf_idea==4:
        #ラベルごとに素性を分ける
        constant=True;
        #g_ で始まる素性は作る
        easy_domain_flag=True;
    elif tfidf_idea==5:
        #素性ベクトルを定数倍する
        constant=True;
        #g_ で始まる素性は作らない
        easy_domain_flag=False;        
    
    feature_map_character, L2_normalized_map=create_tfidf_feat_idea_general(training_map, feature_map_character, constant, easy_domain_flag, args);
            
    return feature_map_character, L2_normalized_map;

def create_tfidf_feat_idea_general(training_map, feature_map_character, constant, easy_domain_flag, args):
    """
    ラベルごとの重要語を取り出す
    TFIDFスコアを文書集合から算出した後，ラベル文書ごとに閾値（足切り値）を求め，閾値以下の語は素性を作らない
    これで，「あるラベルに特徴的な語」を示す素性が作れた．と思う

    ラベルごとで素性選択を行う．
    疑似文書（ラベル）の作成→ラベル文書ごとにTFIDF計算→ラベルごとに閾値計算→
    閾値を設定する→素性選択
    """
    import math;
    tfidf_type=args.tfidf_type;
    L2_flag=True;
    num_of_discarded_feat=0;
    #単語スコアを保存しておくmap
    word_score_map={};
    #------------------------------------------------------------
    print 'TFIDF(Idea-2,3,4) score calculating'
    print 'L2 flag:{}'.format(L2_flag)
    #------------------------------------------------------------
    #訓練コーパスからラベルごとに疑似文書を作成する
    #TFIDF計算の入力にはlist documents [list document [ unicode token ] ]が必要
    if tfidf_type=='normal':
        print 'use tfidf(normal)'
        all_training_instances=[];
        tmp_document_controll_map={};
        for label in training_map:
            #for k, v in sorted(training_map[subdata].items()):
            label_document=[t for t in training_map[label]];
            if label in tmp_document_controll_map:
                tmp_document_controll_map[label]+=label_document;
            else:
                tmp_document_controll_map[label]=label_document
     
        for k, v in sorted(tmp_document_controll_map.items()):
            all_training_instances.append(tmp_document_controll_map[k]);
        w_dt_maps_list=tf_idf.tf_idf_interface(all_training_instances);
    #------------------------------------------------------------
    elif args.tfidf_type=='nishimura':
        #TFIDF計算の入力にはlist documents [list document [ list sub-document [unicode token] ] ]が必要
        #IDFの計算にはlist documents[unicode token]が必要だが，これは関数内で自動で変換してくれる
        print 'use tfidf(nishimura)'
        all_training_instances=[];
        tmp_document_controll_map={};
        for subdata in training_map:
            for k, v in sorted(training_map[subdata].items()):
                if k in tmp_document_controll_map:
                    tmp_document_controll_map[k]+=training_map[subdata][k];
                else:
                    tmp_document_controll_map[k]=training_map[subdata][k];
        for k, v in sorted(tmp_document_controll_map.items()):         
            all_training_instances.append(tmp_document_controll_map[k]);
        w_dt_maps_list=tf_idf.tf_idf_nishimura_interface(all_training_instances);
        all_training_instances=[document for sub_document in all_training_instances for document in sub_document]
    #------------------------------------------------------------
    #全文書での重みスコアを足す
    for document_index,w_dt_map in enumerate(w_dt_maps_list):
        for t in w_dt_map:
            if t not in word_score_map:
                word_score_map[t]=w_dt_map[t];
            else:
                word_score_map[t]+=w_dt_map[t];
    #アルファベットのリスト
    alphabet_list=[chr(i) for i in range(65,65+26)];
    alphabet_list.remove('I');
    alphabet_list.remove('O');
    alphabet_list.remove('Y');
    #全文書の重みスコアにL2正則化をかけて，閾値を算出 
    if L2_flag==True:
        #ラベルごとの閾値を保存しておくmap
        #map threshold_point_map {unicode label:float threshold_point}
        threshold_point_map={};
        #L2正則化をかける        
        weight_sum=0;    
        for key in word_score_map:
            weight=word_score_map[key];
            weight_sum+=(weight)**2;
        L2_norm=math.sqrt(weight_sum);
        #L2正則化された単語重みを保存しておくmap
        #map L2_normalized_map {map L2_normalized_map_per_label unicode label:{unicode token:float weight}}
        L2_normalized_map={};
        
        for label_index,w_dt_map in enumerate(w_dt_maps_list):
            alphabet_label=alphabet_list[label_index];
            L2_normalized_map_per_label={};
            threshold_point_in_label=0;
            for token in w_dt_map:
                L2_normalized_weight=w_dt_map[token]/L2_norm;
                if constant==True:
                    L2_normalized_map_per_label[token]=L2_normalized_weight;        
                else:
                    L2_normalized_map_per_label[token]=1;        
                #L2で正規化したラベル内の重みの和を求める
                threshold_point_in_label+=L2_normalized_weight;
            #ラベル内の閾値
            threshold_point_map[alphabet_label]=threshold_point_in_label/len(w_dt_map);
            if constant==True:
                #ラベル内の単語重み(L2正則化済み)の保存
                L2_normalized_map[alphabet_label]=L2_normalized_map_per_label;
            else:
                L2_normalized_map[alphabet_label]=L2_normalized_map_per_label;

        for alphabet_label in L2_normalized_map:
            threshold=threshold_point_map[alphabet_label];
            for t in L2_normalized_map[alphabet_label]:
                #足切り
                if L2_normalized_map[alphabet_label][t] < threshold:
                    num_of_discarded_feat+=1;
                else:
                    if constant==True:
                        weight_format=u'{}_{}_{}'.format(alphabet_label, t, L2_normalized_map[alphabet_label][t]);
                    else:
                        #素性ベクトルを定数倍しないので，１をたてておく.つまりBinary素性
                        weight_format=u'{}_{}_{}'.format('normal', t, 1);
                    if t not in feature_map_character:
                        feature_map_character[t]=[weight_format];
                    elif weight_format not in feature_map_character[t]:
                        feature_map_character[t].append(weight_format);
    #閾値で足切りしない場合
    else:
        for t in word_score_map:
            weight_format=u'{}_{}_{}'.format('normal', t, word_score_map[t]);

    print 'The number of discarded features:{}'.format(num_of_discarded_feat);

    return feature_map_character, L2_normalized_map;
