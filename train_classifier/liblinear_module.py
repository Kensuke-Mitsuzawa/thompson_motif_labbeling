#! /usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Thu Dec 12 12:16:52 2013

@author: kensuke-mi
__date__="2014/02/25"
"""
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import sys, codecs, random, re, subprocess;
import feature_function;
env='pine';

if env=='pine':
    #change below by an environment
    libsvm_wrapper_path='/home/kensuke-mi/opt/libsvm-3.17/python/';
elif env=='local':
    libsvm_wrapper_path='/Users/kensuke-mi/opt/libsvm-3.17/python/';
    liblinear_wrapper_path='/Users/kensuke-mi/opt/liblinear-1.94/python/';
    sys.path.append(liblinear_wrapper_path);
sys.path.append(libsvm_wrapper_path);
from liblinearutil import *;
from svmutil import *;
import scale_grid;

#option parameter
put_weight_constraint=True;
under_sampling=False;
sclale_flag=False;
#The path to save libsvm format training files
prefix_path_to_training_f='../classifier/libsvm_format/';
suffix_path_to_tarining_f='.traindata.'
#実際には保存先パスは下のようにしたい
#'../classifier/libsvm_format/'+correct_label_key+'.traindata.'+exno

#The path to save semi-supervised(updated by dutch folktale instances) libsvm format training files 
prefix_path_to_updated_f='../classifier/libsvm_format_updated/';
suffix_path_to_updated_f='.updated.';
#The path to save arow trained model

#The path to save liblinear trained model

#The path to save liblinear logistic model
suffix_name_log='.logistic.';
#model 1st
tmp_modelpath_log='../classifier/logistic_1st/';
#model 2nd
model_path_log='../classifier/logistic_2nd/';
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def unify_tarining_feature_space(training_map_feature_space):
    unified_map={};
    for subdata_key in training_map_feature_space:
        for label in training_map_feature_space[subdata_key]:
            if label not in unified_map:
                unified_map[label]=training_map_feature_space[subdata_key][label];
            else:
                unified_map[label]+=training_map_feature_space[subdata_key][label];
    return unified_map;

def split_for_train_test(correct_instances_stack, incorrect_instances_stack, instace_lines_num_map, ratio_of_training_instance):
    #ここでtrainとtestに分けられるはず
    all_instances=len(correct_instances_stack)+len(incorrect_instances_stack);
    random.shuffle(correct_instances_stack);
    random.shuffle(incorrect_instances_stack);
    ratio_of_correct=float(instace_lines_num_map['C'])/(instace_lines_num_map['C']+instace_lines_num_map['N']);
    ratio_of_incorrect=float(instace_lines_num_map['N'])/(instace_lines_num_map['C']+instace_lines_num_map['N']);
    #訓練用のインスタンス数
    #headerの変数で量を調整可能
    num_instance_for_train=int(ratio_of_training_instance*all_instances);
    #テスト用のインスタンス数
    num_instance_for_test=all_instances-num_instance_for_train;
    #正例と負例の事例スタックから何行ずつとって来ればいいのか？を計算
    num_of_instances_of_correct_for_test=int(num_instance_for_test*ratio_of_correct);
    num_of_instances_of_incorrect_for_test=int(num_instance_for_test*ratio_of_incorrect);
    #スライス機能を使って，テスト用のインスタンスを獲得
    instances_for_test=correct_instances_stack[:num_of_instances_of_correct_for_test]\
            +incorrect_instances_stack[:num_of_instances_of_incorrect_for_test];
    #スライス機能を使って，訓練用のインスタンスを獲得
    instances_for_train=correct_instances_stack[num_of_instances_of_correct_for_test:]\
            +incorrect_instances_stack[num_of_instances_of_incorrect_for_test:];
    return instances_for_train, instances_for_test;

def get_training_ratio(additional_instance_tuple):
    """
    This code gets the ratio between true instances and flase instances. After getting ratio, this code returns weight parameter for liblinear.
    RETURN: unicode weight_parm
    """
    true_instance=additional_instance_tuple[1];
    false_instane=additional_instance_tuple[2];
    num_true=len(true_instance);
    num_false=len(false_instane);
    true_ratio=float(num_true)/(num_true+num_false);
    false_ratio=float(num_false)/(num_true+num_false);
    
    weight_parm='-w-1 {} -w1 {} -s 0'.format(true_ratio*100, false_ratio*100);
    print weight_parm;

    return weight_parm;

def tuning_training_liblinear(correct_label_key,weight_parm,exno):
    
    return_value=scale_grid.main('../classifier/libsvm_format/'+correct_label_key+'.traindata.'+exno,
                                 '../classifier/libsvm_format/'+correct_label_key+'.traindata.'+exno,
                                 sclale_flag);
    
    weight_parm+=u' -c {} -p {}'.format(return_value[0], return_value[1]);
    train_y, train_x=svm_read_problem('../classifier/libsvm_format/'+correct_label_key+'.traindata.'+exno); 
    print weight_parm;
    model=train(train_y, train_x, str(weight_parm));
    save_model('../classifier/liblinear/'+correct_label_key+'.liblin.model.'+exno, model);
    #------------------------------------------------------------ 
    try:
        os.remove('{}.traindata.{}.out'.format(correct_label_key, exno));
    except OSError:
        print '{}.traindata.{}.out does not exist'.format(correct_label_key, exno)
    #------------------------------------------------------------ 
    try:
        os.remove('{}.traindata.{}.scale'.format(correct_label_key, exno));
    except OSError:
        print '{}.traindata.{}.scale does not exist'.format(correct_label_key, exno)
    #------------------------------------------------------------ 
    try:
        os.remove('{}.traindata.{}.range'.format(correct_label_key, exno));
    except OSError:
        print '{}.traindata.{}.range does not exist'.format(correct_label_key, exno)
    #------------------------------------------------------------ 
    try:
        os.remove('{}.traindata.{}.scale.out'.format(correct_label_key, exno));
    except OSError:
        print '{}.traindata.{}.scale.out does not exist'.format(correct_label_key, exno)
    #------------------------------------------------------------ 
    try:
        os.remove('{}.devdata.{}.scale'.format(correct_label_key, exno));
    except OSError:
        print '{}.devdata.{}.scale does not exist'.format(correct_label_key, exno)
  

    print u'-'*30;

def training_logistic_model(best_acc,weight_parm,correct_label_key,tmp_modelpath_log,training_file,exno):
    """
    汎化した関数 TMIだけでのモデルとTMI+DFDの２つのモデル構築を担当
    This function trains liblinear logistic regression model.
    RETURN:None
    """
    train_y, train_x=svm_read_problem('../classifier/libsvm_format/'+correct_label_key+'.traindata.'+exno); 
    model=train(train_y, train_x, str(weight_parm));
    savepath=tmp_modelpath_log+correct_label_key+suffix_name_log+exno;
    save_model(savepath,model);
    print 'Logistic classifier for {} is constructed in {}'.format(correct_label_key,
                                                                   savepath)

    print u'-'*30;
                
def decode_and_add_logistic(additional_instances_stack,instances_for_train,instances_for_test,correct_label_key,exno,args):
    """
    Try decode the new instance(sentences from document). If the probability of classification is above the threshold, add as new-instance.
    RETURN: list additional_instances_stack [ tuple (unicode thompson_label, list True_instance_libsvmformat [unicode instance], list False_instance_libsvm_fomrat [unicode instance])]
    """
    threshold=float(args.arow_thres);

    dutch_testfile_path='../classifier/libsvm_format_dutch_semi/'+correct_label_key+'.traindata.'+exno;
    #livsvm用のフォーマットだけ先に作成してしまう
    with codecs.open(dutch_testfile_path,'w','utf-8') as f:
        f.writelines(instances_for_train);
    with codecs.open('../classifier/libsvm_format_dutch_semi/'+correct_label_key+'.devdata.'+exno,'w','utf-8') as f:
        f.writelines(instances_for_test);
    
    model_name=tmp_modelpath_log+correct_label_key+suffix_name_log+exno;
    model=load_model(model_name);
    test_y,test_x=svm_read_problem(dutch_testfile_path);
    p_label,p_acc,p_val=predict(test_y,test_x,model,'-b 1');

    true_instances_libsvm_format=[];
    false_instances_libsvm_format=[];
    for classification_i,classification in enumerate(p_label):
        #If classifier returns True, p_label is 1.0
        if classification==1.0:
            #p_val have probability of each classification.
            #list p_val [ list probability [probability for True, probability for False] ]
            if p_val[classification_i][0]>threshold:
                true_instances_libsvm_format.append(instances_for_train[classification_i]);
            else:
                false_instances_libsvm_format.append(instances_for_train[classification_i]);
        else:
            false_instances_libsvm_format.append(instances_for_train[classification_i]);
    
    additional_instances_stack.append((correct_label_key,true_instances_libsvm_format,false_instances_libsvm_format))

    return additional_instances_stack;

def tuning_arow(correct_label_key,exno,training_file,dev_file,mode):
    """
    arowのモデルtrainingをするために，ハイパーパラメータのgrid探索を行う
    modeが'first'と'second'の分岐を行う．
    thompsonだけの訓練の時にはfirst，オランダ語も加えた時にはsecondを使う
    RETURN: tuple (float best_accuracy, float hyper_parameter )
    """

    print training_file;
    print dev_file;
    #最適なハイパーパラメータを探索する
    hyp_start=1;
    hyp_end=10;
    hyp_list=[0.01*x for x in range(hyp_start,hyp_end)];

    iter_times=1;

    best_acc=None;
    result_acc=0.0;
    subproc_args={'stdin':subprocess.PIPE,
                    'stdout':subprocess.PIPE,
                    'stderr':subprocess.STDOUT,
                    'close_fds':True,};
                    
    for hyp_tmp in hyp_list:
        processed_line=u'';
        #arow_learn-i num_iter -r hyp -s train_file model_file
        if mode=='first':
            tmp_modelpath='../classifier/arow/'+correct_label_key+'.arowmodel1st.'+exno;
        elif mode=='second':
            tmp_modelpath='../classifier/arow/'+correct_label_key+'.arowmodel2nd.'+exno;            
        #print '-'*30;
        #print '{} hyp:{}'.format(tmp_modelpath, hyp_tmp);
        arow_args=['arow_learn', '-i', str(iter_times), '-r', str(hyp_tmp), '-s', training_file, tmp_modelpath];

        try:
            #p_train=subprocess.Popen(arow_args, **subproc_args);
            p_train=subprocess.Popen(arow_args, **subproc_args);
        except OSError:
            print "Failed to execute command: %s" % args[0];
            sys.exit(1);

        try:
            arow_test_args=['arow_test', dev_file, tmp_modelpath];
            p_test=subprocess.Popen(arow_test_args, **subproc_args);

            for line in p_test.stdout:
                #時々，Accuracyが文頭にならない変な現象が発生するため
                #Accuracyの直前に何らかの確信度が表示されるバグがある．なので，Accuracyの前にメタ文字のgreedy searchを入れてこの現象を回避
                if re.search(ur'.*Accuracy\s(.+)%\s\(.+\)', line):
                    processed_line=re.sub(ur'.*Accuracy\s(.+)%\s\(.+\)', ur'\1', line);

        except AttributeError: 
            print '[Warning] occured Attribute error'            
            continue;
            
        if not processed_line==u'' and float(result_acc)<float(processed_line):
            result_acc=float(processed_line);
            print 'best score {} at {}'.format(result_acc, hyp_tmp);
            best_acc=(result_acc, hyp_tmp);
    #--------------------------------------------------------------
    print '{} hyp:{}'.format(tmp_modelpath, hyp_tmp);
    print 'Best result is {}'.format(best_acc);
    print '-'*30;

    return best_acc;

def training_arowmodel(best_acc,correct_label_key,tmp_modelpath,training_file,exno):
    iter_times=1;
    hyp=best_acc[1];        
    #parameter:arow_learn-i num_iter -r hyp -s train_file model_file
    
    arow_args=['arow_learn', '-i', str(iter_times), '-r', str(hyp), '-s', training_file, tmp_modelpath];    
    subproc_args={'stdin':subprocess.PIPE,
                    'stdout':subprocess.PIPE,
                    'stderr':subprocess.STDOUT,
                    'close_fds':True,};

    try:
        p_train=subprocess.Popen(arow_args, **subproc_args);
    except OSError:
        print "Failed to execute command: %s" % args[0];
        sys.exit(1);

def decode_and_add_arow(additional_instances_stack,instances_for_train,instances_for_test,correct_label_key,exno,args):
    """
    This function decodes DFD instances with trained model(with TMI).
    If We get probability above threshold, this function adds such instances into additional_instances_stack.
    RETURN:list additional_instances_stack [tuple additional_instance_tuple (unicode thompson_label, list [unicode True_instance_libsvmformat, unicode False_instance_libsvm_fomrat])]
    """
    dutch_testfile_path='../classifier/libsvm_format_dutch_semi/'+correct_label_key+'.traindata.'+exno;
    #livsvm用のフォーマットだけ先に作成してしまう
    with codecs.open(dutch_testfile_path,'w','utf-8') as f:
        f.writelines(instances_for_train);
    with codecs.open('../classifier/libsvm_format_dutch_semi/'+correct_label_key+'.devdata.'+exno, 'w', 'utf-8') as f:
        f.writelines(instances_for_test);
    
    #分類時の確信度を保存しておくスタック
    prob_stack=[];
    #訓練済みのモデルで分類を行う           
    #分類時の細かい設定をしておく
    model_name='../classifier/arow/'+correct_label_key+'.arowmodel1st.'+exno;
    subproc_args={'stdin':subprocess.PIPE,
                    'stdout':subprocess.PIPE,
                    'stderr':subprocess.STDOUT,
                    'close_fds':True,};
    try:
        #Usage: arow_test test_file model_file
        arow_args=['arow_test', dutch_testfile_path, model_name];
        p_test=subprocess.Popen(arow_args, shell=False, **subproc_args);
        
        (stdouterr, stdin) = (p_test.stdout, p_test.stdin)
        #先に全分類だけをしてしまう．確信度だけを求めてから，高い確信度の行数を得る
        while True:
            line=stdouterr.readline()
            if not line:
                break
            
            line=line.rstrip();
            if re.search(ur'ERROR', line):
                sys.exit('Some problems happened in semi-supervised Arow model decoding');
            else: 
                prob_stack.append(line);
        #必要のない~4番目までの情報を削除する
        del prob_stack[:4];

        #確信度が高い行数を調べる
        additional_instances_tuple=get_instance_above_threshold(prob_stack,correct_label_key,dutch_testfile_path,args);
        #ラベルごとに正例と負例が，additional_instances_stackに集められる
        #ラベルはaddtional_instances_tupleの[0]インデックスに格納されている                
        additional_instances_stack.append(additional_instances_tuple);

    except OSError:
        print "Failed to execute command: %s" % args[0];
        sys.exit(1);

    return additional_instances_stack;

def get_instance_above_threshold(prob_stack,correct_label_key,dutch_testfile_path,args):
    """
    prob_stackを分析して，「予測されたラベルが正しい」かつ，「確信度が閾値以上」で「正例」の行数を正例として訓練事例に加える
    またprob_stackを分析して，「予測されたラベルが間違い」かつ，「確信度が閾値以上」であれば「負例」として訓練事例に加える
    ARGS: 省略
    RETURN additional_insrances_tuple: tuple (unicode label, list [unicode correct_instance_in_libsvmformat], list [unicode incorrect_instance_in_libsvmformat])
    """
    threshold=args.arow_thres;

    line_number_stack_for_correctlabel=[];
    line_number_stack_for_incorrectlabel=[];
    for line_index, line in enumerate(prob_stack):
        if re.search(ur'LOG\(ERROR\)\:', line):
            sys.exit('Some problems happened in Arow model decoding');
       
        elif re.search(ur'(\+1|-1)\s0', line) or re.search(ur'(\+1|-1)\s(\+|-)[0-9]+.+', line):
            print line;
            if re.search(ur'Accuracy .+% \(.+\)', line):
                line=re.sub(ur'Accuracy .+% \(.+\)', ur'', line);
            decision, prob=line.split();
            if float(prob) >= float(threshold):
                if decision==u'+1':
                    line_number_stack_for_correctlabel.append(line_index);
            elif decision==u'-1' and not prob=='0':
                line_number_stack_for_incorrectlabel.append(line_index);

    # ファイルを読み込んで，行数を照らし合わせ，livsvmformatの事例を得る．
    correct_instance_stack=[];
    incorrect_instance_stack=[];
    lines=codecs.open(dutch_testfile_path, 'r', 'utf-8').readlines();

    #仮に閾値を超えた事例が一つもなければ，そのまま空にしておく
    if not line_number_stack_for_correctlabel==[]:            
        correct_instance_stack=[lines[line_number]  for line_number in line_number_stack_for_correctlabel];
    if not line_number_stack_for_incorrectlabel==[]:
        incorrect_instance_stack=[lines[line_number] for line_number in line_number_stack_for_incorrectlabel];
    
    additional_instances_tuple=(correct_label_key, correct_instance_stack, incorrect_instance_stack);
    return additional_instances_tuple;

def add_additional_instances(additional_instance_stack,args):
    """
    追加すべき（分類時の閾値が設定値以上の）事例のみをthompson treeからのinstanceに足す
    新しい事例が追加済みのファイルは'../classifier/libsvm_format_updated/'+label+'.updated.'+args.experiment_no　に保存される
    ARGS: 省略
    RETURN: None
    """
    for additional_instance_tuple in additional_instance_stack:
        correct_label=additional_instance_tuple[0];
        #correct_instance_stackの中にはlivsvm formatの事例がいくつも格納されている
        correct_instance_stack=additional_instance_tuple[1];
        #incorrect_instance_stackの中にはlivsvm formatの事例がいくつも格納されている        
        incorrect_instance_stack=additional_instance_tuple[2];
        #正例にも負例にも更新事例がなければ，モデルは新たに訓練しないで，訓練済みのモデルをコピーしてしまって良いだろう
        if correct_instance_stack==[] and incorrect_instance_stack==[]:
            print 'There is no update instance for correct label {}. Skip this label'.format(correct_label);
        else:
            print '[semi-supervised instances] label:{}'.format(correct_label); 
            print 'added instances for correct label:{}'.format(len(correct_instance_stack))
            print 'added instances for incorrect label:{}'.format(len(incorrect_instance_stack))
        
        #トンプソンのモチーフラベルのみが保存されているパス
        already_path_to_training_f=prefix_path_to_training_f+correct_label+suffix_path_to_tarining_f+args.experiment_no;
        already_training_lines=codecs.open(already_path_to_training_f, 'r', 'utf-8').readlines();
        #オランダ語コーパスの閾値以上の確信度を持つ事例を追加する
        already_training_lines+=correct_instance_stack;
        already_training_lines+=incorrect_instance_stack;
        
        #更新済みのtraining fileを保存するパス
        updated_path_to_training_f=prefix_path_to_updated_f+correct_label+suffix_path_to_tarining_f+args.experiment_no;
        with codecs.open(updated_path_to_training_f, 'w', 'utf-8') as f:
            f.writelines(already_training_lines);

        if args.training=='arow':
            #devsetの上で評価して最適なハイパーパラメータを返す
            best_acc=tuning_arow(correct_label,args.experiment_no,updated_path_to_training_f,updated_path_to_training_f,'second');
            
            tmp_modelpath='../classifier/arow/'+correct_label+'arowmodel2nd.'+args.experiment_no;    
            training_file=prefix_path_to_updated_f+correct_label+'.traindata.'+args.experiment_no;
            #最適なパラメータでモデルを作成
            #この関数で，オランダ語コーパスを用いたsemi-supervisedなモデルの訓練はおしまい
            #モデルは ../classifier/arow/'+correct_label+'arowmodel2nd.'+args.experiment_no に保存される
            training_arowmodel(best_acc,correct_label,tmp_modelpath,training_file,args.experiment_no);

        elif args.training=='logistic':
            #TODO 必要なら，チューニング用のコードを書いておくこと．2/11の時は，「DFDドメインに適合しすぎるとまずいんじゃない？」という考えから，チューニングはしない
            best_acc=0;
            correct_label_key=additional_instance_tuple[0];

            #分離平面に重みを置くため，trainingの正例と負例の比を調べる
            weight_parm=get_training_ratio(additional_instance_tuple) 
            #追加したデータを利用して，２回めの分類器を作成
            training_logistic_model(best_acc,weight_parm,correct_label_key,model_path_log,updated_path_to_training_f,args.experiment_no);

def shape_format(training_map,mode,args):
    """
    make format which is readable for liblinear,libsvm
    libsvm用の入力フォーマットに整える
    *RETURN exists only when mode=='semi'
    （modeが'semi'の時のみ意味がある戻り値）
    RETURN additional_instances_stack: list [ tuple (unicode label, list [unicode libsvm format(instance of correct label) ] , list [unicode libsvm format(instance of incorrect label)] ) ]
    """
    #------------------------------------------------------------ 
    regularization=args.n_t;
    exno=args.experiment_no;
    additional_instances_stack=[];
    #------------------------------------------------------------ 
    for correct_label_key in training_map:
        if re.search(ur'NOT_',correct_label_key):
            continue;
        else:
            instance_lines_num_map={'C':0, 'N':0};
            lines_for_correct_instances_stack=[];
            lines_for_incorrect_instances_stack=[];
            instances_in_correct_label=training_map[correct_label_key];
            #------------------------------------------------------------  
            #正例の処理をする
            for one_instance in instances_in_correct_label:
                instance_lines_num_map['C']+=1;
                one_instance_stack=one_instance;
                one_instance_stack=list(set(one_instance_stack));
                one_instance_stack.sort();
                one_instance=[str(tuple_item[0])+u':'+str(tuple_item[1]) for tuple_item in one_instance_stack];
                lines_for_correct_instances_stack.append(u'{} {}\n'.format('+1', u' '.join(one_instance)));
            #負例の処理を行う．重みかアンダーサンプリングかのオプションを設定している
            #------------------------------------------------------------  
            #分離平面に重みを置く場合
            if put_weight_constraint==True and under_sampling==False:
                instances_in_incorrect_label=training_map['NOT_'+correct_label_key];
                for one_instance in instances_in_incorrect_label:
                    #仮にこの変数名にしておく
                    one_instance_stack=one_instance; 
                    instance_lines_num_map['N']+=1;
                    one_instance_stack=list(set(one_instance_stack));
                    one_instance_stack.sort();
                    one_instance_stack=[str(tuple_item[0])+u':'+str(tuple_item[1]) for tuple_item in one_instance_stack];
                    lines_for_incorrect_instances_stack.append(u'{} {}\n'.format('-1', u' '.join(one_instance_stack)));
                ratio_c=float(instance_lines_num_map['C']) / (instance_lines_num_map['C']+instance_lines_num_map['N']);
                ratio_n=float(instance_lines_num_map['N']) / (instance_lines_num_map['C']+instance_lines_num_map['N']);
                if int(ratio_c*100)==0:
                    weight_parm='-w-1 {} -w1 {} -s {} -q'.format(1, int(ratio_n*100), regularization);
                else:
                    weight_parm='-w-1 {} -w1 {} -s {} -q'.format(int(ratio_c*100), int(ratio_n*100), regularization);
            #------------------------------------------------------------  
            #アンダーサンプリングする場合
            elif put_weight_constraint==False and under_sampling==True:
                #どうせ使わないなら消してもいいんじゃない？
                #各ラベルのインスタンス比率を求める
                num_of_incorrect_training_instance=0;
                instance_ratio_map={};
                #負例の数を計算
                for label in training_map:
                    if label!=correct_label_key:
                        num_of_incorrect_training_instance+=len(training_map[label]);
                        instance_lines_num_map['N']+=len(training_map[label]);
                #負例のうちの特定のラベルが何行分出力すれば良いのか？を計算する
                for label in training_map:
                    if label!=correct_label_key:
                        instance_ratio_map[label]=\
                                int((float(len(training_map[label]))/num_of_incorrect_training_instance)*instance_lines_num_map['C']);
                for label in training_map:
                    if label!=correct_label_key:
                        for instance_index, one_instance in enumerate(training_map[label]):
                            #あとで変数名を変えておくこと，これだと意味が違う
                            one_instance_stack=one_instance;
                            one_instance_stack=list(set(one_instance_stack));
                            one_instance_stack.sort();
                            one_instance_stack=[str(tuple_item[0])+u':'+str(tuple_item[1]) for tuple_item in one_instance_stack];
                            lines_for_incorrect_instances_stack.append(u'{} {}\n'.format('-1', u' '.join(one_instance_stack)));
                        #比率にもとづいて計算された行数を追加し終わったら，次のラベルに移る 
                        if instance_index==instance_ratio_map[label]: continue;
                weight_parm='-s {} -q'.format(regularization);
            #------------------------------------------------------------  
            #両方ともTrueになっているときはエラーをはいて終わる
            elif put_weight_constraint==True and under_sampling==True:
                sys.exit('[Warning] Both put_weight_constraint and under_sampling is True');
            #------------------------------------------------------------  
            #トレーニング量の調整をしない場合
            elif put_weight_constraint==False and under_sampling==False:
                instances_in_incorrect_label=training_map['NOT_'+correct_label_key];
                for one_instance in instances_in_incorrect_label:
                    instance_lines_num_map['N']+=1;
                    one_instance_stack=list(set(one_instance));
                    one_instance_stack.sort();
                    one_instance_stack=[str(tuple_item[0])+u':'+str(tuple_item[1]) for tuple_item in one_instance_stack];
                    lines_for_incorrect_instances_stack.append(u'{} {}\n'.format('-1', u' '.join(one_instance_stack)));
                weight_parm='-s {}'.format(regularization);
            #------------------------------------------------------------  
            #ファイルに書き出しの処理をおこなう
            #インドメインでのtrainとtestに分離
            training_amount=float(args.training_amount);
            instances_for_train, instances_for_test=split_for_train_test(lines_for_correct_instances_stack,
                                                                         lines_for_incorrect_instances_stack,
                                                                         instance_lines_num_map,
                                                                         training_amount);
                                                                         
            #汚いやり方だが，これで訓練とテストを再統合できる
            #そもそもどうしてこんなことしてるかというと，close_testとかいう勘違い関数があったから
            #長い目でみると，書き換えが必要
            instances_line_for_train=instances_for_train+instances_for_test; 

            if mode=='super':
                #writeout training data to liblinear format file
                with codecs.open(prefix_path_to_training_f+correct_label_key+suffix_path_to_tarining_f+exno,
                                 'w','utf-8') as f:
                    f.writelines(instances_line_for_train); 
               
                #train model
                if args.training=='liblinear':
                    tuning_training_liblinear(correct_label_key,weight_parm,exno);
                elif args.training=='arow':
                    training_file=prefix_path_to_training_f+correct_label_key+suffix_path_to_tarining_f+exno;
                    dev_file=prefix_path_to_training_f+correct_label_key+'.devdata.'+exno;
                    #devsetの上で評価して最適なハイパーパラメータを返す
                    #tuning model on dev. set
                    best_acc=tuning_arow(correct_label_key,exno,training_file,dev_file,'first');
                    
                    tmp_modelpath='../classifier/arow/'+correct_label_key+'arowmodel1st.'+exno;    
                    #training_file='../classifier/libsvm_format/'+correct_label_key+'.traindata.'+exno;
                    #最適なパラメータでモデルを作成
                    training_arowmodel(best_acc,correct_label_key,tmp_modelpath,training_file,exno);
                
                elif args.training=='logistic':
                    training_file=prefix_path_to_training_f+correct_label_key+suffix_path_to_tarining_f+exno;
                    #TODO devの上でのtuningって必要？ 
                    
                    #re-write -s parameter to -s 0 (regression)
                    weight_parm=re.sub(ur'-s\s\d',u'-s 0',weight_parm);
                    print '-s parameter is re-written to -s 0';
                    #Liblinear logisticのモデルを構築
                    best_acc=0;
                    training_logistic_model(best_acc,weight_parm,correct_label_key,tmp_modelpath_log,training_file,exno);
            else:
                if args.training=='arow':
                    additional_instances_stack=decode_and_add_arow(additional_instances_stack,instances_for_train,
                                                                   instances_for_test,correct_label_key,exno,args);
               
                elif args.training==u'logistic':
                    additional_instances_stack=decode_and_add_logistic(additional_instances_stack,instances_for_train,
                                                                       instances_for_test,correct_label_key,exno,args);
        if mode=='semi':             
            #additional_instances_stackが返されるのはmodeが'semi'の場合のみ
            return additional_instances_stack;        
                
def conv_to_featurespace_for_dutch_in_arowmode(dutch_training_tree,feature_map_character,feature_map_numeric):
    """
    オランダ語コーパスの一文ごとを素性空間に変換して，返す関数
    RETURN: map { unicode key : list [ list [ tuple ( int feature number, float featurevalue ) ] ] }
    """    
    #ここから変換開始
    dutch_trainingmap_featurespace={};
    for label in dutch_training_tree:
        #one_insは一文ごとが格納されている
        for one_ins in dutch_training_tree[label]:
            one_ins_featurespace=[];
            for t in one_ins:
                if t in feature_map_character:                
                    for feature_candidate in feature_map_character[t]:
                        if len(feature_candidate.split(u'_'))==2:
                            domainlabel=feature_candidate.split(u'_')[0];
                            token=u'_';
                            featurevalue=feature_candidate.split(u'_')[1];
                        elif len(feature_candidate.split(u'_'))==3: 
                            domainlabel, token, featurevalue=feature_candidate.split(u'_');
                        
                        feature_number=feature_map_numeric[feature_candidate];
                        if re.search(ur'[0-9]+', featurevalue):
                            one_ins_featurespace.append( (feature_number, float(featurevalue) ));
                        elif re.search(ur'unigram', featurevalue):
                            one_ins_featurespace.append( (feature_number, 1 ));
            if not one_ins_featurespace==[]:
                if label not in dutch_trainingmap_featurespace:
                    dutch_trainingmap_featurespace[label]=[one_ins_featurespace];
                else:
                    dutch_trainingmap_featurespace[label].append(one_ins_featurespace);
    return dutch_trainingmap_featurespace;

def out_to_libsvm_format_logistic(doc_based_map,sentence_labeled_map,feature_map_numeric,
                        feature_map_character, tfidf, tfidf_score_map,
                        exno, tfidf_idea, args):

    thompson_training_tree_featurespace=feature_function.convert_to_feature_space_arow(sentence_labeled_map,
                                                        feature_map_character,
                                                        feature_map_numeric,
                                                        tfidf_score_map, tfidf, tfidf_idea, args);
    #First step, train model only from thompson tree
    #トンプソン木からのデータだけを先に訓練してしまう
    shape_format(thompson_training_tree_featurespace,'super',args);

    #文書単位のデータ（一文ごと）をlibsvm_formatに変換する       
    doc_based_trainingmap_featurespace=conv_to_featurespace_for_dutch_in_arowmode(doc_based_map,feature_map_character,feature_map_numeric);
    #thompson木で訓練したモデルで判断して，閾値以上の確信度が得られた事例だけ得る
    additional_instance_stack=shape_format(doc_based_trainingmap_featurespace,'semi',args);
    #TODO この下の関数に似た，logistic用の関数を作り出す
    #この下の関数内に新たにlogistic用を書き加えば良い
    add_additional_instances(additional_instance_stack,args);
 
def out_to_libsvm_format_arow(doc_based_map,sentence_based_map,feature_map_numeric,
                        feature_map_character, tfidf, tfidf_score_map,
                        exno, tfidf_idea, args):
    #TODO libsvm用のフォーマットを出力するディレクトリをheadで指定できるようにする
    #今のままだと各関数内の空間内でバラバラに定義されていて良くない                            
    thompson_training_tree_featurespace=feature_function.convert_to_feature_space_arow(sentence_based_map,
                                                        feature_map_character,
                                                        feature_map_numeric,
                                                        tfidf_score_map, tfidf, tfidf_idea, args);
    #First step, train model only from thompson tree
    #トンプソン木からのデータだけを先に訓練してしまう
    shape_format(thompson_training_tree_featurespace,'super',args);

    #文書単位のデータ（一文ごと）をlibsvm_formatに変換する       
    doc_based_trainingmap_featurespace=conv_to_featurespace_for_dutch_in_arowmode(doc_based_map,feature_map_character,feature_map_numeric);
    #thompson木で訓練したモデルで判断して，閾値以上の確信度が得られた事例だけ得る
    additional_instance_stack=shape_format(doc_based_trainingmap_featurespace,'semi',args);
    #thompsonから生成した訓練データ libsvm_formatと↑を足し合わせて，ファイルに書き込む
    add_additional_instances(additional_instance_stack,args);
    
def out_to_libsvm_format(training_map_original, feature_map_numeric,
                        feature_map_character, tfidf, tfidf_score_map,
                        exno, tfidf_idea, args):
    
    training_map_feature_space=feature_function.convert_to_feature_space\
                                                        (training_map_original,
                                                        feature_map_character,
                                                        feature_map_numeric,
                                                        tfidf_score_map, tfidf, tfidf_idea, args);


    training_map=training_map_feature_space;
    #============================================================ 
    shape_format(training_map, 'super', args);
