# 2014/2/10
#システムを動かすためのデモコード

#liblinearで訓練する場合，かつ，素性がbag-of-wordsの時
#If you use liblinear as 1-V.S.-Rest classifier. And feature is bag-of-words
#-exno: specify model number
#-dutch: use dutch folktale database as training, -thompson: use thompson index as training
#-normalization_term: normalizarion term of classifier. 2 is L2 norm. 5 is L1 normalizarion
python construct_bigdoc_or_classifier.py -mode class -exno 00 -stop -thompson -dutch -normalization_term 2 

#liblinearで訓練し，かつ，tfidfで素性選択を行う場合
#If you do feature selection using tfidf value
python construct_bigdoc_or_classifier.py -mode class -exno 00 -stop -thompson -dutch -normalization_term 2 -tfidf

#arowを使ったsemi-supervisedな分類を行う場合
#If you use arow as classifier. In this case, classification is not document based, but sentence based classification
#-ins_range: you must specify if you use sentence based classification
#-arow_thres: This system add sentence as new instance if the probability of classification is above this threshold
python construct_bigdoc_or_classifier.py -mode class -exno 00 -stop -thompson -dutch -ins_range sentence -arow_thres 0.1
