# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 06:29:34 2014

@author: kensuke-mi
@date : 2014/03/02
"""
import sys;

env='pine'; 
if env=='pine':
    #change below by an environment
    libsvm_wrapper_path='/home/kensuke-mi/opt/libsvm-3.17/python/';
elif env=='local':
    libsvm_wrapper_path='/Users/kensuke-mi/opt/libsvm-3.17/python/';
    liblinear_wrapper_path='/Users/kensuke-mi/opt/liblinear-1.94/python/';
    sys.path.append(liblinear_wrapper_path);
sys.path.append(libsvm_wrapper_path);
import liblinearutil;
import svmutil;


arow_model_dir_path='../classifier/arow/';
logistic_model_dir_path='../classifier/logistic_2nd/'
