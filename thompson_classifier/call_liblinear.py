import codecs, sys, os;
sys.path.append('/Users/kensuke-mi/opt/liblinear-1.94/python/');
from liblinearutil import *;
import file_decoder;

def eval_with_liblinear(exno):
    test_y, test_x=svm_read_problem('./test.data');
    result_map={};
    libsvm_format_dir='../classifier/liblinear/';
    for libsvm_format in file_decoder.load_files(libsvm_format_dir, suffix=u'.model.'+exno):
        filename=os.path.basename(libsvm_format)[0];
        model=load_model(libsvm_format);
        p_label, p_acc, p_val=predict(test_y, test_x, model, '-q');
        print 'classification detail for classifier {}'.format(filename);
        print p_label, p_acc, p_val;
        if p_label==[-1.0]:
            result_map[filename]=0;
        else:
            result_map[filename]=1;
    return result_map;
