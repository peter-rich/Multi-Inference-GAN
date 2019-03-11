from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from option import BaseOptions
import os
import pdb
import tensorflow as tf
from dataset import init_datasets, plot_all
from train import train
from test import test
n_viz = 1



if __name__ == '__main__':
    option = BaseOptions().parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu_id
    os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
    # 只显示 Error  
    # parameters 
    # Data Initiaization                                                    
    [data_x, data_y, data_w, data_z] = init_datasets(option)
    
    """ Construct model and training ops """
    [sess, DG_xyw, DG_xz, DG_yz, DG_wz, model3]=train(option, data_x, data_y, data_w, data_z)
    
    dmb1 = [[0]*2]*128
    dmb2 = [[1]*2]*128    
    
    result = test(option, data_x.get_dataset_test(), data_y.get_dataset_test(), data_w.get_dataset_test(), data_z.get_dataset_test() ,model3,n_viz,sess,dmb1,dmb2)

    # print all
    plot_all(option, DG_xz, DG_yz, DG_wz, DG_xyw, result, data_x.get_label_test(), data_y.get_label_test(), data_w.get_label_test(), data_z.get_label_test(), n_viz)   
