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

if __name__ == '__main__':
    option = BaseOptions().parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu_id
    os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
    # 只显示 Error  
    # parameters 
    # Data Initiaization                                                    
    [data_x, data_y, data_z] = init_datasets(option)

    """ Construct model and training ops """
    [sess,DG_xy,DG_xz, DG_yz,model3]=train(option, data_x, data_y, data_z)
    n_viz = 1
    [im, rm] = test(option, data_x.get_dataset_test(), data_y.get_dataset_test(), data_z.get_dataset_test() ,model3,n_viz,sess)
    
    # print all
    plot_all(option, DG_xz, DG_yz, DG_xy, im, rm, data_x.get_label_test(), data_y.get_label_test(), data_z.get_label_test(), n_viz)   
