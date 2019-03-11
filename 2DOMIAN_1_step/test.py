from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from network import standard_normal,x_generative_network,x_inference_network,y_generative_network,y_inference_network,x_data_network,y_data_network

from model import DG

from option import BaseOptions
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

from utils.data_utils import shuffle, iter_data
from tqdm import tqdm
from dataset import init_datasets, plot, plot_lr, print_xy, print_xz, print_yz

from utils.data_gmm import GMM_distribution, sample_GMM, plot_GMM
#from train import train

def test(option, X_np_data_test, Y_np_data_test, Z_np_data_test, model,n_viz,sess):
    q_xz = model.get_q_xz()
    rec_xz = model.get_rec_xz()
    p_x = model.get_p_x()
    q_yz = model.get_q_yz()
    rec_yz = model.get_rec_yz()
    p_y = model.get_p_y()
    rec_zx = model.get_rec_zx()
    rec_zy = model.get_rec_zy()
    x = model.get_x()
    y = model.get_y()
    z = model.get_z()
    d1 = model.get_d1()
    d2 = model.get_d2()

    dmb1 = [[1]*2]*128
    dmb2 = [[0]*2]*128

    temp = np.array([]);

    im = [temp]*6
    rm = [temp]*8

    imxz = np.array([]); imzx = np.array([]); imyzx = np.array([]);
    imyz = np.array([]); imzy = np.array([]); imxzy = np.array([]); 
    
    rmxzyzx = np.array([]); rmyzy = np.array([]); rmxzyz = np.array([]); 
    rmyzxzy = np.array([]); rmxzx = np.array([]); rmyzxz = np.array([]); 
    rmzxz = np.array([]); rmzyz = np.array([]);

    batch_size = option.batch_size

    for _ in range(n_viz):
        for xmb, ymb, zmb in iter_data(X_np_data_test, Y_np_data_test, Z_np_data_test, size=batch_size):
            temp_imzx = sess.run(p_x, feed_dict={x: xmb, y: ymb, z:zmb, d1:dmb1, d2:dmb2})
            imzx = np.vstack([imzx, temp_imzx]) if imzx.size else temp_imzx
            
            temp_imzy = sess.run(p_y, feed_dict={x: xmb, y: ymb, z:zmb, d1:dmb1, d2:dmb2})
            imzy = np.vstack([imzy, temp_imzy]) if imzy.size else temp_imzy

            temp_imxz = sess.run(q_xz, feed_dict={x: xmb, y: ymb, z:zmb, d1:dmb2, d2:dmb1})
            imxz = np.vstack([imxz, temp_imxz]) if imxz.size else temp_imxz

            temp_imy = sess.run(p_y, feed_dict={x: xmb, y: ymb, z:temp_imxz, d1:dmb2, d2:dmb1})
            imxzy = np.vstack([imxzy, temp_imy]) if imxzy.size else temp_imy

            temp_imyz = sess.run(q_yz, feed_dict={x: xmb, y: ymb, z:zmb, d1:dmb2, d2:dmb1})
            imyz = np.vstack([imyz, temp_imyz]) if imyz.size else temp_imyz

            temp_imx = sess.run(p_x, feed_dict={x: xmb, y: ymb, z:temp_imyz, d1:dmb2, d2:dmb1})
            imyzx = np.vstack([imyzx, temp_imx]) if imyzx.size else temp_imx

            temp_rmz = sess.run(rec_yz, feed_dict={x: xmb, y: temp_imy, z:zmb, d1:dmb2, d2:dmb1})
            rmxzyz = np.vstack([rmxzyz, temp_rmz]) if rmxzyz.size else temp_rmz

            temp_rmx = sess.run(rec_zx, feed_dict={x: xmb, y: temp_imy, z:temp_rmz, d1:dmb2, d2:dmb1})
            rmxzyzx = np.vstack([rmxzyzx, temp_rmx]) if rmxzyzx.size else temp_rmx
            
            temp_rmx = sess.run(rec_zy, feed_dict={x: xmb, y: ymb, z:temp_rmz, d1:dmb2, d2:dmb1})
            rmyzy = np.vstack([rmyzy, temp_rmx]) if rmyzy.size else temp_rmx

            temp_rmz = sess.run(rec_xz, feed_dict={x: temp_imx, y: ymb, z:zmb, d1:dmb2, d2:dmb1})
            rmyzxz = np.vstack([rmyzxz, temp_rmz]) if rmyzxz.size else temp_rmz

            temp_rmx = sess.run(rec_zy, feed_dict={x: temp_imx, y: ymb, z:temp_rmz, d1:dmb2, d2:dmb1})
            rmyzxzy = np.vstack([rmyzxzy, temp_rmx]) if rmyzxzy.size else temp_rmx

            temp_rmx = sess.run(rec_zx, feed_dict={x: xmb, y: ymb, z:temp_rmz, d1:dmb2, d2:dmb1})
            rmxzx = np.vstack([rmxzx, temp_rmx]) if rmxzx.size else temp_rmx
            
            temp_rmzxz = sess.run(rec_xz, feed_dict={x: xmb, y: ymb, z:zmb, d1:dmb1, d2:dmb2})
            rmzxz = np.vstack([rmzxz, temp_rmzxz]) if rmzxz.size else temp_rmzxz
            
            temp_rmzyz = sess.run(rec_yz, feed_dict={x: xmb, y: ymb, z:zmb, d1:dmb1, d2:dmb2})
            rmzyz = np.vstack([rmzyz, temp_rmzyz]) if rmzyz.size else temp_rmzyz

    im[0]=imxz
    im[1]=imzx
    im[2]=imyzx
    im[3]=imyz
    im[4]=imzy
    im[5]=imxzy

    rm[0]=rmxzyzx
    rm[1]=rmyzy
    rm[2]=rmxzyz
    rm[3]=rmyzxzy
    rm[4]=rmxzx
    rm[5]=rmyzxz
    rm[6]=rmzxz
    rm[7]=rmzyz
    
    return im, rm