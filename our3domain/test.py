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
from dataset import init_datasets, plot, plot_lr, print_xy, print_xz, print_yz,Result

from utils.data_gmm import GMM_distribution, sample_GMM, plot_GMM
#from train import train

def test(option, X_np_data_test, Y_np_data_test, W_np_data_test, Z_np_data_test, model, n_viz,sess, dmb1, dmb2):
    q_xz = model.get_q_xz()
    rec_xz = model.get_rec_wzxz()
    p_x = model.get_p_wzx()

    q_yz = model.get_q_yz()
    rec_yz = model.get_rec_xzyz()
    p_y = model.get_p_xzy()

    rec_zx = model.get_rec_xzx()
    rec_zy = model.get_rec_yzy()
    rec_zw = model.get_rec_wzw()

    q_wz = model.get_q_wz()
    rec_wz = model.get_rec_yzwz()
    p_w = model.get_p_yzw()

    x = model.get_x()
    y = model.get_y()
    w = model.get_w()
    z = model.get_z()

    d1 = model.get_d1()
    d2 = model.get_d2()

    dmbb1 = dmb2
    dmbb2 = dmb1
    
    imxz = np.array([]); rmwzxz = np.array([]); imxzy = np.array([]); rmxzyzx = np.array([]); rmyzy = np.array([]); imzx = np.array([]);rmzxz = np.array([]);
    imyz = np.array([]); rmxzyz = np.array([]); imyzw = np.array([]); rmyzwzy = np.array([]); rmxzx = np.array([]); imzy = np.array([]);rmzyz = np.array([]);
    imwz = np.array([]); rmyzwz = np.array([]); imwzx = np.array([]); rmwzxzw = np.array([]); rmwzw = np.array([]); imzw = np.array([]);rmzwz = np.array([]);
    batch_size = option.batch_size
    for _ in range(n_viz):
        for xmb, wmb, ymb, zmb in iter_data(X_np_data_test, W_np_data_test, Y_np_data_test,  Z_np_data_test, size=batch_size):
            
            temp_imwz = sess.run(q_wz, feed_dict={x: xmb, y: ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            imwz = np.vstack([imwz, temp_imwz]) if imwz.size else temp_imwz    

            temp_imxz = sess.run(q_xz, feed_dict={x: xmb, y: ymb, w: wmb, z:zmb, d1:dmb1, d2:dmb2})
            imxz = np.vstack([imxz, temp_imxz]) if imxz.size else temp_imxz
            
            temp_imyz = sess.run(q_yz, feed_dict={x: xmb, y: ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            imyz = np.vstack([imyz, temp_imyz]) if imyz.size else temp_imyz

            temp_imy = sess.run(p_y, feed_dict={x: xmb, y: ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            imxzy = np.vstack([imxzy, temp_imy]) if imxzy.size else temp_imy

            temp_imw = sess.run(p_w, feed_dict={x: xmb, y: ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            imyzw = np.vstack([imyzw, temp_imw]) if imyzw.size else temp_imw

            temp_imx = sess.run(p_x, feed_dict={x: xmb, y: ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            imwzx = np.vstack([imwzx, temp_imx]) if imwzx.size else temp_imx

            ############# clip ****
            temp_rmyz = sess.run(rec_yz, feed_dict={x: xmb, y: temp_imy, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            rmxzyz = np.vstack([rmxzyz, temp_rmyz]) if rmxzyz.size else temp_rmyz

            temp_rmwz = sess.run(rec_wz, feed_dict={x: xmb, y: ymb, w:temp_imw, z:zmb, d1:dmb1, d2:dmb2})
            rmyzwz = np.vstack([rmyzwz, temp_rmwz]) if rmyzwz.size else temp_rmwz

            temp_rmxz = sess.run(rec_xz, feed_dict={x: temp_imx, y: ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            rmwzxz = np.vstack([rmwzxz, temp_rmxz]) if rmwzxz.size else temp_rmxz

            ############# clip
            temp_rmx = sess.run(rec_zx, feed_dict={x: xmb, y: temp_imy, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            rmxzyzx = np.vstack([rmxzyzx, temp_rmx]) if rmxzyzx.size else temp_rmx

            temp_rmx = sess.run(rec_zx, feed_dict={x: xmb, y: ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            rmxzx = np.vstack([rmxzx, temp_rmx]) if rmxzx.size else temp_rmx
            
            temp_rmy = sess.run(rec_zy, feed_dict={x: xmb, y: ymb, w:temp_imw, z:zmb, d1:dmb1, d2:dmb2})
            rmyzwzy = np.vstack([rmyzwzy, temp_rmy]) if rmyzwzy.size else temp_rmy

            temp_rmy = sess.run(rec_zy, feed_dict={x: xmb, y: ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            rmyzy = np.vstack([rmyzy, temp_rmy]) if rmyzy.size else temp_rmy
            
            temp_rmw = sess.run(rec_zw, feed_dict={x: temp_imx, y: ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            rmwzxzw = np.vstack([rmwzxzw, temp_rmw]) if rmwzxzw.size else temp_rmw
            
            temp_rmw = sess.run(rec_zw, feed_dict={x: xmb, y: ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            rmwzw = np.vstack([rmwzw, temp_rmw]) if rmwzw.size else temp_rmw           
    
            temp_imy = sess.run(p_y, feed_dict={x: xmb, y: ymb, w:wmb, z:zmb, d1:dmbb1, d2:dmbb2})
            imzy = np.vstack([imzy, temp_imy]) if imzy.size else temp_imy

            temp_imw = sess.run(p_w, feed_dict={x: xmb, y: ymb, w:wmb, z:zmb, d1:dmbb1, d2:dmbb2})
            imzw = np.vstack([imzw, temp_imw]) if imzw.size else temp_imw

            temp_imx = sess.run(p_x, feed_dict={x: xmb, y: ymb, w:wmb, z:zmb, d1:dmbb1, d2:dmbb2})
            imzx = np.vstack([imzx, temp_imx]) if imzx.size else temp_imx

            temp_rmyz = sess.run(rec_yz, feed_dict={x: xmb, y: temp_imy, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            rmzyz = np.vstack([rmzyz, temp_rmyz]) if rmzyz.size else temp_rmyz

            temp_rmwz = sess.run(rec_wz, feed_dict={x: xmb, y: ymb, w:temp_imw, z:zmb, d1:dmb1, d2:dmb2})
            rmzwz = np.vstack([rmzwz, temp_rmwz]) if rmzwz.size else temp_rmwz

            temp_rmxz = sess.run(rec_xz, feed_dict={x: temp_imx, y: ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            rmzxz = np.vstack([rmzxz, temp_rmxz]) if rmzxz.size else temp_rmxz


    result = Result()
    result.set_z_domian(imzx, imzy, imzw, rmzxz, rmzyz, rmzwz)
    result.set_all(imxz, imyz, imwz, imxzy, imyzw, imwzx, rmxzyz, rmyzwz, rmwzxz, rmxzyzx, rmyzy, rmyzwzy, rmxzx, rmwzxzw, rmwzw)
    return result