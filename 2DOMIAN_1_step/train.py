from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from network import standard_normal,x_generative_network,x_inference_network,y_generative_network,y_inference_network,x_data_network,y_data_network
from model import DG, Model, Trainer
from test import test
from dataset import plot_all
from option import BaseOptions
import tensorboard
import time
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
option = BaseOptions().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu_id

from utils.data_utils import shuffle, iter_data
from tqdm import tqdm
from dataset import init_datasets, plot, plot_lr, print_xy, print_xz, print_yz
from utils.data_gmm import GMM_distribution, sample_GMM, plot_GMM
n_viz = 1 
batch_size = option.batch_size

def train_model2(train_op, n_epoch_2, opt, model1, sess, X_dataset, Y_dataset, Z_dataset, qvars, pvars, dvars, qvars_y, pvars_y, dvars_y):

    train_gen_op_xy =  train_op.get_train_gen_op_xy()
    train_disc_op_xy = train_op.get_train_disc_op_xy()

    FD_xy = []
    FG_xy = []
    DG_xy = DG()
    DG_xy.initial()

    dmb1 = [[0]*2]*128
    dmb2 = [[1]*2]*128

    x = model1.get_x()
    y = model1.get_y()
    z = model1.get_z()
    d1 = model1.get_d1()
    d2 = model1.get_d2()

    for epoch in tqdm( range(n_epoch_2), total=n_epoch_2):
        X_dataset = shuffle(X_dataset)
        Y_dataset = shuffle(Y_dataset)
        Z_dataset = shuffle(Z_dataset)
        i = 0
        for xmb, ymb, zmb in iter_data(X_dataset, Y_dataset, Z_dataset, size=batch_size):
            i = i + 1
            for _ in range(1):
                f_d_xy, _ = sess.run([model1.get_disc_loss_xy(), train_disc_op_xy], feed_dict={x: xmb, y:ymb, z:zmb, d1:dmb1, d2:dmb2})
            for _ in range(5):
                f_g_xy, _ = sess.run([[model1.get_gen_loss_xy(), model1.get_gen_loss_x(), model1.get_cost_x(), model1.get_cost_xz()], train_gen_op_xy], feed_dict={x: xmb, y:ymb, z:zmb, d1:dmb1, d2:dmb2})
            FG_xy.append(f_g_xy)
            FD_xy.append(f_d_xy)
        print_xy(epoch, i, f_d_xy, f_g_xy[0], f_g_xy[1], f_g_xy[2], f_g_xy[3])
    DG_xy.set_FD(FD_xy)
    DG_xy.set_FG(FG_xy)

    return sess, DG_xy, model1

def train_model1(sess, train_op, n_epoch, opt, model1, X_dataset, Y_dataset, Z_dataset, qvars, pvars, dvars, qvars_y, pvars_y, dvars_y):
    train_gen_op_x =  train_op.get_train_gen_op_x()
    train_disc_op_x = train_op.get_train_disc_op_x()
    train_gen_op_y =  train_op.get_train_gen_op_y()
    train_disc_op_y = train_op.get_train_disc_op_y()

    x = model1.get_x()
    y = model1.get_y()
    z = model1.get_z()
    d1 = model1.get_d1()
    d2 = model1.get_d2()
    
    FG_x = []
    FG_y = []
    FD_x = []
    FD_y = []
    DG_xz = DG()
    DG_xz.initial()
    DG_yz = DG()
    DG_yz.initial()
    
    dmb1 = [[1]*2]*128
    dmb2 = [[0]*2]*128
    
    for epoch in tqdm( range(n_epoch), total=n_epoch):
        X_dataset = shuffle(X_dataset)
        Y_dataset = shuffle(Y_dataset)
        Z_dataset = shuffle(Z_dataset)
        i = 0
        
        #print(dmb)
        for xmb, ymb, zmb in iter_data(X_dataset, Y_dataset, Z_dataset, size=batch_size):
            #print(xmb)
            i = i + 1
            for _ in range(1):
                f_d_x, _ = sess.run([model1.get_disc_loss_x(), train_disc_op_x], feed_dict={x: xmb, y:ymb, z:zmb, d1:dmb1, d2:dmb2})
            for _ in range(5):
                f_g_x, _ = sess.run([[model1.get_gen_loss_x(), model1.get_gen_loss_xz(), model1.get_cost_x(), model1.get_cost_xz()], train_gen_op_x], feed_dict={x: xmb, y:ymb, z:zmb, d1:dmb1, d2:dmb2})
            FG_x.append(f_g_x)
            FD_x.append(f_d_x)
            for _ in range(1):
                f_d_y, _ = sess.run([model1.get_disc_loss_y(), train_disc_op_y], feed_dict={x: xmb, y:ymb, z:zmb, d1:dmb1, d2:dmb2})
            for _ in range(5):
                f_g_y, _ = sess.run([[model1.get_gen_loss_y(), model1.get_gen_loss_yz(), model1.get_cost_y(), model1.get_cost_yz()], train_gen_op_y], feed_dict={x: xmb, y:ymb, z:zmb, d1:dmb1, d2:dmb2})
            FG_y.append(f_g_y)
            FD_y.append(f_d_y)
        print_xz(epoch, i, f_d_x, f_g_x[0], f_g_x[1], f_g_x[2], f_g_x[3])
        
        print_yz(epoch, i, f_d_y, f_g_y[0], f_g_y[1], f_g_y[2], f_g_y[3])

    DG_xz.set_FD(FD_x)
    DG_xz.set_FG(FG_x)
    DG_yz.set_FD(FD_y)
    DG_yz.set_FG(FG_y)

    return sess, DG_xz, DG_yz, model1


def train(option, data_x, data_y, data_z):
    n_iter = option.n_iter
    setattr(tf.GraphKeys, "VARIABLES", "variables")
    DG_xz = DG()
    DG_xz.initial()
    DG_yz = DG()
    DG_yz.initial()
    DG_xy = DG()
    DG_xy.initial()   

    tf.reset_default_graph()
    

    
    model1 = Model()    
    model1.initial(option)

    # decoder and encoder
    qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference_x")
    pvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative_x")
    dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_x")

    qvars_y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference_y")
    pvars_y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative_y")
    dvars_y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_y")

    opt = tf.train.AdamOptimizer(1e-3, beta1=0.5)

    train_op = Trainer()
    train_op.initial(model1, opt, qvars, pvars, dvars, qvars_y, pvars_y, dvars_y)
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer()) 
    sess.graph.finalize()

    DG_xz = DG()
    DG_xz.initial()
    DG_yz = DG()
    DG_yz.initial()
    DG_xy = DG()    
    DG_xy.initial()  

    """ training """    
    X_dataset = data_x.get_dataset()
    Y_dataset = data_y.get_dataset()
    Z_dataset = data_z.get_dataset()
    
    [sess, DG_xz, DG_yz, model1] = train_model1(sess, train_op, option.n_epoch, opt, model1, X_dataset, Y_dataset, Z_dataset, qvars, pvars, dvars, qvars_y, pvars_y, dvars_y)
    
    #global summary_writer
    #summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs', sess.graph)
    
    '''
    [sess, DG_xy, model1] = train_model2(train_op, option.n_epoch_2,opt, model1, sess, X_dataset, Y_dataset, Z_dataset, qvars, pvars, dvars, qvars_y, pvars_y, dvars_y)
    
    
    for i in range(0,n_iter):
        #if(i!=0):
            
            #saver = tf.train.Saver()
            #saver.restore(sess=sess,save_path="model/my-model"+str(i-1)+".ckpt")
            
            
        time_start=time.time()
        print("#####################")
        print("iteration:")
        print(i)
        print("#####################")
        print("#####################")
        print("#####################")
        #[sess, DG_xz, DG_yz, model1] = train_model1_3(train_op, option.n_epoch_3, opt, model1, sess, X_dataset, Y_dataset, Z_dataset, qvars, pvars, dvars, qvars_y, pvars_y, dvars_y)
        [sess, DG_xz, DG_yz, model1] = train_model1(train_op, option.n_epoch_3, opt, model1, sess, X_dataset, Y_dataset, Z_dataset, qvars, pvars, dvars, qvars_y, pvars_y, dvars_y)
        print("#####################")
        
        
        print("#####################")
        [sess, DG_xy, model1] = train_model2(train_op, option.n_epoch_4,opt, model1, sess, X_dataset, Y_dataset, Z_dataset, qvars, pvars, dvars, qvars_y, pvars_y, dvars_y)
        print("#####################")
        #saver = tf.train.Saver()
        #saver.save(sess=sess,save_path="model/my-model"+str(i)+".ckpt")
        #summary_writer.add_summary(summary_str, total_step)
        time_end=time.time()
        print('Model1_2: totally cost',time_end-time_start)
    
    #
    [sess, DG_xz, DG_yz, model1] =  train_model1(train_op, option.n_epoch_5, opt, model1, sess, X_dataset, Y_dataset, Z_dataset, qvars, pvars, dvars, qvars_y, pvars_y, dvars_y)
    [sess, DG_xy, model1] =         train_model2(train_op, option.n_epoch_6, opt, model1, sess, X_dataset, Y_dataset, Z_dataset, qvars, pvars, dvars, qvars_y, pvars_y, dvars_y)
    '''
    return sess, DG_xy, DG_xz, DG_yz, model1