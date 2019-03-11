
from model import DG, Model, Trainer
from test import test
from dataset import plot_all
from option import BaseOptions
import tensorboard
import time
import os
import pdb
import numpy as np
from network import standard_normal,x_generative_network,x_inference_network,y_generative_network,y_inference_network,x_data_network,y_data_network
from network import w_generative_network,w_inference_network,w_data_network
import tensorflow as tf
option = BaseOptions().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu_id

from utils.data_utils import shuffle, iter_data
from tqdm import tqdm
from dataset import init_datasets, plot, plot_lr, print_xy, print_xz, print_yz, print_wz
n_viz = 1 
batch_size = option.batch_size

def train_model2(train_op, n_epoch_2, opt, model1, sess, X_dataset, Y_dataset, W_dataset, Z_dataset):

    train_gen_op_xyw =  train_op.get_train_gen_op_xyw()
    train_disc_op_xyw = train_op.get_train_disc_op_xyw()

    FD_xyw = []
    FG_xyw = []
    DG_xyw = DG()
    DG_xyw.initial()

    dmb1 = [[0]*2]*128
    dmb2 = [[1]*2]*128

    x = model1.get_x()
    y = model1.get_y()
    w = model1.get_w()
    z = model1.get_z()
    d1 = model1.get_d1()
    d2 = model1.get_d2()

    for epoch in tqdm( range(n_epoch_2), total=n_epoch_2):
        X_dataset = shuffle(X_dataset)
        Y_dataset = shuffle(Y_dataset)
        W_dataset = shuffle(W_dataset)
        Z_dataset = shuffle(Z_dataset)
        i = 0
        for xmb, ymb, wmb, zmb in iter_data(X_dataset, Y_dataset, W_dataset, Z_dataset, size=batch_size):
            i = i + 1
            for _ in range(1):
                f_d_xyw, _ = sess.run([model1.get_disc_loss_xyw(), train_disc_op_xyw], feed_dict={x: xmb, y:ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            for _ in range(5):
                f_g_xyw, _ = sess.run([[model1.get_gen_loss_xyw(), model1.get_gen_loss_x(), model1.get_cost_x(), model1.get_cost_xz()], train_gen_op_xyw], feed_dict={x: xmb, y:ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            FG_xyw.append(f_g_xyw)
            FD_xyw.append(f_d_xyw)
        print_xy(epoch, i, f_d_xyw, f_g_xyw[0], f_g_xyw[1], f_g_xyw[2], f_g_xyw[3])
    DG_xyw.set_FD(FD_xyw)
    DG_xyw.set_FG(FG_xyw)

    return sess, DG_xyw, model1

def train_model1(sess, train_op, n_epoch, opt, model1, X_dataset, Y_dataset, W_dataset, Z_dataset):
    train_gen_op_x =  train_op.get_train_gen_op_x()
    train_disc_op_x = train_op.get_train_disc_op_x()
    train_gen_op_y =  train_op.get_train_gen_op_y()
    train_disc_op_y = train_op.get_train_disc_op_y()
    train_gen_op_w =  train_op.get_train_gen_op_w()
    train_disc_op_w = train_op.get_train_disc_op_w()

    x = model1.get_x()
    y = model1.get_y()
    w = model1.get_w()
    z = model1.get_z()
    d1 = model1.get_d1()
    d2 = model1.get_d2()
    
    FG_x = []
    FG_y = []
    FG_w = []
    FD_x = []
    FD_y = []
    FD_w = []
    DG_xz = DG()
    DG_xz.initial()
    DG_yz = DG()
    DG_yz.initial()
    DG_wz = DG()
    DG_wz.initial()


    
    for epoch in tqdm( range(n_epoch), total=n_epoch):
        X_dataset = shuffle(X_dataset)
        Y_dataset = shuffle(Y_dataset)
        W_dataset = shuffle(W_dataset)
        Z_dataset = shuffle(Z_dataset)
        i = 0
        dmb1 = [[1]*2]*128
        dmb2 = [[0]*2]*128
        #print(dmb)
        for xmb, ymb, wmb, zmb in iter_data(X_dataset, Y_dataset, W_dataset, Z_dataset, size=batch_size):
            #print(xmb)
            i = i + 1
            for _ in range(1):
                f_d_w, _ = sess.run([model1.get_disc_loss_w(), train_disc_op_w], feed_dict={x: xmb, y:ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            for _ in range(5):
                f_g_w, _ = sess.run([[model1.get_gen_loss_w(), model1.get_gen_loss_wz(), model1.get_cost_w(), model1.get_cost_wz()], train_gen_op_w], feed_dict={x: xmb, y:ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            FG_w.append(f_g_w)
            FD_w.append(f_d_w)
            for _ in range(1):
                f_d_x, _ = sess.run([model1.get_disc_loss_x(), train_disc_op_x], feed_dict={x: xmb, y:ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            for _ in range(5):
                f_g_x, _ = sess.run([[model1.get_gen_loss_x(), model1.get_gen_loss_xz(), model1.get_cost_x(), model1.get_cost_xz()], train_gen_op_x], feed_dict={x: xmb, y:ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            FG_x.append(f_g_x)
            FD_x.append(f_d_x)
            for _ in range(1):
                f_d_y, _ = sess.run([model1.get_disc_loss_y(), train_disc_op_y], feed_dict={x: xmb, y:ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            for _ in range(5):
                f_g_y, _ = sess.run([[model1.get_gen_loss_y(), model1.get_gen_loss_yz(), model1.get_cost_y(), model1.get_cost_yz()], train_gen_op_y], feed_dict={x: xmb, y:ymb, w:wmb, z:zmb, d1:dmb1, d2:dmb2})
            FG_y.append(f_g_y)
            FD_y.append(f_d_y)

        print_xz(epoch, i, f_d_x, f_g_x[0], f_g_x[1], f_g_x[2], f_g_x[3])
        print_yz(epoch, i, f_d_y, f_g_y[0], f_g_y[1], f_g_y[2], f_g_y[3])
        print_wz(epoch, i, f_d_w, f_g_w[0], f_g_w[1], f_g_w[2], f_g_w[3])

    DG_xz.set_FD(FD_x)
    DG_xz.set_FG(FG_x)
    DG_yz.set_FD(FD_y)
    DG_yz.set_FG(FG_y)
    DG_wz.set_FD(FD_w)
    DG_wz.set_FG(FG_w)

    return sess, DG_xz, DG_yz, DG_wz, model1

def train(option, data_x, data_y, data_w, data_z):

    setattr(tf.GraphKeys, "VARIABLES", "variables")
    DG_xz = DG()
    DG_xz.initial()
    DG_yz = DG()
    DG_yz.initial()
    DG_wz = DG()
    DG_wz.initial()
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

    qvars_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference_w")
    pvars_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative_w")
    dvars_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_w")

    opt = tf.train.AdamOptimizer(1e-3, beta1=0.5)

    train_op = Trainer()
    train_op.initial(model1, opt, qvars, pvars, dvars, qvars_y, pvars_y, dvars_y, qvars_w, pvars_w, dvars_w)
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer()) 
    sess.graph.finalize()

    """ training """    
    X_dataset = data_x.get_dataset()
    Y_dataset = data_y.get_dataset()
    W_dataset = data_w.get_dataset()
    Z_dataset = data_z.get_dataset()
    
    [sess, DG_xz, DG_yz, DG_wz, model1] = train_model1(sess, train_op, option.n_epoch, opt, model1, X_dataset, Y_dataset, W_dataset, Z_dataset)
    #global summary_writer
    #summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs', sess.graph)
    
    
    [sess, DG_xy, model1] = train_model2(train_op, option.n_epoch_2,opt, model1, sess, X_dataset, Y_dataset, W_dataset, Z_dataset)
    
   
    for i in range(0,option.n_iter):
        #if(i!=0):
            
            #saver = tf.train.Saver()
            #saver.restore(sess=sess,save_path="model/my-model"+str(i-1)+".ckpt")
        time_start=time.time()
        print("#####################")
        print("iteration:")
        print(i)
        print("#####################")
        print("#####################")
        [sess, DG_xz, DG_yz, DG_wz, model1] = train_model1(sess, train_op, option.n_epoch_3, opt, model1, X_dataset, Y_dataset, W_dataset, Z_dataset)
        print("#####################")
        print("#####################")
        [sess, DG_xy, model1] = train_model2(train_op, option.n_epoch_4,opt, model1, sess, X_dataset, Y_dataset, W_dataset, Z_dataset)
        print("#####################")
        #saver = tf.train.Saver()
        #saver.save(sess=sess,save_path="model/my-model"+str(i)+".ckpt")
        #summary_writer.add_summary(summary_str, total_step)
        time_end=time.time()
        print('Model2: totally cost',time_end-time_start) 
    #
    [sess, DG_xz, DG_yz, DG_wz, model1] =  train_model1(sess, train_op, option.n_epoch_5, opt, model1, X_dataset, Y_dataset, W_dataset, Z_dataset)
    [sess, DG_xy, model1] = train_model2(train_op, option.n_epoch_6, opt, model1, sess, X_dataset, Y_dataset, W_dataset, Z_dataset)
    
    return sess, DG_xy, DG_xz, DG_yz, DG_wz,  model1