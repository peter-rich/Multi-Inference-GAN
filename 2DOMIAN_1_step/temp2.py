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
    
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
from network import standard_normal,x_generative_network,x_inference_network,y_generative_network,y_inference_network,x_data_network,y_data_network
from model import DG
from option import BaseOptions
import os
import pdb
import tensorflow as tf
from tqdm import tqdm
from dataset import init_datasets, plot, plot_lr, print_xy, print_xz, print_yz, plot_all
#from train import train
from test import test
graph_replace = tf.contrib.graph_editor.graph_replace
option = BaseOptions().parse()
os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu_id
# parameters 
n_epoch = option.n_epoch
batch_size = option.batch_size
input_dim = option.input_dim  
eps_dim = option.eps_dim
n_layer_disc = option.n_layer_disc
n_hidden_disc = option.n_hidden_disc
n_layer_gen = option.n_layer_gen
n_hidden_gen = option.n_hidden_gen
n_layer_inf = option.n_layer_inf
n_hidden_inf = option.n_hidden_inf
latent_dim = option.latent_dim
# Data Initiaization
[data_x, data_y, data_z] = init_datasets(option)

""" Construct model and training ops """
tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=(batch_size, input_dim))
y = tf.placeholder(tf.float32, shape=(batch_size, input_dim))
z = tf.placeholder(tf.float32, shape=(batch_size, latent_dim))

# decoder and encoder
p_x = x_generative_network(z, input_dim , n_layer_gen, n_hidden_gen, eps_dim)
p_y = y_generative_network(z, input_dim , n_layer_gen, n_hidden_gen, eps_dim)

q_xz = x_inference_network(x, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)
q_yz = y_inference_network(y, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)







###

decoder_logit_x = x_data_network(p_x, z, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
encoder_logit_x = x_data_network(x, q_xz, n_layers=n_layer_disc, n_hidden=n_hidden_disc)

decoder_logit_y = y_data_network(p_y, z, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
encoder_logit_y = y_data_network(y, q_yz, n_layers=n_layer_disc, n_hidden=n_hidden_disc)

decoder_loss_x = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(decoder_logit_x), logits=decoder_logit_x)
encoder_loss_x = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(encoder_logit_x), logits=encoder_logit_x)

decoder_loss_y = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(decoder_logit_y), logits=decoder_logit_y)
encoder_loss_y = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(encoder_logit_y), logits=encoder_logit_y)

disc_loss_x = tf.reduce_mean(encoder_loss_x) + tf.reduce_mean(decoder_loss_x)
disc_loss_y = tf.reduce_mean(encoder_loss_y) + tf.reduce_mean(decoder_loss_y)

rec_xz = x_inference_network(p_x, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)
rec_zx = x_generative_network(q_xz, input_dim , n_layer_gen, n_hidden_gen,  eps_dim)

rec_yz = y_inference_network(p_y, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)
rec_zy = y_generative_network(q_yz, input_dim , n_layer_gen, n_hidden_gen,  eps_dim)
#
#
#
cost_xz = tf.reduce_mean(tf.pow(rec_xz - z, 2))# no
cost_yz = tf.reduce_mean(tf.pow(rec_yz - z, 2))# no

cost_x = tf.reduce_mean(tf.pow(rec_zx - x, 2))# yes -> xzx
cost_y = tf.reduce_mean(tf.pow(rec_zy - y, 2))# yes -> yzy

decoder_loss2_x = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(decoder_logit_x), logits=decoder_logit_x)
encoder_loss2_x = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(encoder_logit_x), logits=encoder_logit_x)

decoder_loss2_y = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(decoder_logit_y), logits=decoder_logit_y)
encoder_loss2_y = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(encoder_logit_y), logits=encoder_logit_y)


gen_loss_xz = tf.reduce_mean(decoder_loss2_x) + tf.reduce_mean(encoder_loss2_x)
gen_loss_yz = tf.reduce_mean(decoder_loss2_y) + tf.reduce_mean(encoder_loss2_y)


gen_loss_x = 1.*gen_loss_xz + 1.0*cost_x + 1.0*cost_xz
gen_loss_y = 1.*gen_loss_yz + 1.0*cost_y + 1.0*cost_yz


qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference_x")
pvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative_x")
dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_x")

qvars_y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference_y")
pvars_y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative_y")
dvars_y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_y")

opt = tf.train.AdamOptimizer(1e-3, beta1=0.5)

train_gen_op_x =  opt.minimize(gen_loss_x, var_list=qvars + pvars)
train_disc_op_x = opt.minimize(disc_loss_x, var_list=dvars)

train_gen_op_y =  opt.minimize(gen_loss_y, var_list=qvars_y + pvars_y)
train_disc_op_y = opt.minimize(disc_loss_y, var_list=dvars_y)


""" training """
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

DG_xz = DG()
DG_xz.initial()
DG_yz = DG()
DG_yz.initial()
DG_xy = DG()
DG_xy.initial()   
    
FG_x = []
FG_y = []
FD_x = []
FD_y = []
FG_xy = []
FD_xy = []

X_dataset = data_x.get_dataset()
Y_dataset = data_y.get_dataset()
Z_dataset = data_z.get_dataset()

n_epoch = option.n_epoch
batch_size = option.batch_size
n_epoch_2 = option.n_epoch_2
for epoch in tqdm( range(n_epoch), total=n_epoch):
    X_dataset = shuffle(X_dataset)
    Y_dataset = shuffle(Y_dataset)
    Z_dataset = shuffle(Z_dataset)
    i = 0
    for xmb, ymb, zmb in iter_data(X_dataset, Y_dataset, Z_dataset, size=batch_size):
        i = i + 1
        for _ in range(1):
            f_d_x, _ = sess.run([disc_loss_x, train_disc_op_x], feed_dict={x: xmb, y:ymb, z:zmb})
        for _ in range(5):
            f_g_x, _ = sess.run([[gen_loss_x, gen_loss_xz, cost_x, cost_xz], train_gen_op_x], feed_dict={x: xmb, y:ymb, z:zmb})
        FG_x.append(f_g_x)
        FD_x.append(f_d_x)
        for _ in range(1):
            f_d_y, _ = sess.run([disc_loss_y, train_disc_op_y], feed_dict={x: xmb, y:ymb, z:zmb})
        for _ in range(5):
            f_g_y, _ = sess.run([[gen_loss_y, gen_loss_yz, cost_y, cost_yz], train_gen_op_y], feed_dict={x: xmb, y:ymb, z:zmb})
        FG_y.append(f_g_y)
        FD_y.append(f_d_y)
    print_xz(epoch, i, f_d_x, f_g_x[0], f_g_x[1], f_g_x[2], f_g_x[3])
    print_yz(epoch, i, f_d_y, f_g_y[0], f_g_y[1], f_g_y[2], f_g_y[3])



DG_xz.set_FD(FD_x)
DG_xz.set_FG(FG_x)
DG_yz.set_FD(FD_y)
DG_yz.set_FG(FG_y)
 


q_xz = x_inference_network(x, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)
p_y = y_generative_network(q_xz, input_dim , n_layer_gen, n_hidden_gen, eps_dim)

q_yz = y_inference_network(y, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)
p_x = x_generative_network(q_yz, input_dim , n_layer_gen, n_hidden_gen, eps_dim)

decoder_logit_x = x_data_network(x, q_xz, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
encoder_logit_x = y_data_network(p_y,q_xz, n_layers=n_layer_disc, n_hidden=n_hidden_disc)

decoder_logit_y = y_data_network(y,q_yz, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
encoder_logit_y = x_data_network(p_x, q_yz, n_layers=n_layer_disc, n_hidden=n_hidden_disc)

#
#
decoder_loss_x = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(decoder_logit_x), logits=decoder_logit_x)
encoder_loss_y = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(encoder_logit_x), logits=encoder_logit_x)

decoder_loss_y = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(decoder_logit_y), logits=decoder_logit_y)
encoder_loss_x = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(encoder_logit_y), logits=encoder_logit_y)

#
#
disc_loss_x = tf.reduce_mean(encoder_loss_x) + tf.reduce_mean(decoder_loss_x)
disc_loss_y = tf.reduce_mean(encoder_loss_y) + tf.reduce_mean(decoder_loss_y)

disc_loss = disc_loss_x + disc_loss_y# + disc_loss_z# + 0.001*noise_loss

#
#
rec_yz = y_inference_network(p_y, latent_dim, n_layer_inf, n_hidden_inf, eps_dim )
rec_zx = x_generative_network(q_xz, input_dim , n_layer_gen, n_hidden_gen,  eps_dim )

rec_xz = x_inference_network(p_x, latent_dim, n_layer_inf, n_hidden_inf, eps_dim )
rec_zy = y_generative_network(q_yz, input_dim , n_layer_gen, n_hidden_gen,  eps_dim )

cost_yz = tf.reduce_mean(tf.pow(rec_yz - q_xz, 2))
cost_zx = tf.reduce_mean(tf.pow(rec_zx - x, 2))

cost_xz = tf.reduce_mean(tf.pow(rec_xz - q_yz, 2))
cost_zy = tf.reduce_mean(tf.pow(rec_zy - y, 2))

cost_x = cost_xz + cost_zx
cost_y = cost_yz + cost_zy

decoder_loss2_x = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(decoder_logit_x), logits=decoder_logit_x)
encoder_loss2_x = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(encoder_logit_x), logits=encoder_logit_x)

decoder_loss2_y = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(decoder_logit_y), logits=decoder_logit_y)
encoder_loss2_y = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(encoder_logit_y), logits=encoder_logit_y)

gen_loss_xz = tf.reduce_mean(  decoder_loss2_x )  + tf.reduce_mean( encoder_loss2_x )
gen_loss_yz = tf.reduce_mean(  decoder_loss2_y )  + tf.reduce_mean( encoder_loss2_y )


gen_loss_x = 1.*gen_loss_xz + 1.0*cost_zx  + 1.0*cost_xz
gen_loss_y = 1.*gen_loss_yz + 1.0*cost_zy  + 1.0*cost_yz
gen_loss = gen_loss_x + gen_loss_y# + gen_loss_z #+ 0.002*cost_noise# + 0.001*gen_loss_noise

qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference_x")
pvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative_x")
dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_x")

qvars_y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference_y")
pvars_y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative_y")
dvars_y = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_y")

train_gen_op_xy =  opt.minimize(gen_loss, var_list= qvars + pvars + qvars_y + pvars_y)
train_disc_op_xy = opt.minimize(disc_loss, var_list= dvars + dvars_y)

X_dataset = data_x.get_dataset()
Y_dataset = data_y.get_dataset()
Z_dataset = data_z.get_dataset()

for epoch in tqdm( range(n_epoch_2), total=n_epoch_2):
    X_dataset = shuffle(X_dataset)
    Y_dataset = shuffle(Y_dataset)
    Z_dataset = shuffle(Z_dataset)
    i = 0
    for xmb, ymb, zmb in iter_data(X_dataset, Y_dataset, Z_dataset, size=batch_size):
        i = i + 1
        for _ in range(1):
            f_d_xy, _ = sess.run([disc_loss, train_disc_op_xy], feed_dict={x: xmb, y:ymb, z:zmb})
        for _ in range(5):
            f_g_xy, _ = sess.run([[gen_loss, gen_loss_xz, cost_x, cost_xz], train_gen_op_xy], feed_dict={x: xmb, y:ymb, z:zmb})
        FG_xy.append(f_g_xy)
        FD_xy.append(f_d_xy)
    print_xy(epoch, i, f_d_xy, f_g_xy[0], f_g_xy[1], f_g_xy[2], f_g_xy[3])

DG_xy.set_FD(FD_xy)
DG_xy.set_FG(FG_xy)  
#[sess,DG_xz,DG_yz,DG_xy]=train(option, sess, data_x.get_dataset(), data_y.get_dataset(), data_z.get_dataset(), print_xz, train_gen_op_xy, train_disc_op_xy, train_gen_op_x, train_disc_op_x, gen_loss_x, disc_loss_x,train_gen_op_y, train_disc_op_y, gen_loss_y, disc_loss_y, x, y, z,gen_loss_xz, cost_x, cost_xz, gen_loss_yz, cost_y, cost_yz, gen_loss_xy, disc_loss_xy, cost_yzxz, cost_xzyz)

""" testing the results """
n_viz = 1
[imxz,rmxz,imzx,rmzx,imyz,rmyz,imzy,rmzy,rmyzy,rmxzx] = test(option, data_x.get_dataset_test(), data_y.get_dataset_test(), data_z.get_dataset_test(), q_xz,rec_xz,p_x,q_yz,rec_yz,p_y,rec_zx,rec_zy,n_viz,sess, x, y, z)
# print all
plot_all(option, DG_xz, DG_yz, DG_xy, imxz, rmxz, imzx, rmzx, imyz, rmyz, imzy, rmzy, rmyzy, rmxzx, data_x.get_label_test(), data_y.get_label_test(), data_z.get_label_test(), n_viz)

