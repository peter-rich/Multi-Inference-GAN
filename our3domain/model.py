from utils.data_gmm import GMM_distribution, sample_GMM, plot_GMM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import time

from network import standard_normal,x_generative_network,x_inference_network,y_generative_network,y_inference_network,x_data_network,y_data_network
from network import w_generative_network,w_inference_network,w_data_network

graph_replace = tf.contrib.graph_editor.graph_replace

class Trainer():
    def set_train_gen_op_x(self, train_gen_op_x):
        self.train_gen_op_x = train_gen_op_x

    def set_train_gen_op_y(self, train_gen_op_y):
        self.train_gen_op_y = train_gen_op_y

    def set_train_gen_op_w(self, train_gen_op_w):
        self.train_gen_op_w = train_gen_op_w

    def set_train_gen_op_xyw(self, train_gen_op_xyw):
        self.train_gen_op_xyw = train_gen_op_xyw
    
    def set_train_disc_op_x(self, train_disc_op_x):
        self.train_disc_op_x = train_disc_op_x

    def set_train_disc_op_y(self, train_disc_op_y):
        self.train_disc_op_y = train_disc_op_y
    
    def set_train_disc_op_w(self, train_disc_op_w):
        self.train_disc_op_w = train_disc_op_w

    def set_train_disc_op_xyw(self, train_disc_op_xyw):
        self.train_disc_op_xyz = train_disc_op_xyw

    def get_train_gen_op_x(self):
        return self.train_gen_op_x

    def get_train_gen_op_y(self):
        return self.train_gen_op_y

    def get_train_gen_op_w(self):
        return self.train_gen_op_w

    def get_train_gen_op_xyw(self):
        return self.train_gen_op_xyw
    
    def get_train_disc_op_x(self):
        return self.train_disc_op_x

    def get_train_disc_op_y(self):
        return self.train_disc_op_y

    def get_train_disc_op_w(self):
        return self.train_disc_op_w

    def get_train_disc_op_xyw(self):
        return self.train_disc_op_xyw

    def initial(self, model, opt, qvars, pvars, dvars, qvars_y, pvars_y, dvars_y, qvars_w, pvars_w, dvars_w):
        self.train_gen_op_x =   opt.minimize(model.get_gen_loss_x(), var_list=qvars + pvars)
        self.train_disc_op_x =  opt.minimize(model.get_disc_loss_x(), var_list=dvars)
        self.train_gen_op_y =   opt.minimize(model.get_gen_loss_y(), var_list=qvars_y + pvars_y)
        self.train_disc_op_y =  opt.minimize(model.get_disc_loss_y(), var_list=dvars_y)   
        self.train_gen_op_w =   opt.minimize(model.get_gen_loss_w(), var_list=qvars_w + pvars_w)
        self.train_disc_op_w =  opt.minimize(model.get_disc_loss_w(), var_list=dvars_w)      
        self.train_gen_op_xyw = opt.minimize(model.get_gen_loss_xyw(), var_list=qvars + pvars + qvars_y + pvars_y + qvars_w + pvars_w)
        self.train_disc_op_xyw =opt.minimize(model.get_disc_loss_xyw(), var_list=dvars + dvars_y + dvars_w)

class DG():
    def set_FD(self, FD):
        self.FD = FD
    
    def set_FG(self, FG):
        self.FG = FG

    def get_FD(self):
        return self.FD
    
    def get_FG(self):
        return self.FG

    def initial(self):
        self.FD = []
        self.FG = []

# TO do List
#         

class Model():

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_w(self):
        return self.w
    
    def get_z(self):
        return self.z

    def get_d1(self):
        return self.d1
    
    def get_d2(self):
        return self.d2  
    
    def get_p_wzx(self):
        return self.p_wzx

    def get_p_xzy(self):
        return self.p_xzy
    
    def get_p_yzw(self):
        return self.p_yzw

    def get_p_yzx(self):
        return self.p_yzx

    def get_p_wzy(self):
        return self.p_wzy
    
    def get_p_xzw(self):
        return self.p_xzw

    def get_q_xz(self):
        return self.q_xz
    
    def get_q_yz(self):
        return self.q_yz

    def get_q_wz(self):
        return self.q_wz

    def get_decoder_logit_xz(self):
        return self.decoder_logit_xz

    def get_decoder_logit_yz(self):
        return self.decoder_logit_yz

    def get_decoder_logit_wz(self):
        return self.decoder_logit_wz

    def get_encoder_logit_xz(self):
        return self.encoder_logit_xz

    def get_encoder_logit_yz(self):
        return self.encoder_logit_yz

    def get_encoder_logit_wz(self):
        return self.encoder_logit_wz

    def get_decoder_loss_xz(self):
        return self.decoder_loss_xz
    
    def get_decoder_loss_yz(self):
        return self.decoder_loss_yz

    def get_decoder_loss_wz(self):
        return self.decoder_loss_wz

    def get_encoder_loss_xz(self):
        return self.encoder_loss_xz
    
    def get_encoder_loss_yz(self):
        return self.encoder_loss_yz

    def get_encoder_loss_wz(self):
        return self.encoder_loss_wz

    def get_disc_loss_x(self):
        return self.disc_loss_x

    def get_disc_loss_y(self):
        return self.disc_loss_y

    def get_disc_loss_w(self):
        return self.disc_loss_w

    def get_rec_wzxz(self):
        return self.rec_wzxz
    
    def get_rec_xzyz(self):
        return self.rec_xzyz

    def get_rec_yzwz(self):
        return self.rec_yzwz

    def get_rec_yzy(self):
        return self.rec_yzy

    def get_rec_xzx(self):
        return self.rec_xzx

    def get_rec_wzw(self):
        return self.rec_wzw

    def get_cost_xz(self):
        return self.cost_xz

    def get_cost_yz(self):
        return self.cost_yz

    def get_cost_wz(self):
        return self.cost_wz

    def get_cost_x(self):
        return self.cost_x
    
    def get_cost_y(self):
        return self.cost_y

    def get_cost_w(self):
        return self.cost_w

    def get_decoder_loss2_xz(self):
        return self.decoder_loss2_xz

    def get_decoder_loss2_yz(self):
        return self.decoder_loss2_yz
    
    def get_decoder_loss2_wz(self):
        return self.decoder_loss2_wz

    def get_encoder_loss2_xz(self):
        return self.encoder_loss2_xz

    def get_encoder_loss2_yz(self):
        return self.encoder_loss2_yz

    def get_encoder_loss2_wz(self):
        return self.encoder_loss2_wz

    def get_gen_loss_xz(self):
        return self.gen_loss_xz

    def get_gen_loss_yz(self):
        return self.gen_loss_yz

    def get_gen_loss_wz(self):
        return self.gen_loss_wz

    def get_gen_loss_x(self):
        return self.gen_loss_x
    
    def get_gen_loss_y(self):
        return self.gen_loss_y

    def get_gen_loss_w(self):
        return self.gen_loss_w

    def get_gen_loss_xyw(self):
        return self.gen_loss_xyw

    def get_disc_loss_xyw(self):
        return self.disc_loss_xyw
    
    def initial(self, option):
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
        
        self.x = tf.placeholder(tf.float32, shape=(batch_size, input_dim))
        self.y = tf.placeholder(tf.float32, shape=(batch_size, input_dim))
        self.w = tf.placeholder(tf.float32, shape=(batch_size, input_dim))
    
        self.z = tf.placeholder(tf.float32, shape=(batch_size, latent_dim)) 
         
        self.d1 = tf.placeholder(tf.float32,shape=(batch_size, latent_dim)) 
        self.d2 = tf.placeholder(tf.float32,shape=(batch_size, latent_dim)) 

        self.q_wz = w_inference_network(self.w, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)
        self.q_xz = x_inference_network(self.x, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)
        self.q_yz = y_inference_network(self.y, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)
        
        
        one = [[0]*2]*128
        zero = [[1]*2]*128
        
        #print(self.d)
        z = tf.multiply(self.z, self.d1)
        
        q_xz = tf.multiply(self.q_xz, self.d2)
        q_yz = tf.multiply(self.q_yz, self.d2)
        q_wz = tf.multiply(self.q_wz, self.d2)

        xz_layer = tf.add(z,q_xz)
        yz_layer = tf.add(z,q_yz)
        wz_layer = tf.add(z,q_wz)

        self.p_wzx = x_generative_network(wz_layer, input_dim , n_layer_gen, n_hidden_gen, eps_dim)
        self.p_yzx = x_generative_network(yz_layer, input_dim , n_layer_gen, n_hidden_gen, eps_dim)
        self.p_yzw = w_generative_network(yz_layer, input_dim , n_layer_gen, n_hidden_gen, eps_dim)
        self.p_xzw = w_generative_network(xz_layer, input_dim , n_layer_gen, n_hidden_gen, eps_dim)
        self.p_xzy = y_generative_network(xz_layer, input_dim , n_layer_gen, n_hidden_gen, eps_dim)
        self.p_wzy = y_generative_network(wz_layer, input_dim , n_layer_gen, n_hidden_gen, eps_dim)

        self.decoder_logit_wz = w_data_network(self.p_yzw, yz_layer, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
        self.encoder_logit_wz = w_data_network(self.w, self.q_wz, n_layers=n_layer_disc, n_hidden=n_hidden_disc)

        self.decoder_logit_xz = x_data_network(self.p_wzx, wz_layer, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
        self.encoder_logit_xz = x_data_network(self.x, self.q_xz, n_layers=n_layer_disc, n_hidden=n_hidden_disc)

        self.decoder_logit_yz = y_data_network(self.p_xzy, xz_layer, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
        self.encoder_logit_yz = y_data_network(self.y, self.q_yz, n_layers=n_layer_disc, n_hidden=n_hidden_disc)

        self.decoder_loss_xz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(self.decoder_logit_xz), logits=self.decoder_logit_xz)
        self.encoder_loss_xz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.encoder_logit_xz), logits=self.encoder_logit_xz)

        self.decoder_loss_yz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(self.decoder_logit_yz), logits=self.decoder_logit_yz)
        self.encoder_loss_yz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.encoder_logit_yz), logits=self.encoder_logit_yz)

        self.decoder_loss_wz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(self.decoder_logit_wz), logits=self.decoder_logit_wz)
        self.encoder_loss_wz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.encoder_logit_wz), logits=self.encoder_logit_wz)

        self.disc_loss_x = tf.reduce_mean(self.encoder_loss_xz) + tf.reduce_mean(self.decoder_loss_xz)
        self.disc_loss_y = tf.reduce_mean(self.encoder_loss_yz) + tf.reduce_mean(self.decoder_loss_yz)
        self.disc_loss_w = tf.reduce_mean(self.encoder_loss_wz) + tf.reduce_mean(self.decoder_loss_wz)

        self.rec_wzxz = x_inference_network(self.p_wzx, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)
        self.rec_xzx = x_generative_network(self.q_xz, input_dim , n_layer_gen, n_hidden_gen,  eps_dim)

        self.rec_xzyz = y_inference_network(self.p_xzy, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)
        self.rec_yzy = y_generative_network(self.q_yz, input_dim , n_layer_gen, n_hidden_gen,  eps_dim)

        self.rec_yzwz = w_inference_network(self.p_yzw, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)
        self.rec_wzw = w_generative_network(self.q_wz, input_dim , n_layer_gen, n_hidden_gen,  eps_dim)

        self.cost_xz = tf.reduce_mean(tf.pow(self.rec_wzxz - wz_layer, 2))# no
        self.cost_yz = tf.reduce_mean(tf.pow(self.rec_xzyz - xz_layer, 2))# no
        self.cost_wz = tf.reduce_mean(tf.pow(self.rec_yzwz - yz_layer, 2))# no

        self.cost_x = tf.reduce_mean(tf.pow(self.rec_xzx - self.x, 2))# yes -> xzx
        self.cost_y = tf.reduce_mean(tf.pow(self.rec_yzy - self.y, 2))# yes -> yzy
        self.cost_w = tf.reduce_mean(tf.pow(self.rec_wzw - self.w, 2))# yes -> wzw

        self.decoder_loss2_xz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.decoder_logit_xz), logits=self.decoder_logit_xz)
        self.encoder_loss2_xz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(self.encoder_logit_xz), logits=self.encoder_logit_xz)

        self.decoder_loss2_yz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.decoder_logit_yz), logits=self.decoder_logit_yz)
        self.encoder_loss2_yz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(self.encoder_logit_yz), logits=self.encoder_logit_yz)

        self.decoder_loss2_wz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.decoder_logit_wz), logits=self.decoder_logit_wz)
        self.encoder_loss2_wz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(self.encoder_logit_wz), logits=self.encoder_logit_wz)

        self.gen_loss_xz = tf.reduce_mean(self.decoder_loss2_xz) + tf.reduce_mean(self.encoder_loss2_xz)
        self.gen_loss_yz = tf.reduce_mean(self.decoder_loss2_yz) + tf.reduce_mean(self.encoder_loss2_yz)
        self.gen_loss_wz = tf.reduce_mean(self.decoder_loss2_wz) + tf.reduce_mean(self.encoder_loss2_wz)

        self.gen_loss_x = 1.*self.gen_loss_xz + 1.0*self.cost_x + 1.0*self.cost_xz
        self.gen_loss_y = 1.*self.gen_loss_yz + 1.0*self.cost_y + 1.0*self.cost_yz
        self.gen_loss_w = 1.*self.gen_loss_wz + 1.0*self.cost_w + 1.0*self.cost_wz

        self.gen_loss_xyw = self.gen_loss_x + self.gen_loss_y + self.gen_loss_w
        self.disc_loss_xyw = self.disc_loss_x + self.disc_loss_y + self.disc_loss_w

        del z
        del q_xz
        del q_yz
        del q_wz
        del xz_layer
        del wz_layer
        del yz_layer



