from utils.data_gmm import GMM_distribution, sample_GMM, plot_GMM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import time

from network import standard_normal,x_generative_network,x_inference_network,y_generative_network,y_inference_network,x_data_network,y_data_network
graph_replace = tf.contrib.graph_editor.graph_replace

class Trainer():
    def set_train_gen_op_x(self, train_gen_op_x):
        self.train_gen_op_x = train_gen_op_x

    def set_train_gen_op_y(self, train_gen_op_y):
        self.train_gen_op_y = train_gen_op_y

    def set_train_gen_op_xy(self, train_gen_op_xy):
        self.train_gen_op_xy = train_gen_op_xy
    
    def set_train_disc_op_x(self, train_disc_op_x):
        self.train_disc_op_x = train_disc_op_x

    def set_train_disc_op_y(self, train_disc_op_y):
        self.train_disc_op_y = train_disc_op_y

    def set_train_disc_op_xy(self, train_disc_op_xy):
        self.train_disc_op_xy = train_disc_op_xy

    def get_train_gen_op_x(self):
        return self.train_gen_op_x

    def get_train_gen_op_y(self):
        return self.train_gen_op_y

    def get_train_gen_op_xy(self):
        return self.train_gen_op_xy
    
    def get_train_disc_op_x(self):
        return self.train_disc_op_x

    def get_train_disc_op_y(self):
        return self.train_disc_op_y

    def get_train_disc_op_xy(self):
        return self.train_disc_op_xy

    def initial(self, model, opt, qvars, pvars, dvars, qvars_y, pvars_y, dvars_y):
        self.train_gen_op_x = opt.minimize(model.get_gen_loss_x(), var_list=qvars + pvars)
        self.train_disc_op_x = opt.minimize(model.get_disc_loss_x(), var_list=dvars)
        self.train_gen_op_y =  opt.minimize(model.get_gen_loss_y(), var_list=qvars_y + pvars_y)
        self.train_disc_op_y = opt.minimize(model.get_disc_loss_y(), var_list=dvars_y)        
        self.train_gen_op_xy =  opt.minimize(model.get_gen_loss_xy(), var_list=qvars + pvars + qvars_y + pvars_y)
        self.train_disc_op_xy = opt.minimize(model.get_disc_loss_xy(), var_list=dvars + dvars_y)

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

    def get_z(self):
        return self.z

    def get_d1(self):
        return self.d1
    
    def get_d2(self):
        return self.d2  
    
    def get_p_x(self):
        return self.p_x

    def get_p_y(self):
        return self.p_y

    def get_q_xz(self):
        return self.q_xz
    
    def get_q_yz(self):
        return self.q_yz

    def get_decoder_logit_xz(self):
        return self.decoder_logit_xz

    def get_decoder_logit_yz(self):
        return self.decoder_logit_yz

    def get_encoder_logit_xz(self):
        return self.encoder_logit_xz

    def get_encoder_logit_yz(self):
        return self.encoder_logit_yz
    
    def get_decoder_loss_xz(self):
        return self.decoder_loss_xz
    
    def get_decoder_loss_yz(self):
        return self.decoder_loss_yz

    def get_encoder_loss_xz(self):
        return self.encoder_loss_xz
    
    def get_encoder_loss_yz(self):
        return self.encoder_loss_yz

    def get_disc_loss_x(self):
        return self.disc_loss_x

    def get_disc_loss_y(self):
        return self.disc_loss_y

    def get_rec_xz(self):
        return self.rec_xz
    
    def get_rec_yz(self):
        return self.rec_yz

    def get_rec_zy(self):
        return self.rec_zy

    def get_rec_zx(self):
        return self.rec_zx

    def get_cost_xz(self):
        return self.cost_xz

    def get_cost_yz(self):
        return self.cost_yz

    def get_cost_x(self):
        return self.cost_x
    
    def get_cost_y(self):
        return self.cost_y
    
    def get_decoder_loss2_xz(self):
        return self.decoder_loss2_xz

    def get_decoder_loss2_yz(self):
        return self.decoder_loss2_yz

    def get_encoder_loss2_xz(self):
        return self.encoder_loss2_xz

    def get_encoder_loss2_yz(self):
        return self.encoder_loss2_yz
    
    def get_gen_loss_xz(self):
        return self.gen_loss_xz

    def get_gen_loss_yz(self):
        return self.gen_loss_yz

    def get_gen_loss_x(self):
        return self.gen_loss_x
    
    def get_gen_loss_y(self):
        return self.gen_loss_y

    def get_gen_loss_xy(self):
        return self.gen_loss_xy

    def get_disc_loss_xy(self):
        return self.disc_loss_xy

    def copy(self, model):
        self.x = model.get_x()
        self.y = model.get_y()
        self.z = model.get_z()
        self.d1 = model.get_d1()
        self.d2 = model.get_d2()
        self.p_x = model.get_p_x()
        self.p_y = model.get_p_y()
        self.q_xz = model.get_q_xz()
        self.q_yz = model.get_q_yz()
        self.decoder_logit_xz = model.get_decoder_logit_xz()
        self.decoder_logit_yz = model.get_decoder_logit_yz()
        self.encoder_logit_xz = model.get_encoder_logit_xz()
        self.encoder_logit_yz = model.get_encoder_logit_yz()
        self.decoder_loss_xz = model.get_decoder_loss_xz()
        self.decoder_loss_yz = model.get_decoder_loss_yz()
        self.encoder_loss_xz = model.get_encoder_loss_xz()
        self.encoder_loss_yz = model.get_encoder_loss_yz()
        self.disc_loss_x = model.get_disc_loss_x()
        self.disc_loss_y = model.get_disc_loss_y()
        self.rec_xz = model.get_rec_xz()
        self.rec_yz = model.get_rec_yz()
        self.rec_zx = model.get_rec_zx()
        self.rec_zy = model.get_rec_zy()
        self.cost_xz = model.get_cost_xz()
        self.cost_yz = model.get_cost_yz()
        self.cost_x = model.get_cost_x()
        self.cost_y = model.get_cost_y()
        self.decoder_loss2_xz = model.get_decoder_loss2_xz()
        self.decoder_loss2_yz = model.get_decoder_loss2_yz()
        self.encoder_loss2_xz = model.get_encoder_loss2_xz()
        self.encoder_loss2_yz = model.get_encoder_loss2_yz()
        self.gen_loss_xz = model.get_gen_loss_xz()
        self.gen_loss_yz = model.get_gen_loss_yz()
        self.gen_loss_x = model.get_gen_loss_x()
        self.gen_loss_y = model.get_gen_loss_y()
        self.gen_loss_xy = model.get_gen_loss_xy()
        self.disc_loss_xy = model.get_disc_loss_xy()

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
        self.z = tf.placeholder(tf.float32, shape=(batch_size, latent_dim)) 
        self.d1 = tf.placeholder(tf.float32,shape=(batch_size, latent_dim)) 
        self.d2 = tf.placeholder(tf.float32,shape=(batch_size, latent_dim)) 

        self.q_xz = x_inference_network(self.x, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)
        self.q_yz = y_inference_network(self.y, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)

        #print(self.d)
        z = tf.multiply(self.z, self.d1)
        q_xz = tf.multiply(self.q_xz, self.d2)
        q_yz = tf.multiply(self.q_yz, self.d2)

        xz_layer = tf.add(z,q_xz)
        yz_layer = tf.add(z,q_yz)

        self.p_x = x_generative_network(yz_layer, input_dim , n_layer_gen, n_hidden_gen, eps_dim)
        self.p_y = y_generative_network(xz_layer, input_dim , n_layer_gen, n_hidden_gen, eps_dim)

        self.decoder_logit_xz = x_data_network(self.p_x, yz_layer, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
        self.encoder_logit_xz = x_data_network(self.x, self.q_xz, n_layers=n_layer_disc, n_hidden=n_hidden_disc)

        self.decoder_logit_yz = y_data_network(self.p_y, xz_layer, n_layers=n_layer_disc, n_hidden=n_hidden_disc)
        self.encoder_logit_yz = y_data_network(self.y, self.q_yz, n_layers=n_layer_disc, n_hidden=n_hidden_disc)

        self.decoder_loss_xz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(self.decoder_logit_xz), logits=self.decoder_logit_xz)
        self.encoder_loss_xz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.encoder_logit_xz), logits=self.encoder_logit_xz)

        self.decoder_loss_yz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(self.decoder_logit_yz), logits=self.decoder_logit_yz)
        self.encoder_loss_yz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.encoder_logit_yz), logits=self.encoder_logit_yz)

        self.disc_loss_x = tf.reduce_mean(self.encoder_loss_xz) + tf.reduce_mean(self.decoder_loss_xz)
        self.disc_loss_y = tf.reduce_mean(self.encoder_loss_yz) + tf.reduce_mean(self.decoder_loss_yz)

        self.rec_xz = x_inference_network(self.p_x, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)
        self.rec_zx = x_generative_network(self.q_xz, input_dim , n_layer_gen, n_hidden_gen,  eps_dim)

        self.rec_yz = y_inference_network(self.p_y, latent_dim, n_layer_inf, n_hidden_inf, eps_dim)
        self.rec_zy = y_generative_network(self.q_yz, input_dim , n_layer_gen, n_hidden_gen,  eps_dim)

        self.cost_xz = tf.reduce_mean(tf.pow(self.rec_xz - yz_layer, 2))# no
        self.cost_yz = tf.reduce_mean(tf.pow(self.rec_yz - xz_layer, 2))# no

        self.cost_x = tf.reduce_mean(tf.pow(self.rec_zx - self.x, 2))# yes -> xzx
        self.cost_y = tf.reduce_mean(tf.pow(self.rec_zy - self.y, 2))# yes -> yzy

        self.decoder_loss2_xz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.decoder_logit_xz), logits=self.decoder_logit_xz)
        self.encoder_loss2_xz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(self.encoder_logit_xz), logits=self.encoder_logit_xz)

        self.decoder_loss2_yz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(self.decoder_logit_yz), logits=self.decoder_logit_yz)
        self.encoder_loss2_yz = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(self.encoder_logit_yz), logits=self.encoder_logit_yz)

        self.gen_loss_xz = tf.reduce_mean(self.decoder_loss2_xz) + tf.reduce_mean(self.encoder_loss2_xz)
        self.gen_loss_yz = tf.reduce_mean(self.decoder_loss2_yz) + tf.reduce_mean(self.encoder_loss2_yz)

        self.gen_loss_x = 1.*self.gen_loss_xz + 1.0*self.cost_x + 1.0*self.cost_xz
        self.gen_loss_y = 1.*self.gen_loss_yz + 1.0*self.cost_y + 1.0*self.cost_yz

        self.gen_loss_xy = self.gen_loss_x + self.gen_loss_y
        self.disc_loss_xy = self.disc_loss_x + self.disc_loss_y

        del z
        del q_xz
        del q_yz
        del xz_layer
        del yz_layer



