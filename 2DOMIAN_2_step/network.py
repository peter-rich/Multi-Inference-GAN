from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
GPUID = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from utils.data_gmm import GMM_distribution, sample_GMM, plot_GMM
from utils.data_utils import shuffle, iter_data
from tqdm import tqdm

slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace

""" Networks """
def standard_normal(shape, **kwargs):
    """Create a standard Normal StochasticTensor."""
    return tf.cast(st.StochasticTensor(
        ds.MultivariateNormalDiag(mu=tf.zeros(shape), diag_stdev=tf.ones(shape), **kwargs)),  tf.float32)


def x_generative_network(z, input_dim, n_layer, n_hidden, eps_dim):
    with tf.variable_scope("generative_x"):
        eps = standard_normal([z.get_shape().as_list()[0], eps_dim], name="eps") * 1.0
        h = tf.concat([z, eps], 1) 
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        x = slim.fully_connected(h, input_dim, activation_fn=None, scope="p_x")
    return x

def y_generative_network(z, input_dim, n_layer, n_hidden, eps_dim):
    with tf.variable_scope("generative_y"):
        eps = standard_normal([z.get_shape().as_list()[0], eps_dim], name="eps") * 1.0
        h = tf.concat([z, eps], 1) 
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        y = slim.fully_connected(h, input_dim, activation_fn=None, scope="p_y")
    return y

def x_inference_network(x, latent_dim, n_layer, n_hidden, eps_dim):
    with tf.variable_scope("inference_x"):
        eps = standard_normal([x.get_shape().as_list()[0], eps_dim], name="eps") * 1.0
        h = tf.concat([x, eps], 1) 
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        z = slim.fully_connected(h, latent_dim, activation_fn=None, scope="q_xz")
    return z

def y_inference_network(y, latent_dim, n_layer, n_hidden, eps_dim):
    with tf.variable_scope("inference_y"):
        eps = standard_normal([y.get_shape().as_list()[0], eps_dim], name="eps") * 1.0
        h = tf.concat([y, eps], 1) 
        h = slim.repeat(h, n_layer, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        z = slim.fully_connected(h, latent_dim, activation_fn=None, scope="q_yz")
    return z

def x_data_network(x,z, n_layers=2, n_hidden=256, activation_fn=None):
    """Approximate x log data density."""
    h = tf.concat([x,z], 1)
    with tf.variable_scope('discriminator_x'):
        h = slim.repeat(h, n_layers, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=activation_fn)
    return tf.squeeze(log_d, squeeze_dims=[1])

def y_data_network(y,z, n_layers=2, n_hidden=256, activation_fn=None):
    """Approximate y log data density."""
    h = tf.concat([y,z], 1)
    with tf.variable_scope('discriminator_y'):
        h = slim.repeat(h, n_layers, slim.fully_connected, n_hidden, activation_fn=tf.nn.relu)
        log_d = slim.fully_connected(h, 1, activation_fn=activation_fn)
    return tf.squeeze(log_d, squeeze_dims=[1])