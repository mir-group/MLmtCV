"""
Tools for training

TO DO:
    parameter initialization for the weights and bias
"""
import tensorflow as tf
import numpy as np
import copy

# TO DO, double check the dimensions...


def jacobian(y, x):
    num_y = y.shape[1]
    y_list = tf.unstack(y, num=num_y, axis=1)
    jab = []
    for i in range(num_y):
        g = tf.gradients(y_list[i], x)
        jab += [g[0]]
    jab = tf.stack(jab, axis=-2)
    return jab

    # jacobian_list = [[
    #     tf.gradients(y_, x)[0][i]
    #     for y_ in tf.unstack(y_list[i])]
    #     for i in range(num_y)]
    # return tf.stack(jacobian_list)


# from mtmlcv.separate_pe_models import JointAE

# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# from sklearn import manifold


def init_weights_kaiming_uniform(m):
    """ Weight initialization for linear layers """
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.fill_(0.01)


def init_weights_kaiming_normal(m):
    """ Weight initialization for linear layers """
    if type(m) == nn.Linear:
        nn.init.kaiming_normal(m.weight)
        m.bias.data.fill_(0.01)


def init_weights_xavier_uniform(m):
    """ Weight initialization for linear layers """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


def init_weights_xavier_normal(m):
    """ Weight initialization for linear layers """
    if type(m) == nn.Linear:
        nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0.01)


def init_weights(model, init_scheme=None):
    """ Initiliaze Network weight for model based on common schemes"""

    if init_scheme == None:
        return model
    elif init_scheme == "xavier-uniform":
        model.apply(init_weights_xavier_uniform)
    elif init_scheme == "xavier-normal":
        model.apply(init_weights_xavier_normal)
    elif init_scheme == "kaiming-uniform":
        model.apply(init_weights_kaiming_uniform)
    elif init_scheme == "kaiming-normal":
        model.apply(init_weights_kaiming_normal)

    return model
