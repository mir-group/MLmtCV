from time import gmtime, strftime

import numpy as np

import tensorflow as tf
import tensorflow.keras as K

from mtmlcv.loss import Loss


class LossVAE(Loss):

    loss_terms = ["reconst", "dist", "reg", "pe", "label", "kl"]

    def __init__(self, model, coeffs):
        self.loss_terms = LossDist.loss_terms
        super(LossDist, self).__init__(model, coeffs)

    def kl_loss(self, model, predictions, data, w):
        mu = predictions["mu"]
        logvar = predictions["logvar"]
        return 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(logvar) - 1.0 - logvar)
