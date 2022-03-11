from time import gmtime, strftime

import numpy as np

from sklearn.metrics import mean_absolute_error, accuracy_score

from mtmlcv.utils import init_weights
from mtmlcv.data_loader import Data_Loader
from mtmlcv.separate_pe_models import JointAE

import tensorflow as tf
import tensorflow.keras as K

import sklearn


class LossSmooth(Loss):

    loss_terms = ["reconst", "dist", "reg", "pe", "label", "kl", "smooth"]

    def __init__(self, model, coeffs):
        self.loss_terms = LossDist.loss_terms
        super(LossSmooth, self).__init__(model, coeffs)

    def smooth_loss(self, model, predictions, data, w):

        # L2 norm for smoothness
        xmax = tf.reduce_max(predictions["latent"], axis=0)[0]
        xmin = tf.reduce_min(predictions["latent"], axis=0)[0]
        dx = (xmax - xmin) / 1004.0
        x = tf.reshape(tf.range(xmin, xmax, delta=dx), [-1, 1])
        y = model.pe_net(x)
        ypp = y[4:, :]
        y0 = y[2:-2, :]
        ymm = y[:-4, :]

        return tf.reduce_sum(((ypp + ymm - 2 * y0) / (2 * dx) ** 2) ** 2)
