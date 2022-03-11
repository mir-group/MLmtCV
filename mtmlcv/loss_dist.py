from time import gmtime, strftime

import numpy as np

import tensorflow as tf
import tensorflow.keras as K
from mtmlcv.loss import Loss


class LossDist(Loss):

    loss_terms = ["reconst", "dist", "reg", "pe", "label", "kl"]

    def __init__(self, model, coeffs):
        self.loss_terms = LossDist.loss_terms
        super(LossDist, self).__init__(model, coeffs)

    def dist_loss(self, model, predictions, data, w):
        A = predictions["latent"]
        r = tf.reduce_sum(A * A, 1)
        r = tf.reshape(r, [-1, 1])
        D_l = r - 2 * tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
        predictions["D_l"] = D_l
        # distance in latent space

        A = data["pe"]
        r = tf.reduce_sum(A * A, 1)
        r = tf.reshape(r, [-1, 1])
        D_p = r - 2 * tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
        predictions["D_p"] = D_p

        # # sketch-map like loss function
        F_l = tf.sigmoid(D_l / self.latent_scale)
        F_p = tf.sigmoid(D_p / 0.000625)
        predictions["F_l"] = F_l
        predictions["F_p"] = F_p
        difference = tf.subtract(F_l, F_p)
        difference = tf.square(difference)

        # # distance loss function
        # difference = tf.subtract(D_l/self.latent_scale, D_p/0.025)
        # difference = tf.square(difference)

        # F_l = 1-tf.sigmoid(tf.divide(D_l-0.5, 0.1))
        # predictions["F_l"] = F_l
        # predictions["F_p"] = F_p

        # w = tf.stop_gradient(F_l)
        # w = tf.stop_gradient(1-tf.sigmoid(tf.divide(D_p-0.1, 0.000625)))
        # # w = F_p+F_l

        # predictions["dist_w"] = w
        # details["average_F_l"] = tf.reduce_mean(F_l)
        # details["average_F_p"] = tf.reduce_mean(F_p)
        # details["direct_sum"] = tf.reduce_sum(difference)
        # difference = tf.multiply(w, difference)
        # details["weighted_sum"] = tf.reduce_sum(difference)

        if "label" in data.keys():
            v = tf.broadcast_to(data["label"], tf.shape(D_p))
            mask = tf.dtypes.cast(tf.equal(v, tf.transpose(v)), dtype=tf.float32)
            mask = mask - tf.eye(tf.shape(D_p)[0])
            mask = tf.stop_gradient(mask)
            predictions["mask"] = mask
            # details["counted_pairs"] = tf.reduce_sum(mask)
            # details["total_pairs"] = tf.shape(D_p)[0]*tf.shape(D_p)[0]
            difference = tf.multiply(difference, mask)

            # ones = tf.ones_like(difference)
            # mask_a = tf.matrix_band_part(ones, 0, -1) # Upper triangular matrix of 0s and 1s
            # mask_b = tf.matrix_band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
            # mask = tf.cast(mask_a - mask_b, dtype=tf.bool) # Make a bool mask
            # upper_mask = tf.stop_gradient(mask)
            # difference = tf.boolean_mask(difference, upper_mask)
        else:
            mask = -tf.eye(tf.shape(D_p)[0]) + 1

        return tf.reduce_sum(difference) / tf.reduce_sum(mask)  # self.mse(F_l, F_p)
