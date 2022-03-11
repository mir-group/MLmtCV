"""

class for the neural net model

TO DO:
    network for encoder
      decoder
      pe prediction
      label prediction and decision function
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Reshape,
import copy
from math import pi
import numpy as np

from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

from mtmlcv.joint_model import JointAE as original_JAE
from mtmlcv.joint_model import Sparse
from mtmlcv.utils import jacobian


class Rotation(tf.keras.layers.Layer):
    def __init__(self):  # , tehta, phi, alpha):
        super(Rotation, self).__init__()

    def call(self, x, training=False):

        # shift
        x0 = tf.reshape(tf.stop_gradient(x[:, 8, :]), [-1, 1, 3])
        x0 = tf.matmul(tf.ones((x.shape[1], 1)), x0)

        newx = tf.subtract(x, x0)

        dr = tf.stop_gradient(newx[:, 10, :])
        r_tc = tf.norm(dr, axis=-1)
        r_tcxy = tf.norm(dr[:, :2], axis=-1)
        ct = dr[:, 2] / r_tc
        st = tf.sqrt(1 - ct ** 2)
        cp = dr[:, 0] / r_tcxy
        sp = dr[:, 1] / r_tcxy
        r1 = tf.stack([ct * cp, -sp, st * cp], axis=-1)
        r2 = tf.stack([ct * sp, cp, st * sp], axis=-1)
        r3 = tf.stack([-st, ct - ct, ct], axis=-1)
        rotation1 = tf.stack([r1, r2, r3], axis=-2)
        rotation1 = tf.stop_gradient(rotation1)
        newx = tf.matmul(newx, rotation1)

        dr = tf.stop_gradient(newx[:, 6, :])
        r_lcxy = tf.norm(dr[:, :2], axis=-1)
        cp = dr[:, 0] / r_lcxy
        sp = dr[:, 1] / r_lcxy
        r1 = tf.stack([cp, sp, sp - sp])
        r2 = tf.stack([-sp, cp, sp - sp])
        r3 = tf.stack([sp - sp, sp - sp, sp - sp + 1])
        rotation2 = tf.transpose(tf.stack([r1, r2, r3], axis=0))
        rotation2 = tf.stop_gradient(rotation2)
        newx = tf.matmul(newx, rotation2)

        return newx


class JointAE(original_JAE):
    """ Autoencoder that learns jointly on reconstruction of internal coordinations and labels + potential energy """

    def __init__(
        self,
        n_features,
        n_latent,
        model_name="default",
        encoder_arch=[],
        dropoutrate=0.0,
        bn=False,
        bnmomentum=0.9,
        use_reconst=False,
        decoder_arch=[],
        use_pe=False,
        pe_arch=[],
        use_labels=True,
        labels_arch=[],
        nclass=2,
        onehot=False,
        use_dist=False,
        gaus_latent=None,
        gaus_pe_latent=None,
        gaus_pe=None,
        gaus_input=None,
        input_mean=None,
        input_std=None,
        output_mean=None,
        output_std=None,
        af=tf.nn.elu,
    ):

        super(JointAE, self).__init__(
            n_features=n_features,
            n_latent=n_latent,
            model_name=model_name,
            encoder_arch=encoder_arch,
            dropoutrate=dropoutrate,
            bn=bn,
            bnmomentum=bnmomentum,
            use_reconst=use_reconst,
            decoder_arch=decoder_arch,
            use_pe=use_pe,
            pe_arch=pe_arch,
            use_labels=use_labels,
            labels_arch=labels_arch,
            nclass=nclass,
            onehot=onehot,
            use_dist=use_dist,
            gaus_latent=gaus_latent,
            gaus_pe_latent=gaus_pe_latent,
            gaus_pe=gaus_pe,
            gaus_input=gaus_input,
            input_mean=input_mean,
            input_std=input_std,
            output_mean=output_mean,
            output_std=output_std,
            af=af,
        )

        del self.encode

        if input_mean is not None:
            self.input_mean = tf.convert_to_tensor(input_mean)
            self.input_std = tf.convert_to_tensor(input_std)

            def input_normalize(input_x):
                return tf.divide(tf.subtract(input_x, self.input_mean), self.input_std)

            def input_unnormalize(input_x):
                return tf.add(tf.multiply(input_x, self.input_std), self.input_mean)

        if output_mean is not None:
            self.output_mean = tf.convert_to_tensor(output_mean)
            self.output_std = tf.convert_to_tensor(output_std)

            def output_unnormalize(output_x):
                return tf.add(tf.multiply(output_x, self.output_std), self.output_mean)

        # construct encoder container
        self.reshape1 = Reshape((int(n_features / 3), 3), input_shape=(n_features,))
        self.rot = Rotation()
        self.reshape2 = Reshape((n_features,), input_shape=(int(n_features / 3), 3))

        self.preprocess = keras.models.Sequential()
        self.preprocess.add(self.reshape1)
        self.preprocess.add(self.rot)
        self.preprocess.add(self.reshape2)
        # normalize = Activation(input_normalize)
        # if (input_mean is not None):
        #     self.preprocess.add(normalize)

        model = keras.models.Sequential()
        model.add(self.reshape1)
        model.add(self.rot)
        model.add(self.reshape2)
        # if (input_mean is not None):
        #     model.add(normalize)

        self.encode = self.build_net(
            n_features, n_latent, encoder_arch, bn, dropoutrate, self.af, model=model
        )

        self.add_encoder_kernels()
