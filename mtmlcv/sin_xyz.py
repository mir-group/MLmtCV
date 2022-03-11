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
from tensorflow.keras.layers import Reshape

from mtmlcv.joint_model import JointAE as original_JAE

two_pi = 3.141592653589793*2

class Wrap(keras.layers.Layer):

    def __init__(self, lx=None, ly=None, lz=None):
        super(Wrap, self).__init__()
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.dim = 3
        if self.lx is not None:
            self.dim += 1
        if self.ly is not None:
            self.dim += 1
        if self.lz is not None:
            self.dim += 1

    def call(self, x, training=False):

        xx, yy, zz = tf.split(x, num_or_size_splits=3, axis=-1)
        if self.lx is not None:
            xx_s = tf.sin(xx/self.lx*two_pi)*self.lx
            xx_c = tf.cos(xx/self.lx*two_pi)*self.lx
            stack = [xx_s, xx_c]
        else:
            stack = [xx]
        if self.ly is not None:
            yy_s = tf.sin(yy/self.ly*two_pi)*self.ly
            yy_c = tf.cos(yy/self.ly*two_pi)*self.ly
            stack += [yy_s, yy_c]
        else:
            stack += [yy]
        if self.lz is not None:
            zz_s = tf.sin(zz/self.lz*two_pi)*self.lz
            zz_c = tf.cos(zz/self.lz*two_pi)*self.lz
            stack += [zz_s, zz_c]
        else:
            stack += [zz]
        newx = tf.stack(stack, axis=2)
        print("stack", newx.shape)
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
        cell=None,
        af=tf.nn.elu,
        lx=None,
        ly=None,
        lz=None
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

        self.use_aux = False

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
        reshape1 = Reshape((n_features // 3, 3), input_shape=(n_features,))
        wrap = Wrap(lx=lx, ly=ly, lz=lz)
        reshape2 = Reshape((n_features // 3 * wrap.dim,), input_shape=(n_features // 3, wrap.dim))

        self.preprocess = keras.models.Sequential()
        self.preprocess.add(reshape1)
        self.preprocess.add(wrap)
        self.preprocess.add(reshape2)

        # normalize = Activation(input_normalize)
        # if (input_mean is not None):
        #     self.preprocess.add(normalize)

        model = keras.models.Sequential()
        model.add(reshape1)
        model.add(wrap)
        model.add(reshape2)
        # if (input_mean is not None):
        #     model.add(normalize)

        del self.encode
        self.encode = self.build_net(
            n_features // 3 * wrap.dim, n_latent, encoder_arch, bn, dropoutrate, self.af, model=model
        )

        self.all_nets["encode"] = self.encode

        self.reg_vars = []
        for var in self.encode.trainable_variables:
            try:
                if "kernel" in var.name:
                    self.reg_vars += [tf.reshape(var, [1, -1])]
            except Exception as e:
                print(f"{var} fail:  {e}")
        self.reg_vars = tf.concat(self.reg_vars, axis=-1)
