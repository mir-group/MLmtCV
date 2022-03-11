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
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
    Softmax,
    GaussianNoise,
    Activation,
)


class JointAE(keras.Model):
    """ Autoencoder that learns jointly on reconstruction of internal coordinations and labels + potential energy """

    def __init__(
        self,
        n_features,
        n_latent,
        encoder_arch,
        decoder_arch,
        dropoutrate=0.0,
        bn=False,
        use_reconst=False,
        use_labels=True,
        labels_arch=None,
        use_pe=False,
        pe_arch=None,
        onehot=False,
        gaus_latent=0,
        gaus_pe_latent=0,
        gaus_pe=0,
        gaus_input=0,
        input_mean=0,
        input_std=1,
        output_mean=0,
        output_std=1,
        af=tf.nn.relu,
    ):

        super(JointAE, self).__init__()
        self.n_features = n_features
        self.n_latent = n_latent
        self.encoder_arch = encoder_arch
        self.decoder_arch = decoder_arch
        self.use_reconst = use_reconst
        self.use_labels = use_labels
        self.use_pe = use_pe
        self.dropoutrate = dropoutrate
        self.bn = bn
        self.af = af  # tf.nn.relu

        # if (input_mean is 0):
        #     input_mean = np.zeros(n_features)
        #     input_std = np.ones(n_features)
        # if (output_mean is 0):
        #     output_mean = np.zeros(n_features)
        #     output_std = np.ones(n_features)

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
        model = keras.models.Sequential()
        if input_mean is not None:
            model.add(Activation(input_normalize))
        self.encode = self.build_net(
            n_features, n_latent, encoder_arch, bn, dropoutrate, self.af, model=model
        )
        self.gaus_latent = gaus_latent
        self.gaus_pe = gaus_pe
        self.gaus_pe_latent = gaus_pe_latent
        self.gaus_input = gaus_input

        if gaus_latent > 0:
            self.noise_latent = GaussianNoise(stddev=gaus_latent)

        if gaus_pe > 0:
            self.noise_pe = GaussianNoise(stddev=gaus_pe)

        if gaus_pe_latent > 0:
            self.noise_pe_latent = GaussianNoise(stddev=gaus_latent)

        if gaus_input > 0:
            self.noise_input = GaussianNoise(stddev=gaus_input)

        if self.use_reconst:
            self.decode = self.build_net(
                n_latent, n_features, decoder_arch, bn, dropoutrate, self.af
            )
            if input_mean is not None:
                self.decode.add(Activation(input_unnormalize))

        if self.use_labels:
            self.onehot = onehot
            if onehot:
                self.label_net = self.build_net(
                    n_latent, 2, labels_arch, False, 0, self.af, tf.nn.softmax
                )
            else:
                self.label_net = self.build_net(
                    n_latent, 1, labels_arch, False, 0, self.af, tf.nn.sigmoid
                )
        if self.use_pe:
            self.pe_net = self.build_net(
                n_latent, 1, pe_arch, bn, dropoutrate, self.af, gaus=gaus_pe
            )
            if output_mean is not None:
                self.pe_net.add(Activation(output_unnormalize))

    def build_net(
        self,
        n_input,
        n_output,
        arch,
        bn=False,
        dropoutrate=0,
        af=tf.nn.relu,
        lastaf=None,
        gaus=0,
        model=None,
    ):
        """build a multi-layer net

        :param n_input: int, input dimension
        :param n_output: int, output dimension
        :param arch: list of int, number of nodes for each layer
        :param bn: bool, whether to add batch normalization layer
        :param dropoutrate: float, drop out rate for the dropout layer
        :param af: tf function name, activation function
        :param lastaf: tf function name, activation function for the last layer
        :return: keras.models.Sequential, neural net
        """

        nlayer = len(arch)
        if model is None:
            model = keras.models.Sequential()

        if nlayer > 0:

            # first linear layer
            model.add(Dense(arch[0], input_shape=(n_input,), activation=af))
            if bn:
                model.add(BatchNormalization())
            if dropoutrate > 0:
                model.add(Dropout(rate=dropoutrate))

            # middle layer contains three layers (linear, batch_norm, and dropout)
            for layer_idx in range(nlayer - 1):
                model.add(
                    Dense(
                        arch[layer_idx + 1],
                        input_shape=(arch[layer_idx],),
                        activation=af,
                    )
                )
                if bn:
                    model.add(BatchNormalization())
                if dropoutrate > 0:
                    model.add(Dropout(rate=dropoutrate))

            # last layer, no activation function
            if lastaf:
                model.add(
                    Dense(n_output, input_shape=(arch[nlayer - 1],), activation=lastaf)
                )
            else:
                model.add(Dense(n_output, input_shape=(arch[nlayer - 1],)))

        else:
            if lastaf is not None:
                model.add(Dense(n_output, input_shape=(n_input,), activation=lastaf))
            else:
                model.add(Dense(n_output, input_shape=(n_input,)))
        # if (gaus > 0):
        #     model.add(GaussianNoise(stddev=gaus))

        return model

    def predict_pe(self, z):

        return self.pe_net(z)

    def call(self, x, training=False):
        """forward pass through the network. always evaluates reconstruction of coordinates, optionally evaluates models for clas label and pe.

        :param x: torch.tensor, inputs, including potential energies and labels
        :return result, dict of torch.tensors, reconstructed coordinates, optionally predicted energies and class labels"""

        # get latent space representation, used as input for all other models
        if self.gaus_input > 0 and training:
            oldx = self.noise_input(x)
        else:
            oldx = x

        if self.gaus_latent > 0 and training:
            z = self.noise_latent(self.encode(x))
        else:
            z = self.encode(x)
        result = {"latent": z, "oldx": x}

        # variational encoder
        # mean, logvar = tf.split(self.encode(x), num_or_size_splits=2, axis=1)

        # collect results
        if self.use_reconst:
            result["newx"] = self.decode(z)

        if self.use_pe:
            if self.gaus_pe_latent > 0 and training:
                result["pe"] = self.pe_net(self.noise_pe_latent(z))
            else:
                result["pe"] = self.pe_net(z)
            if self.gaus_pe > 0 and training:
                result["pe"] = self.noise_pe(result["pe"])

        if self.use_labels:
            if self.onehot:
                result["label"] = self.label_net(z)
            else:
                result["label"] = self.label_net(z)
        return result
