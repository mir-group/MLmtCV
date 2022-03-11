"""

class for the neural net model

"""

import copy
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
from tensorflow.python.saved_model.builder import SavedModelBuilder
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
import yaml

import logging

from mtmlcv.utils import jacobian


class Identity(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        super(Identity, self).__init__(**kwargs)

    def call(self, x):
        return x


class Sparse(tf.keras.layers.Layer):
    def __init__(self, input_dim, **kwargs):
        self.input_dim = input_dim
        super(Sparse, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            name="kernel", shape=(self.input_dim), initializer="ones", trainable=True
        )
        self.b = self.add_weight(
            name="bias", shape=(self.input_dim), initializer="zeros", trainable=True
        )
        # Be sure to call this at the end
        super(Sparse, self).build(input_shape)

    def call(self, x):
        return tf.add(tf.multiply(x, self.w), self.b)


class JointAE(keras.Model):
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
        input_std=1,
        output_mean=None,
        output_std=1,
        af=tf.nn.elu,
    ):

        super(JointAE, self).__init__()

        self.model_name = copy.deepcopy(model_name)
        self.n_features = n_features
        self.n_latent = n_latent
        self.encoder_arch = encoder_arch
        self.decoder_arch = decoder_arch
        self.use_reconst = use_reconst
        self.use_labels = use_labels
        self.use_dist = use_dist
        self.use_pe = use_pe
        self.dropoutrate = dropoutrate
        self.bn = bn
        self.bnmomentum = bnmomentum
        self.af = af
        self.gaus_latent = gaus_latent
        self.gaus_pe = gaus_pe
        self.gaus_pe_latent = gaus_pe_latent
        self.gaus_input = gaus_input
        self.preprocess = None

        self.reg_vars = None
        self.use_aux = False

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

        self.all_nets = {}

        self.encode = self.build_net(
            n_features, n_latent, encoder_arch, bn, dropoutrate, self.af, model=model
        )
        self.all_nets["encode"] = self.encode

        if gaus_latent is not None:
            self.noise_latent = GaussianNoise(stddev=gaus_latent)

        if gaus_pe is not None:
            self.noise_pe = GaussianNoise(stddev=gaus_pe)

        if gaus_pe_latent is not None:
            self.noise_pe_latent = GaussianNoise(stddev=gaus_pe_latent)

        if gaus_input is not None:
            self.noise_input = GaussianNoise(stddev=gaus_input)

        if self.use_reconst:
            self.decode = self.build_net(
                n_latent, n_features, decoder_arch, bn, dropoutrate, self.af
            )
            if input_mean is not None:
                self.decode.add(Activation(input_unnormalize))
            self.all_nets["decode"] = self.decode

        self.onehot = onehot
        self.nclass = nclass
        if self.use_labels:
            if onehot:
                self.label_net = self.build_net(
                    n_latent, nclass, labels_arch, False, 0, self.af, tf.nn.softmax
                )
            else:
                self.label_net = self.build_net(
                    n_latent, 1, labels_arch, False, 0, self.af, tf.nn.sigmoid
                )
            self.all_nets["label"] = self.label_net

        if self.use_pe:
            self.pe_net = self.build_net(
                n_latent, 1, pe_arch, False, 0, af=tf.nn.tanh, gaus=gaus_pe
            )
            if output_mean is not None:
                self.pe_net.add(Activation(output_unnormalize))

            self.all_nets["pe"] = self.pe_net

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
        kernel_init="glorot_normal",
        bias_init="glorot_normal",
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

        if model is None:
            model = keras.models.Sequential()
        else:
            for layer in model.layers:
                print(f"one predefined layer {repr(layer)}")

        if arch is None:
            model.add(Identity(n_input))
            return model
        else:
            nlayer = len(arch)

        if nlayer > 0:

            # first linear layer
            model.add(
                Dense(
                    arch[0],
                    input_shape=(n_input,),
                    activation=af,
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init,
                )
            )
            # print(f"add input layer {n_input}")
            # if bn:
            #     model.add(BatchNormalization())
            # if dropoutrate > 0:
            #     model.add(Dropout(rate=dropoutrate))

            # middle layer contains three layers (linear, batch_norm, and dropout)
            for layer_idx in range(nlayer - 1):
                model.add(
                    Dense(
                        arch[layer_idx + 1],
                        input_shape=(arch[layer_idx],),
                        activation=af,
                        kernel_initializer=kernel_init,
                        bias_initializer=bias_init,
                    )
                )
                # print(f"add layer {arch[layer_idx+1]}")
                if bn:
                    model.add(
                        BatchNormalization(trainable=True, momentum=self.bnmomentum)
                    )
                    # print(f"add bn")
                if dropoutrate > 0:
                    model.add(Dropout(rate=dropoutrate))
                    # print(f"add dp")

            # last layer, no activation function
            model.add(
                Dense(
                    n_output,
                    input_shape=(arch[nlayer - 1],),
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init,
                )
            )
            if lastaf is not None:
                model.add(
                    Activation(
                        activation = lastaf
                    )
                )
            # print(f"add output layer {n_output}")

        else:
            model.add(
                Dense(
                    n_output,
                    input_shape=(n_input,),
                    kernel_initializer=kernel_init,
                    bias_initializer=bias_init,
                )
            )
            if lastaf is not None:
                model.add(
                    Activation(
                        activation=lastaf,
                    )
                )
            # print(f"add output layer {n_output}")
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
        if self.gaus_input is not None and training:
            inputx = self.noise_input(x, training=training)
        else:
            inputx = x

        if self.preprocess is not None:
            oldx = self.preprocess(inputx, training=training)
        else:
            oldx = tf.identity(inputx)

        z = self.encode(inputx, training=training)

        if self.gaus_latent is not None:
            z = self.noise_latent(z, training=training)

        result = {"latent": z, "oldx": oldx}

        # variational encoder
        # mean, logvar = tf.split(self.encode(x), num_or_size_splits=2, axis=1)

        # collect results
        if self.use_reconst:
            result["newx"] = self.preprocess(
                self.decode(z, training=training), training=training
            )

        if self.use_pe:
            if self.gaus_pe_latent is not None:
                result["pe"] = self.pe_net(
                    self.noise_pe_latent(z, training=training), training=training
                )
            else:
                result["pe"] = self.pe_net(z, training=training)

            if self.gaus_pe is not None:
                result["pe"] = self.noise_pe(result["pe"], training=training)

        if self.use_labels:
            if self.onehot:
                result["label"] = self.label_net(z, training=training)
            else:
                result["label"] = self.label_net(z, training=training)
        return result

    def load_weights(self, checkpoint, epoch):
        try:
            epoch = int(epoch)
        except:
            pass
        if isinstance(epoch, int):
            chkpt = checkpoint + "/{ftype}-{epoch:04d}.ckpt"
        else:
            chkpt = checkpoint + "/{ftype}-{epoch}.ckpt"
        for key in self.all_nets:
            self.all_nets[key].load_weights(chkpt.format(epoch=epoch, ftype=key))

    def save_weights(self, checkpoint, epoch):
        try:
            epoch = int(epoch)
        except:
            pass

        if isinstance(epoch, int):
            chkpt = checkpoint + "/{ftype}-{epoch:04d}.ckpt"
        else:
            chkpt = checkpoint + "/{ftype}-{epoch}.ckpt"
        for key in self.all_nets:
            self.all_nets[key].save_weights(chkpt.format(epoch=epoch, ftype=key))

    def save_graph(self, sess, export_path=None):

        ph_X = tf.keras.Input(shape=(self.n_features,), name="X")
        ph_z = tf.identity(self.encode(ph_X, training=False), name="z_x")
        ph_zonly = tf.keras.Input(shape=(self.n_latent,), name="zonly")
        jab_z = tf.identity(jacobian(ph_z, ph_X), name="dz_dx")

        inputs = {"X": ph_X, "zonly": ph_zonly}
        outputs = {"z_x": ph_z, "dz_dx": jab_z}
        if self.use_reconst:
            ph_newx = tf.identity(self.decode(ph_z, training=False), name="newx_x")
            ph_newx_z = tf.identity(
                self.decode(ph_zonly, training=False), name="newx_z"
            )
            outputs["newx_x"] = ph_newx
            outputs["newx_z"] = ph_newx_z
        if self.use_pe:
            ph_pe = tf.identity(self.pe_net(ph_z, training=False), name="pe_x")
            ph_pe_z = tf.identity(self.pe_net(ph_zonly, training=False), name="pe_z")
            jab_pe = tf.identity(tf.gradients(ph_pe, ph_X), name="dpe_dx")
            jab_pe_z = tf.identity(tf.gradients(ph_pe_z, ph_zonly), name="dpe_dz")
            outputs["pe_x"] = ph_pe
            outputs["pe_z"] = ph_pe_z
            outputs["dpe_dx"] = jab_pe
            outputs["dpe_dz"] = jab_pe_z
        if self.use_labels:
            ph_label = tf.identity(self.label_net(ph_z, training=False), name="label_x")
            ph_label_z = tf.identity(
                self.label_net(ph_zonly, training=False), name="label_z"
            )
            jab_label = tf.identity(tf.gradients(ph_label, ph_X), name="dlabel_dx")
            jab_label_z = tf.identity(
                tf.gradients(ph_label_z, ph_zonly), name="dlabel_dz"
            )
            outputs["label_x"] = ph_label
            outputs["label_z"] = ph_label_z
            outputs["dlabel_dx"] = jab_label
            outputs["dlabel_dz"] = jab_label_z

        # # debug
        # fgout = open("saved_graph.info", "w+", 1)
        # graph = tf.get_default_graph()
        # for op in graph.get_operations():
        #     print(op, file=fgout)
        # fgout.close()

        # save it to the folder
        export_dir = "models/{}".format(self.model_name) if export_path is None else export_path
        builder = SavedModelBuilder(export_dir)
        signature = predict_signature_def(inputs=inputs, outputs=outputs)
        builder.add_meta_graph_and_variables(
            sess=sess, tags=["serve"], signature_def_map={"predict": signature}
        )
        builder.save()
        print(f"! save to {export_dir}")

    def add_encoder_kernels(self):
        if self.reg_vars is None:
            self.reg_vars = []
        for var in self.encode.trainable_variables:
            if "kernel" in var.name:
                self.reg_vars += [tf.reshape(var, [1, -1])]
        self.reg_vars = tf.concat(self.reg_vars, axis=-1)
