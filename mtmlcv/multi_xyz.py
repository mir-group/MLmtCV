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
from tensorflow.keras.layers import (
    Lambda,
    Activation,
)
import numpy as np

from mtmlcv.xyz import JointAE as original_JAE
from mtmlcv.xyz import Wrap

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

        if self.use_pe and self.use_labels:

            new_model = keras.models.Sequential()
            for layer in self.label_net.layers[:-1]:
                new_model.add(layer)

            if self.onehot:
                new_model.add(Lambda(lambda x: x[:, 0:1]))

            self.pe_net = self.build_net(
                1, 1, pe_arch, False, 0, af=tf.nn.tanh, gaus=gaus_pe,
                model=new_model
            )
            if output_mean is not None:
                self.pe_net.add(Activation(output_unnormalize))

            self.all_nets["pe"] = self.pe_net
