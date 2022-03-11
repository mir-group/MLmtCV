import unittest

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Softmax

from nncv.data_loader import *
from nncv.loss import *
from nncv.joint_model import JointAE


class TestModel(unittest.TestCase):

    net = JointAE(n_features=2, n_latent=2,
                  dropoutrate=0, bn=False,
                  encoder_arch=[],
                  use_reconst=False, decoder_arch=[],
                  use_labels=True, onehot=False, labels_arch=[],
                  use_pe=True, pe_arch=[],
                  input_mean=None,
                  output_mean=None)

    ph_X = tf.keras.Input(shape=(2, ))
    ph_Z = tf.keras.Input(shape=(2, ))
    ph_label = tf.keras.Input(shape=(1, ))
    ph_pe = tf.keras.Input(shape=(1, ))
    ph_w = tf.keras.Input(shape=(1, ))

    def test_build_net(self):
        ''' the label net directly connecting the latent space to
        output. so the weight should be 2 by 1 and the bias should
        be 1 element.'''
        net = self.net.label_net
        w = net.trainable_variables
        self.assertEqual(w[0].shape[0], 2)
        self.assertEqual(w[0].shape[1], 1)
        self.assertEqual(w[1].shape[0], 1)

    def test_pe_net(self):
        model = JointAE(n_features=784, n_latent=2,
                        encoder_arch=[512],
                        use_reconst=False, decoder_arch=[],
                        use_labels=False, onehot=True, nclass=10,
                        use_pe=True, pe_arch=[512],
                        dropoutrate=0.2, bn=False,
                        input_mean=None, output_mean=None)

    def test_label_net(self):
        model = JointAE(n_features=784, n_latent=2,
                        encoder_arch=[512],
                        use_reconst=False, decoder_arch=[],
                        use_labels=True, labels_arch=[512], onehot=True, nclass=10,
                        use_pe=True, pe_arch=[],
                        dropoutrate=0.2, bn=False,
                        input_mean=None, output_mean=None)

    def test_call(self):
        ph_result = self.net.call(self.ph_X)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            net = self.net.label_net
            net.set_weights([np.array([[-1], [1]]), np.array([0.5])])
            net = self.net.encode
            net.set_weights([np.array([[1, 0], [0, 1]]),
                             np.array([0.0, 0.0])])
            net = self.net.pe_net
            net.set_weights([np.array([[1], [0]]), np.array([-0.5])])

            x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
            result = sess.run(ph_result,
                              feed_dict={self.ph_X: x})

            zz = result['latent']
            rr, cc = np.meshgrid(range(4), range(2))
            self.assertTrue(any(zz[i, j] == x[i, j]
                                for (i, j) in zip(rr.flat, cc.flat)))

            l = result['label']
            ref = tf.nn.sigmoid([0.5, 1.5, -0.5, 0.5]).eval()

            pe = result['pe']
            pe0 = [0.5, 1.5, 0.5, 1.5]
            pe1 = [-0.5, -0.5, 0.5, 0.5]
            self.assertTrue(any(pe[i] == pe0[i]*ref[i]+(1-ref[i])*pe1[i]
                                for i in range(4)))
            self.assertTrue(any(l[i][0] == ref[i] for i in range(4)))

    def test_savegraph(self):
        ''' use mnist to test whether the network is functioning works'''

        model = JointAE(model_name="test_savegraph", n_features=784, n_latent=10,
                        encoder_arch=[512], decoder_arch=[],
                        dropoutrate=0.2, bn=False, onehot=True, nclass=10,
                        input_mean=None, output_mean=None,
                        use_reconst=False, use_labels=True, use_pe=False)

        init = tf.global_variables_initializer()
        sess = tf.keras.backend.get_session()
        sess.run(init)

        model.save_graph(sess)
