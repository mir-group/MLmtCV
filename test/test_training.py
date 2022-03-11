import unittest
import numpy as np
from nncv.data_loader import *
from nncv.loss import Loss
from nncv.joint_model import JointAE

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Softmax


class TestTraining(unittest.TestCase):

    net = JointAE(n_features=2, n_latent=2,
                  dropoutrate=0, bn=False,
                  encoder_arch=[],
                  use_reconst=False, decoder_arch=[],
                  use_labels=True, onehot=True, labels_arch=[],
                  use_pe=True, pe_arch=[],
                  input_mean=None,
                  output_mean=None)

    ph_X = tf.keras.Input(shape=(2, ))
    ph_Z = tf.keras.Input(shape=(2, ))
    ph_label = tf.keras.Input(shape=(1, ))
    ph_pe = tf.keras.Input(shape=(1, ))
    ph_w = tf.keras.Input(shape=(1, ))
    _x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    _label = np.array([0, 1, 0, 0]).reshape([-1, 1])
    _pe = np.array([0.8, 1.5, 0.3, 0.2]).reshape([-1, 1])
    _w = np.ones(4).reshape([-1, 1])
    data = {'x': _x, 'label': _label, 'pe': _pe, 'w':_w}
    dataset = tf.data.Dataset.from_tensor_slices(data)

    def test_compute_loss(self):
        l = Loss(False, True, True)
        iterator = self.dataset.batch(2).make_initializable_iterator()
        nextx = iterator.get_next()
        ph_result = l.joint_loss_function(
                self.net, nextx)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(iterator.initializer)
            result = sess.run(ph_result)
            print(result)

    def test_optimization(self):
        ''' use mnist to test whether the network is functioning works'''

        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.float32(x_train).reshape([-1, 784])
        y_train = np.int64(y_train)
        print("shape", x_train.shape, y_train.shape)
        x_test = np.float32(x_test).reshape([-1, 784])
        y_test = np.int64(y_test)

        x_train, x_test = x_train / 255.0, x_test / 255.0
        train_batch = tf.data.Dataset.from_tensor_slices({
            'x': x_train, 'y': y_train})
        train_batch = train_batch.batch(32)
        iterator = train_batch.make_initializable_iterator()
        nextx = iterator.get_next()

        model = JointAE(n_features=784, n_latent=10,
                        encoder_arch=[512], decoder_arch=[],
                        dropoutrate=0.2, bn=False, onehot=True, nclass=10,
                        input_mean=None, output_mean=None,
                        use_reconst=False, use_labels=True, use_pe=False)

        opt = tf.train.AdamOptimizer(learning_rate=0.001,
                                     beta1=0.9, beta2=0.999, epsilon=1e-08,
                                     use_locking=False, name='Adam')

        sca = tf.keras.metrics.SparseCategoricalAccuracy()
        scc = tf.keras.losses.SparseCategoricalCrossentropy()

        def loss(model, data):
            y_pred = model(data['x'])['label']
            y_true = data['y']
            l = scc(y_true, y_pred)
            g = tf.gradients(l, model.trainable_variables)
            e = sca(y_true, y_pred)
            return l, g, y_pred, y_true, e

        ph_l, ph_g, ph_p, ph_y, ph_e = loss(model, nextx)
        training_op = opt.minimize(ph_l)

        init = tf.global_variables_initializer()
        sess = tf.keras.backend.get_session()
        sess.run(init)

        for epoch in range(5):
            sess.run(iterator.initializer)
            for batch in range(50):
                _, l, g, pred, true, e = sess.run(
                    (training_op, ph_l, ph_g, ph_p, ph_y, ph_e))
            print("label", epoch, "Accuracy", e)

    def test_mse(self):
        ''' use mnist to test whether the network is functioning works'''

        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.float32(x_train).reshape([-1, 784])
        y_train = np.int32(y_train)
        x_test = np.float32(x_test).reshape([-1, 784])
        y_test = np.int32(y_test)

        x_train, x_test = x_train / 255.0, x_test / 255.0
        train_batch = tf.data.Dataset.from_tensor_slices({
            'x': x_train, 'y': y_train})
        train_batch = train_batch.batch(32)
        iterator = train_batch.make_initializable_iterator()
        nextx = iterator.get_next()

        model = JointAE(n_features=784, n_latent=2,
                        encoder_arch=[512],
                        use_reconst=False, decoder_arch=[],
                        use_labels=False, onehot=True, nclass=10,
                        use_pe=True, pe_arch=[512],
                        dropoutrate=0.2, bn=False,
                        input_mean=None, output_mean=None)

        opt = tf.train.AdamOptimizer(learning_rate=0.001,
                                     beta1=0.9, beta2=0.999, epsilon=1e-08,
                                     use_locking=False, name='Adam')

        def loss(model, data):
            y_pred = model(data['x'])['pe']
            y_true = data['y']
            l = tf.keras.losses.MSE(y_true, y_pred)
            e = tf.keras.losses.MAE(y_true, y_pred)
            g = tf.gradients(l, model.trainable_variables)
            return l, g, y_pred, y_true, e

        ph_l, ph_g, ph_p, ph_y, ph_e = loss(model, nextx)
        training_op = opt.minimize(ph_l)

        init = tf.global_variables_initializer()
        sess = tf.keras.backend.get_session()
        sess.run(init)

        for epoch in range(5):
            sess.run(iterator.initializer)
            for batch in range(50):
                _, l, g, p, y, e = sess.run(
                    (training_op, ph_l, ph_g, ph_p, ph_y, ph_e))
            print(epoch, "MAE", np.average(e))

    def test_freeze(self):
        ''' test whether the encoder and label_net can be train separately '''


        opt = tf.train.AdamOptimizer(learning_rate=0.001,
                                     beta1=0.9, beta2=0.999, epsilon=1e-08,
                                     use_locking=False, name='Adam')
        opt2 = tf.train.AdamOptimizer(learning_rate=0.001,
                                      beta1=0.9, beta2=0.999, epsilon=1e-08,
                                      use_locking=False, name='Adam')

        sca = tf.keras.metrics.SparseCategoricalAccuracy()
        scc = tf.keras.losses.SparseCategoricalCrossentropy()
        mse = tf.keras.losses.MSE

        def loss(model, x, label, pe):
            result = model(x)
            y_label = result['label']
            y_pe = result['pe']
            l = scc(label, y_label)+mse(pe, y_pe)
            return l, y_label, y_pe

        ph_l, ph_lab, ph_p = loss(self.net, self.ph_X, self.ph_label, self.ph_pe)
        training_op = opt.minimize(ph_l, var_list=self.net.encode.trainable_variables)
        training_op2 = opt2.minimize(ph_l, var_list=self.net.label_net.trainable_variables)

        init = tf.global_variables_initializer()
        sess = tf.keras.backend.get_session()
        sess.run(init)

        initial_label = []
        for var in self.net.label_net.trainable_variables:
            el = sess.run(var)
            initial_label += [np.array(el.flat)]
        initial_label = np.hstack(initial_label)

        _, l, lab, p = sess.run((training_op, ph_l, ph_lab, ph_p),
                feed_dict={self.ph_X:self._x,
                    self.ph_label:self._label,
                    self.ph_pe:self._pe})

        second_label = []
        second_encode = []
        for var in self.net.label_net.trainable_variables:
            el = sess.run(var)
            second_label += [np.array(el.flat)]
        for var in self.net.encode.trainable_variables:
            el = sess.run(var)
            second_encode += [np.array(el.flat)]
        second_label = np.hstack(second_label)
        second_encode = np.hstack(second_encode)

        diff = np.sum(np.abs(initial_label-second_label))
        self.assertEqual(diff, 0)

        _, l, lab, p = sess.run((training_op2, ph_l, ph_lab, ph_p),
                feed_dict={self.ph_X:self._x,
                    self.ph_label:self._label,
                    self.ph_pe:self._pe})

        third_encode = []
        for var in self.net.encode.trainable_variables:
            el = sess.run(var)
            third_encode += [np.array(el.flat)]
        third_encode = np.hstack(third_encode)
        diff = np.sum(np.abs(third_encode-second_encode))
        self.assertEqual(diff, 0)


