import unittest
import numpy as np

import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Softmax
from sklearn.metrics import accuracy_score

from nncv.data_loader import *
from nncv.loss import *

class TestTFFunction(unittest.TestCase):

    _xyz = np.ones([100, 3])
    nfeature = _xyz.shape[1]
    _data = {'x': _xyz}
    data = tf.data.Dataset.from_tensor_slices(_data).batch(49)

    iterator = data.make_one_shot_iterator()

    model = K.models.Sequential([Dense(2,
                                       input_shape=(nfeature,), activation=tf.nn.relu)])

    def test_place_holder(self):

        ph_X = tf.keras.Input(shape=(self.nfeature, ))
        ph_Y = self.model(ph_X)

    def test_iteration(self):

        nextx = self.iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                idb = 0
                while True:
                    x = sess.run(nextx)
                    print("batch", idb, "batch shape", x['x'].shape)
                    idb += 1
            except tf.errors.OutOfRangeError:
                pass

    def test_session(self):
        ''' test how the session and iterator works together with feed dict'''

        ph_X = tf.keras.Input(shape=(self.nfeature, ))
        ph_Y = self.model(ph_X)

        nextx = self.iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                idb = 0
                while True:
                    x = sess.run(nextx)
                    Y = sess.run(ph_Y, feed_dict={ph_X: x['x']})
                    idb += 1
            except tf.errors.OutOfRangeError:
                pass

    def test_feeddict(self):
        ''' test whether the return value can be a dict for the session '''

        def combo(x):
            a = x + 3
            b = x
            return {'a': a, 'b': b}

        ph_X = tf.keras.Input(shape=(self.nfeature, ))
        # ph_Y = self.model(ph_X)
        # ph_dict = {'a': ph_a, 'b': ph_b}
        ph_dict = combo(ph_X)

        nextx = self.iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            try:
                idb = 0
                while True:
                    x = sess.run(nextx)
                    Y = sess.run(ph_dict, feed_dict={ph_X: x['x']})
                    idb += 1
            except tf.errors.OutOfRangeError:
                pass

    def test_optimization(self):

        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.float32(x_train)
        y_train = np.float32(y_train)
        x_test = np.float32(x_test)
        y_test = np.float32(y_test)

        x_train, x_test = x_train / 255.0, x_test / 255.0
        train_batch = tf.data.Dataset.from_tensor_slices({
            'x': x_train, 'y': y_train})
        train_batch = train_batch.batch(32)
        iterator = train_batch.make_initializable_iterator()
        nextx = iterator.get_next()

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

        opt = tf.train.AdamOptimizer(learning_rate=0.001,
                                     beta1=0.9, beta2=0.999, epsilon=1e-08,
                                     use_locking=False, name='Adam')

        def loss(model, data):
            y_pred = model(data['x'])
            y_true = data['y']
            l = tf.keras.backend.sparse_categorical_crossentropy(
                y_true, y_pred)
            g = tf.gradients(l, model.trainable_variables)
            return l, g, y_pred, y_true

        ph_l, ph_g, ph_p, ph_y = loss(model, nextx)
        training_op = opt.minimize(ph_l)

        init = tf.global_variables_initializer()
        sess = tf.keras.backend.get_session()
        sess.run(init)

        for epoch in range(5):
            sess.run(iterator.initializer)
            for batch in range(10):
                _, l, g, p, y = sess.run((training_op, ph_l, ph_g, ph_p, ph_y))
                p_cont = np.argmax(p, axis=-1)
                e = accuracy_score(p_cont, y)
            print("epoch {}, loss {:.4f}, accuracy{:.4f}".format(
                epoch, np.average(l), e))
