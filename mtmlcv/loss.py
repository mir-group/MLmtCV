from time import gmtime, strftime

import numpy as np

import tensorflow as tf
import tensorflow.keras as K


class Scalar2Onehot(tf.keras.layers.Layer):
    def __init__(self, nclass):
        super(Scalar2Onehot, self).__init__()
        self.nclass = nclass

    def call(self, x):  # , training=False):
        vec = tf.zeros_like(x)
        fl = tf.math.maximum(tf.math.floor(x), vec + 1)
        cl = tf.math.minimum(tf.math.ceil(x), vec + nclass)
        frac = x - fl
        i1 = tf.one_hot(tf.cast(fl, tf.int32), self.nclass)
        i2 = tf.one_hot(tf.cast(cl, tf.int32), self.nclass)
        return i1 * frac + i2 * (1 - frac)


class Loss:

    loss_terms = ["reconst", "dist", "reg", "pe", "label", "constraint"]

    def __init__(self, model, coeffs):

        self.latent_scale = coeffs.pop("latent_scale", 1)
        self.pe_scale = coeffs.pop("pe_scale", 0.000625)

        self.onehot = model.onehot
        if model.onehot:
            self.ce = K.losses.SparseCategoricalCrossentropy()
            self.acc = K.metrics.SparseCategoricalAccuracy()
        else:
            if model.nclass == 2:
                self.ce = K.losses.BinaryCrossentropy()
                self.acc = K.metrics.BinaryAccuracy()
            else:
                self.ce = K.losses.SparseCategoricalCrossentropy()
                self.acc = K.metrics.SparseCategoricalAccuracy()

        self.mse = K.losses.MeanSquaredError()
        self.mae = K.losses.MeanAbsoluteError()

        for key in self.loss_terms:
            setattr(self, key + "_coeff", 0)
            setattr(self, key, False)

        for key in coeffs:
            value = coeffs[key]
            if key in self.loss_terms:
                setattr(self, key + "_coeff", value)
                if isinstance(value, float):
                    if value > 0:
                        setattr(self, key, True)
                    else:
                        setattr(self, key, False)
                else:
                    setattr(self, key, True)
            else:
                setattr(self, key, value)

        self.reconst = model.use_reconst and self.reconst
        self.label = model.use_labels and self.label
        self.pe = model.use_pe and self.pe
        self.dist = model.use_dist and self.dist
        self.constraint = model.use_aux and self.constraint
        self.reg = (model.reg_vars is not None) and self.reg

        if self.label:
            if (not model.onehot) and model.nclass > 2:
                self.scalar2onehot = Scalar2Onehot(model.nclass)

        for key in self.loss_terms:
            switch = getattr(self, key)
            coeff = getattr(self, key + "_coeff")
            print("Loss function with", key, switch, coeff)

    def joint_loss_function(self, model, data, training=True):
        """Joint Autoencoder loss, consisting of reconstruction loss and
        optionally label and PE loss"""

        predictions = model.call(data["x"], training=training)

        w = data["w"]
        loss = 0
        details = {}
        for key in self.loss_terms:
            switch = getattr(self, key)
            func = getattr(self, key + "_loss")
            coeff = getattr(self, key + "_coeff")
            if switch:
                l = func(model, predictions, data, w)
                details[key + "_p"] = l
                details[key] = coeff * l
                loss += details[key]

        return loss, details, predictions, data

    def reconst_loss(self, model, predictions, data, w):
        return self.mse(predictions["oldx"], predictions["newx"], sample_weight=w)

    def constraint_loss(self, model, predictions, data, w):
        mu = predictions["mu"]
        # sig = predictions['sigma']
        # return 0.5 * tf.reduce_sum(tf.reduce_sum(tf.square(mu*sig), axis=-1)*w)
        return 0.5 * tf.reduce_sum(tf.reduce_sum(tf.square(mu), axis=-1) * w)

    def label_loss(self, model, predictions, data, w):
        if (not model.onehot) and model.nclass > 2:
            return self.ce(self.scalar2onehot(data["label"]), predictions["label"]) #, sample_weight=w)
        return self.ce(data["label"], predictions["label"]) #, sample_weight=w)

    def pe_loss(self, model, predictions, data, w):
        return self.mse(data["pe"], predictions["pe"], sample_weight=w*data["pe_prefactor"])

    def dist_loss(self, model, predictions, data, w):

        weight = w*data["pe_prefactor"]
        matric_w = tf.matmul(weight, tf.transpose(weight))

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
        F_p = tf.sigmoid(D_p / self.pe_scale)
        predictions["F_l"] = F_l
        predictions["F_p"] = F_p
        difference = tf.subtract(F_l, F_p)
        difference = tf.square(difference)

        if "label" in data.keys():
            v = tf.broadcast_to(data["label"], tf.shape(D_p))
            mask = tf.dtypes.cast(tf.equal(v, tf.transpose(v)), dtype=tf.float32)
            mask = mask - tf.eye(tf.shape(D_p)[0])
            mask = tf.stop_gradient(mask)
            mask = mask * matric_w
            predictions["mask"] = mask
            difference = tf.multiply(difference, mask)

        else:
            mask = -tf.eye(tf.shape(D_p)[0]) + 1
            mask = mask * matric_w

        return tf.reduce_sum(difference) / tf.reduce_sum(mask)

    def reg_loss(self, model, predictions, data, w):
        return tf.reduce_sum(tf.keras.backend.abs(model.reg_vars))

    def compute_MAE(self, pred, target):
        """ Joint Autoencoder loss, consisting of reconstruction loss and optionally label and PE loss """


        metrics = {}

        w = target["w"]

        if self.reconst:
            metrics["mae_reconst"] = self.mae(pred["oldx"], pred["newx"], sample_weight=w)

        if self.label:
            n = pred["label"].shape[-1]
            if self.onehot:
                metrics["clf"] = self.acc(target["label"], pred["label"], sample_weight=w)
            else:
                metrics["clf"] = self.acc(target["label"], pred["label"], sample_weight=w)

        if self.pe:
            metrics["mae_pe"] = self.mae(target["pe"], pred["pe"], sample_weight=w*target["pe_prefactor"])

        return metrics

    def tostr(self, headstring, loss, details, metrics, parameters={}):

        s = f"{headstring}"
        for (k, v) in parameters.items():
            s += f" {k} {v:8.2g}"
        s += "\n"

        s += f"{headstring} loss: {loss:7.2e}\n"
        s += f"{headstring} "
        s1 = ""
        error_list = set(details.keys()) - set(
            ["grad", "prediction", "mae_pe", "mae_reconst", "clf"]
        )
        for k in error_list:
            if k[-2:] != "_p":
                s += f" {k}: {details[k]:.4f}"
                value = details[k + "_p"]
                s1 += f" {k}: {value:7.2e}"
        s += "\n"
        s += f"{headstring} w.o c {s1}\n"

        if self.reconst:
            # coordinates mae
            s += "{:35s} Coordinates MAE: {:7.2e}".format(
                headstring, metrics["mae_reconst"]
            )
            s += "\n"

        if self.label:
            # coordinates mae
            s += "{} Classification Accuracy: {}".format(headstring, metrics["clf"])
            s += "\n"

        if "mae_pe" in metrics.keys():
            # coordinates mae
            s += "{}, Energies MAE: {:7.2e}".format(headstring, metrics["mae_pe"])
            s += "\n"
        s += "\n"
        return s
