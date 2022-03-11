from copy import deepcopy
import tensorflow as tf

class EpochTrainer:
    def __init__(
        self, name, iterator, net, Loss, loss_coeffs=None, coeffs={}, log_freq=100, training=True
    ):

        self.name = name
        self.iterator = iterator
        self.log_freq = log_freq

        self.ph = {}

        # define loss function
        if loss_coeffs is None:
            loss_coeffs = Loss.loss_terms

        _coeffs = deepcopy(coeffs)
        for k in loss_coeffs:
            _coeffs[k] = tf.placeholder("float32", None)
            self.ph[k] = _coeffs[k]
        self.loss_func = Loss(net, _coeffs)

        # get place holder for training and loss
        self.nextx = self.iterator.get_next()
        (
            self.ph_loss,
            self.ph_details,
            self.ph_pred,
            self.ph_data,
        ) = self.loss_func.joint_loss_function(net, self.nextx)
        self.ph_metrics = self.loss_func.compute_MAE(self.ph_pred, self.ph_data)

        # set up the optimizer
        self.ph["lr"] = tf.placeholder("float32", None)
        self.opt = tf.train.AdamOptimizer(
            learning_rate=self.ph["lr"],
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            use_locking=False,
            name="Adam",
        )
        self.training_op = self.opt.minimize(self.ph_loss)
        self.gbu = False

    def count_batch(self, sess):

        self.nbatches = 0
        sess.run(self.iterator.initializer)
        try:
            while True:
                sess.run(self.nextx)
                self.nbatches += 1
        except tf.errors.OutOfRangeError:
            pass

    def __call__(self, epoch, sess, para_dict, fout, training=False):

        feed_dict = {self.ph[k]: para_dict[k] for k in self.ph}

        sess.run(self.iterator.initializer)
        for batch in range(self.nbatches):

            if training:
                metrics, _, loss, details, pred, data = sess.run(
                    (
                        self.ph_metrics,
                        self.training_op,
                        self.ph_loss,
                        self.ph_details,
                        self.ph_pred,
                        self.ph_data,
                    ),
                    feed_dict=feed_dict,
                )
            else:
                metrics, loss, pred, data, details = sess.run(
                    (
                        self.ph_metrics,
                        self.ph_loss,
                        self.ph_pred,
                        self.ph_data,
                        self.ph_details,
                    ),
                    feed_dict=feed_dict,
                )


            if (batch + 1) % self.log_freq == 0 or (batch + 1) == self.nbatches:
                allstring = f"{epoch+1:5d} {self.name:>5s} {batch+1:5d}"
                s = self.loss_func.tostr(allstring, loss, details, metrics, para_dict)
                print(s)
                print(s, file=fout)
                if " nan " in s:
                    self.gbu = True


        return pred, data

