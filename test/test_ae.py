from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.python import saved_model
import tensorflow as tf
from nncv.plot import plot_result
from nncv.loss import Loss
from nncv.joint_model import JointAE
from nncv.data_loader import Data_Loader
from nncv.utils import print_info, jacobian
from time import gmtime, strftime

import unittest

class TestAE(unittest.TestCase):

    def test_ae(self):
        # networks
        encoder_arch = [100, 50]
        decoder_arch = [20, 50]
        labels_arch = []
        pe_arch = [10, 10]
        gaus_pe = 0.05
        gaus_latent = 0.0

        # training parameters
        use_reconst_loss = False
        use_label_loss = True
        use_pe_loss = True
        timestamp = "5d_1d_weighted-gaus-pe"
        rc = 1.0  # initial weighting
        lc = 1.0
        pc = 0.10
        lc_late = 1.0
        pc_late = 0.10
        switch_loss = 100
        num_epochs = 5

        filename = "test/test_files/5dtoy-tpe+md-small.npz"
        testname = "test/test_files/5dtoy-tpe+md-small.npz"

        input_label = ['xyz']
        n_features = 5

        target_label = ['label', 'pe']
        batch_size = 10
        n_sample = 300
        shuffle = True
        early_pe = 0.015
        early_stop = False
        weight_by_pe = True
        weight_by_label = True

        # network parameter
        n_latent = 1
        dropoutrate = 0.5
        bn = False

        # optimizer parameters
        lr = 0.0005
        patience = 5
        factor = 0.1

        # output parameters
        log_freq = 10

        # # dump all parameters
        fout = open("log-label-{}.{}".format(n_latent, timestamp), "w+")
        tmp = dict(locals())
        all_parameters = {}
        for (k, v) in tmp.items():
            if (isinstance(v, (str, int, float, list))):
                print(k, v, file=fout)
                all_parameters[k] = v

        # data loader
        data = Data_Loader(filename=filename,
                           shuffle=shuffle,
                           input_label=input_label,
                           target_label=target_label,
                           n_sample=n_sample,
                           batch_size=batch_size,
                           weight_by_pe=weight_by_pe,
                           weight_by_label=weight_by_label)
        test_data = Data_Loader(filename=testname,
                                shuffle=shuffle,
                                input_label=input_label,
                                target_label=target_label,
                                n_sample=n_sample,
                                batch_size=batch_size,
                                weight_by_pe=weight_by_pe,
                                weight_by_label=weight_by_label,
                                test_only=True)

        # set up nn
        net = JointAE(n_features=n_features, n_latent=n_latent,
                      encoder_arch=encoder_arch,
                      decoder_arch=decoder_arch,
                      dropoutrate=dropoutrate, bn=bn, onehot=False,
                      use_reconst=use_reconst_loss,
                      use_labels=use_label_loss, use_pe=use_pe_loss,
                      labels_arch=labels_arch, pe_arch=pe_arch,
                      gaus_latent=gaus_latent, gaus_pe=gaus_pe,
                      input_mean=data.input_mean,
                      input_std=data.input_std,
                      output_mean=data.output_mean,
                      output_std=data.output_std)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9,
                                           beta2=0.999, epsilon=1e-08,
                                           use_locking=False,
                                           name='Adam')

        # define loss function
        reconst_coeff = tf.placeholder("float32", None)
        pe_coeff = tf.placeholder("float32", None)
        label_coeff = tf.placeholder("float32", None)
        l = Loss(reconst_coeff=reconst_coeff,
                 pe_coeff=pe_coeff, label_coeff=label_coeff)

        # get place holder for training and loss
        nextx = data.iterator.get_next()
        ph_loss, ph_details, ph_pred, ph_data = l.joint_loss_function(net, nextx)
        training_op = optimizer.minimize(ph_loss)

        # get place holder for test and mae metrics
        next_testx = test_data.iterator.get_next()
        ph_tloss, ph_tdetails, ph_tpred, ph_tdata = l.joint_loss_function(
            net, next_testx)
        ph_metrics = l.compute_MAE(ph_tpred, ph_tdata)

        # TO DO: dynamically update learning rate
        # (test this vs exponential decay, sometimes works better)
        # loss_var = tf.Variable(10.0, name='total_loss',
        #        trainable=False, dtype=tf.float32)
        # scheduler = ReduceLROnPlateau(monitor='total_loss', factor=factor,
        #                           patience=pariance, min_lr=0.001)

        init = tf.global_variables_initializer()

        # open a session

        # session_conf = tf.ConfigProto(
        #               intra_op_parallelism_threads=2,
        #               inter_op_parallelism_threads=2)
        # sess = tf.Session(config=session_conf)

        # with tf.Session() as sess:

        sess = tf.keras.backend.get_session()

        sess.run(init)

        # count the number of batches
        total_batch = 0
        sess.run(data.iterator.initializer)
        try:
            while True:
                sess.run(nextx)
                total_batch += 1
        except tf.errors.OutOfRangeError:
            pass

        idrun = 0
        stop = False
        for epoch in range(num_epochs):
            if (stop):
                continue
            sess.run(data.iterator.initializer)
            for batch in range(total_batch):
                if (stop):
                    continue
                _, loss, details, pred, data_true = sess.run((
                    training_op,
                    ph_loss, ph_details, ph_pred, ph_data),
                    feed_dict={reconst_coeff: rc,
                               pe_coeff: pc, label_coeff: lc})
                if (batch+1) % log_freq == 0:
                    allstring = 'Epoch [{}/{}]'.format(
                        epoch+1, num_epochs)
                    allstring += ', Step [{}/{}]'.format(
                        batch+1, total_batch)
                    sess.run(test_data.iterator.initializer)
                    tloss, tdetails, metrics = sess.run((ph_tloss, ph_tdetails, ph_metrics),
                                                        feed_dict={reconst_coeff: rc,
                                                                   pe_coeff: pc, label_coeff: lc})
                    print_info(allstring, tloss, tdetails, metrics, fout)
                    if ('mae_pe' in metrics.keys()):
                        if (metrics['mae_pe'] < early_pe and early_stop):
                            stop = True
            if (epoch > switch_loss):
                lc = lc_late
                pc = pc_late
        #     scheduler.step(loss)

        # # sometimes the optimizer will leave a bunch of uninit variables
        # # initialize the uninit variables to avoid error while saving
        # # the model
        # uninitialized_vars = []
        # for var in tf.global_variables():
        #     try:
        #         sess.run(var)
        #     except tf.errors.FailedPreconditionError:
        #         uninitialized_vars.append(var)
        #         print("find one uninit variables", var)
        # init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        # sess.run(init_new_vars_op)

        # save the network
        net.save_graph(sess)

        # # plot results
        sess.run(data.iterator.initializer)
        sess.run(test_data.iterator.initializer)
        pred, train_data = sess.run((ph_pred, ph_data))
        tpred, test_data = sess.run((ph_tpred, ph_tdata))

        def get_pe(z):
            ph_pe = net.predict_pe(z)
            return sess.run(ph_pe)

        def get_label(z):
            ph_label = net.label_net(z)
            return sess.run(ph_label)
        plot_result(timestamp, get_pe, get_label,
                    train_data, pred, test_data, tpred)

        fout.close()
