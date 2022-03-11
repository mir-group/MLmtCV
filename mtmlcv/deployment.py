import sys
import numpy as np

from mtmlcv.data_loader import Data_Loader
from mtmlcv.xyz import JointAE
from mtmlcv.plot import Plot

import tensorflow as tf

import yaml

def main():

    timestamp = sys.argv[1]
    with open(f"{timestamp}.yaml") as fin:
        args = yaml.load(fin, Loader=yaml.Loader)


    # set up nn
    net = JointAE(**args)

    sess = tf.keras.backend.get_session()

    net.load_weights(timestamp, sys.argv[2])
    net.save_graph(sess, export_path=f"reload/{timestamp}")

    ## plot latent space
    if (args["use_pe"]):
        def get_pe(z):
            ph_pe = net.pe_net(z, training=False)
            return sess.run(ph_pe)
    else:
        get_pe = None

    if (args["use_labels"]):
        def get_label(z):
            ph_label = net.label_net(z, training=False)
            return sess.run(ph_label)
    else:
        get_label = None

    filename = "../tps_300_damp0.2.npz"
    testname = "../tps_800_damp0.2.npz"
    for name, path in {'300':filename,
                       '800':testname,
                       #'290': "../tps_290_damp0.2.npz",
                       'umb':'../dist_round2.npz'}.items():

        data = Data_Loader(filename=path,
                           shuffle=True,
                           input_label=["xyz"],
                           target_label=["pe", "label"],
                           n_sample=4000,
                           batch_size=4000,
                           weight_by_pe=True,
                           weight_by_label=True)
        sess.run(data.iterator.initializer)
        nextx = data.iterator.get_next()
        ph_pred = net(nextx['x'], training=False)
        data, pred = sess.run((nextx, ph_pred))
        if "pe" in data and name == '300':

            ids = np.where(data["pe_prefactor"]!=0)[0]
            for k in data:
                if len(data[k]) == len(data["pe"]):
                    data[k] = data[k][ids]
            for k in pred:
                if len(pred[k]) == len(data["pe"]):
                    pred[k] = pred[k][ids]
        np.savez(f"{timestamp}_{name}_pred.npz", **pred)

        intc = data["intc"]
        plot = Plot(timestamp+name, data, pred, n_latent=args["n_latent"], intc=intc, save_data=True)
        plot.plot(get_pe, get_label)


if __name__ == '__main__':
    main()
