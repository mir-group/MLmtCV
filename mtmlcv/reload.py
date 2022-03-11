from mtmlcv.plot import plot_result
from mtmlcv.utils import print_info
from mtmlcv.data_loader import Data_Loader
import numpy as np
import tensorflow as tf
import os
import glob
import sys
import joblib

sys.path.insert(0, os.path.abspath("../src"))

timestamp = "lc1000.0-lc_late1000.0"
fout = open("log.reload{}".format(timestamp), "w+")

# load data
filename = "/n/holylfs/LABS/kozinsky_lab/Data/Catalyst/all_sorted_data/5dtoy-tpe+md.npz"
testname = "/n/holylfs/LABS/kozinsky_lab/User/leleslx/ND-EnergyLanscape/for_paper/5d/tpe-validation/tpe-small.npz"
input_label = ["xyz"]
target_label = ["label", "pe"]
batch_size = 1000
n_sample = 300000
shuffle = True
# data loader
data = Data_Loader(
    filename=filename,
    shuffle=shuffle,
    input_label=input_label,
    target_label=target_label,
    n_sample=n_sample,
    batch_size=batch_size,
    weight_by_pe=True,
    weight_by_label=True,
)
test_data = Data_Loader(
    filename=testname,
    shuffle=shuffle,
    input_label=input_label,
    target_label=target_label,
    n_sample=n_sample,
    batch_size=batch_size,
    weight_by_pe=True,
    weight_by_label=True,
    test_only=True,
)
sess = tf.Session()

# load NN
export_dir = "models/" + timestamp
tf.saved_model.loader.load(sess, ["serve"], export_dir)

fout = open("read_graph.info", "w+")
graph = tf.get_default_graph()
for op in graph.get_operations():
    print(op, file=fout)
fout.close()

phX = graph.get_tensor_by_name("X:0")
phz = graph.get_tensor_by_name("Z:0")
phlabel = graph.get_tensor_by_name("label:0")
phpe = graph.get_tensor_by_name("pe:0")

phjab_z = graph.get_tensor_by_name("jab_z:0")
phgrad_pe = graph.get_tensor_by_name("grad_pe:0")
phgrad_label = graph.get_tensor_by_name("grad_label:0")

phz_in = graph.get_tensor_by_name("zonly:0")
ph_z_pe = graph.get_tensor_by_name("predict_pe:0")
ph_z_label = graph.get_tensor_by_name("predict_label:0")

nextx = data.iterator.get_next()
sess.run(data.iterator.initializer)
points = sess.run(nextx)
z, label, pe = sess.run((phz, phlabel, phpe), feed_dict={phX: points["x"]})
pred = {"latent": z, "label": label, "pe": pe}

test_nextx = test_data.iterator.get_next()
sess.run(test_data.iterator.initializer)
tpoints = sess.run(test_nextx)
tz, tlabel, tpe = sess.run((phz, phlabel, phpe), feed_dict={phX: tpoints["x"]})
tpred = {"latent": tz, "label": tlabel, "pe": tpe}


def get_pe(z):
    return sess.run(ph_z_pe, feed_dict={phz_in: z})


def get_label(z):
    return sess.run(ph_z_label, feed_dict={phz_in: z})


plot_result(timestamp, get_pe, get_label, points, pred, tpoints, tpred, filetype="png")
