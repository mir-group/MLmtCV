"""
functions to handle data loading

TO DO:
    data augmentation for xyz coordinates
    data augmentation by adding gaussian noise
"""

import numpy as np
import copy

import tensorflow as tf

from collections import Counter
from numpy.linalg import norm


class Data_Loader:

    input_mean = None
    input_std = None
    output_mean = None
    output_std = None

    datatype = np.float32

    def __init__(
        self,
        filename="data.npz",
        shuffle=True,
        input_label=["xyz"],
        target_label=["pe"],
        n_sample=10000,
        test_sample=1000,
        batch_size=100,
        num_epoch=10,
        weight_by_pe=True,
        weight_by_label=True,
        ngrid=100,
        test_only=False,
        input_norm=True,
        output_norm=True,
    ):
        """load the data from npz file
        :param filename: str, root dir of .npy data
        :param shuffle, boolean,
                        whether or not to shuffle training data
        :param input_label: list of str that define the input
        :param target_label: list of str that define the output
        :param n_sample: int, number of training samples to use
        """

        # load data from data.npz
        data = dict(np.load(filename))

        for k in input_label + target_label:
            data[k] = data[k].astype(self.datatype)

        n_config = data[input_label[0]].shape[0]
        # if n_sample is too big
        if (n_sample+test_sample) > n_config:
            if test_sample < n_config:
                n_sample = n_config - test_sample
            else:
                n_sample = n_config //2
                test_sample = n_config - n_sample

        # shuffle data and target with the same permutation
        if shuffle:
            r = np.random.permutation(n_config)
        else:
            r = np.arange(n_config)

        for k in data.keys():
            np.take(data[k], r, axis=0, out=data[k])
            data[k] = data[k][:n_sample+test_sample]

        if "label" in data.keys():
            minlabel = np.min(data["label"])
            if minlabel != 0:
                print(f"WARNING, the data label will be shifted {minlabel}")
            data["label"] = np.int32(data["label"] - minlabel)
            print("type of labels", Counter(data["label"]))
        if "pe" in data.keys():
            values = data["pe"][np.where(data["pe"]!=np.inf)[0]]
            print("potential energy", np.min(values), np.max(values))

        # assemble the input
        if len(input_label) > 1:
            x = []
            for label_id in input_label:
                # # cheating for Alanine dipeptide
                # if (label_id == "xyz" and data[label_id].shape[1]==66):
                #     x += [np.vstack(data[label_id])[:, [3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 30, 31, 32, 42, 43, 44, 45, 46, 47, 48, 49, 50, 54, 55, 56]]]
                if label_id == "colvar" and ("colvar" not in data.keys()):
                    x += [np.vstack(data["intc"])[:, [1, 2]]]
                else:
                    x += [np.vstack(data[label_id])]
                print("x added", label_id)
            x = np.hstack(x)
        else:
            # # cheating for Alanine dipeptide
            # if (input_label[0] == "xyz" and data[input_label[0]].shape[1]==66):
            #     x = np.vstack(data[input_label[0]])[:, [3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 25, 26, 30, 31, 32, 42, 43, 44, 45, 46, 47, 48, 49, 50, 54, 55, 56]]
            if input_label[0] == "colvar" and ("colvar" not in data.keys()):
                x = np.vstack(data["intc"])[:, [1, 2]]
            else:
                x = np.vstack(data[input_label[0]])

        # prepare the normalization mean and std for input
        if (self.input_mean is None) and (input_norm is True):
            xtemp, input_mean, input_std = normalize_data_bound(x, with_inf=False)
            if x.shape[1] == 66:
                newx = rotation(x)
                xtemp, input_mean, input_std = normalize_data_bound(newx, with_inf=False)
                del newx
            self.input_mean = input_mean.astype(self.datatype)
            self.input_std = input_std.astype(self.datatype)
            del xtemp

        # assemble all data
        alldata = {"x": x}
        # for debug. TO DO: remove or add a flag for this
        alldata["xyz"] = data["xyz"]
        if "intc" in data.keys():
            alldata["intc"] = data["intc"]

        # assemble the output
        for label_id in target_label:
            ori_data = data[label_id]
            if len(ori_data.shape) == 1:
                ori_data = np.hstack(ori_data)
            else:
                ori_data = np.vstack(ori_data)
            alldata[label_id] = ori_data
        for k in alldata:
            if len(alldata[k].shape) == 1:
                alldata[k] = alldata[k].reshape([-1, 1])
            print("record added", k, "shape", alldata[k].shape)

        # prepare the normalization mean and std for output
        if (
            "pe" in target_label
            and (self.output_mean is None)
            and (output_norm is True)
        ):
            petemp, output_mean, output_std = normalize_data_bound(data["pe"], with_inf=True)
            print("!!!!", output_mean, output_std)
            self.output_mean = output_mean.astype(self.datatype)
            self.output_std = output_std.astype(self.datatype)
            del petemp

        self.input_dim = x.shape[1]
        self.output_dim = 0


        if "label" in data.keys() and weight_by_label is True:
            l = data["label"]
        else:
            l = None
        if "pe" in data.keys() and weight_by_pe is True:
            p = data["pe"]
        else:
            p = None
        weight = define_weight(label=l, pe=p, ngrid=ngrid)

        avg = np.average(weight)

        weight = self.datatype(weight / avg)

        if "pe" in alldata:
            alldata["pe_prefactor"] = np.array(alldata["pe"] != np.inf, dtype=np.float32)
            ids = np.where(alldata["pe"] == np.inf)[0]
            alldata["pe"][ids] = 0

        self.alldata = alldata
        self.alldata["w"] = weight.reshape([-1, 1])
        self.n_sample = n_sample
        self.test_sample = test_sample
        self.batch_size = batch_size
        self.num_epoch = num_epoch

        self.test_only = test_only

        self.to_tfdataset()

    def to_tfdataset(self):

        alldata = self.alldata
        n_sample = self.n_sample
        test_sample = self.test_sample

        if not self.test_only:
            train_dict = {}
            test_dict = {}
            for k in alldata.keys():
                train_dict[k] = alldata[k][:n_sample]
                test_dict[k] = alldata[k][n_sample : n_sample + test_sample]

            del self.alldata

            self.train_dict = train_dict
            self.test_dict = test_dict
            self.train_dataset = tf.data.Dataset.from_tensor_slices(train_dict)
            self.train_dataset = self.train_dataset.batch(self.batch_size)
            self.test_dataset = tf.data.Dataset.from_tensor_slices(test_dict)
            self.test_dataset = self.test_dataset.batch(self.batch_size * 10)
            self.iterator = self.train_dataset.make_initializable_iterator()
            self.test_iterator = self.test_dataset.make_initializable_iterator()
        else:
            datadict = {}
            for k in alldata.keys():
                datadict[k] = alldata[k]
                datadict[k] = datadict[k][:test_sample]

            del self.alldata

            self.datadict = datadict
            self.dataset = tf.data.Dataset.from_tensor_slices(datadict)
            self.dataset = self.dataset.batch(self.batch_size)
            self.iterator = self.dataset.make_initializable_iterator()
            self.test_iterator = None


def normalize_data_std(data):
    """
    Normalize data such that mean is 0 and std is 1

    :param data: np.ndarray, shape [nsample, nfeature]
    :return: np.ndarray, normalized data
    :return: float, mean value
    :return: float, std value
    """

    normalized_data = copy.deepcopy(data)
    ids = np.where(normalized_data!=np.inf)[0]
    normalized_data = normalized_data[ids]
    if len(data.shape) == 1:
        mean = np.mean(data)
        std = np.std(data) + np.finfo(float).eps
    else:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + np.finfo(float).eps

    return (normalized_data - mean) / std, mean, std


def normalize_data_bound(data, with_inf = True):
    """
    Normalize data such that data ranges from 0 to 1

    :param data: np.ndarray, shape [nsample, nfeature]
    :return: np.ndarray, normalized data
    :return: float, median of the data
    :return: float, range of the data
    """

    values = copy.deepcopy(data)
    if with_inf:
        ids = np.where(values!=np.inf)[0]
        values = values[ids]

    if len(values.shape) == 1:
        xmin = np.min(values)
        xmax = np.max(values)
    else:
        xmin = np.min(values, axis=0)
        xmax = np.max(values, axis=0)
    mean = (xmin + xmax) * 0.5
    std = xmax - xmin + np.finfo(float).eps
    return (values - mean) / std, mean, std


def define_weight(label=None, pe=None, ngrid=100):
    """
    give weight to sample based on the label and potential energies

    :param label: 1D np.array, labels of the data
    :param pe: 1D np.array,    potential energies of the data, has to be float
    :return: np.ndarray,       product of the two weights
    """
    if label is not None:

        label_weight, tlabel = weight_by_label(label)
        nsample = len(label)

        if pe is not None:

            nclass = len(tlabel)
            label_dict = {t: i for (i, t) in enumerate(tlabel)}
            print("label_dict", label_dict)

            # sort out the pe that belong to each class
            sorted_pe = [[] for i in range(nclass)]
            count_w_label = np.zeros(nclass)
            new_pe_id = np.zeros(nsample)
            for idx in range(nsample):
                label_id = label_dict[label[idx]]
                sorted_pe[label_id] += [pe[idx]]
                new_pe_id[idx] = count_w_label[label_id]
                count_w_label[label_id] += 1
            new_pe_id = list(map(int, new_pe_id))

            sorted_weight = [[] for i in range(nclass)]
            for idc in range(nclass):
                sorted_weight[idc] = weight_by_hist(sorted_pe[idc], ngrid)

            pe_weight = np.zeros(nsample)
            for idx in range(nsample):
                label_id = label_dict[label[idx]]
                pe_weight[idx] = sorted_weight[label_id][new_pe_id[idx]]
        else:
            pe_weight = np.ones(nsample)

    else:

        label_weight = np.ones(n_sample)
        if pe is not None:
            pe_weight = weight_by_hist(pe, ngrid)
        else:
            pe_weight = np.ones(nsample)

    return label_weight * pe_weight


def weight_by_label(label):
    """
    give weight to sample based on the counts of the label. the label can be anything that is accepted by collection.Counters. can be string or int or float

    :param label: 1D np.array, value to construct historgram
    :return: np.ndarray, weight
    :return: dict, dict of unique class
    """
    unique_label = Counter(label.reshape([-1]))
    # print("count label", unique_label)
    weight = dict(unique_label)
    for t in unique_label.keys():
        weight[t] = 1.0 / float(unique_label[t])
    label_weight = np.array([weight[t] for t in label])
    return label_weight, unique_label.keys()


def weight_by_hist(values, ngrid):
    """
    give weight to sample based on the inverse of values histogram

    :param values: 1D np.array, value to construct historgram
    :param ngrid:  number of bins for the hisgotram
    :return: np.ndarray, weight
    """

    # get
    ori_n = len(values)
    values = np.array(values)
    print(values[:10])
    ids = np.where(np.array(values) != np.inf )[0]
    if len(ids) == 0:
        return np.ones(ori_n)/ori_n
    if len(ids) < ori_n:
        values = values[ids]

    # build an energy histogram
    vmin = np.min(values)
    vmax = np.max(values)
    vmax_id = np.argmax(values)
    dv = (vmax - vmin) / float(ngrid)

    values_id = map(int, np.floor((values - vmin) / dv))
    values_id = list(values_id)
    values_id[vmax_id] -= 1
    count = Counter(values_id)
    # print("dv", vmax, dv, count.keys())
    # for i in range(len(values_id)):
    #     print(i, values_id[i], values[i])
    # print(count)

    weight = dict(count)
    for t in count.keys():
        weight[t] = 1.0 / float(count[t])
    non_zero_weights = np.array([weight[t] for t in values_id])

    if len(ids) != ori_n:
        hist_weights = np.ones(ori_n) / (ori_n - len(ids))
        hist_weights[ids] = non_zero_weights
        hist_weights = hist_weights / (ngrid+1)
    else:
        hist_weights = non_zero_weights
        hist_weights = hist_weights / ngrid

    print(hist_weights[:10])

    return hist_weights


def rotation(ori_x):
    x = ori_x.reshape([-1, 22, 3])
    x0 = x[:, 8, :].reshape([-1, 1, 3])
    x0 = np.matmul(np.ones((x.shape[1], 1)), x0)

    newx = x - x0

    dr = newx[:, 10, :]
    r_tc = norm(dr, axis=-1)
    r_tcxy = norm(dr[:, :2], axis=-1)
    ct = dr[:, 2] / r_tc
    st = np.sqrt(1 - ct ** 2)
    cp = dr[:, 0] / r_tcxy
    sp = dr[:, 1] / r_tcxy
    r1 = np.stack([ct * cp, -sp, st * cp], axis=-1)
    r2 = np.stack([ct * sp, cp, st * sp], axis=-1)
    r3 = np.stack([-st, ct - ct, ct], axis=-1)
    rotation1 = np.stack([r1, r2, r3], axis=-2)
    newx = np.matmul(newx, rotation1)

    dr = newx[:, 6, :]
    r_lcxy = norm(dr[:, :2], axis=-1)
    cp = dr[:, 0] / r_lcxy
    sp = dr[:, 1] / r_lcxy
    r1 = np.stack([cp, sp, sp - sp])
    r2 = np.stack([-sp, cp, sp - sp])
    r3 = np.stack([sp - sp, sp - sp, sp - sp + 1])
    rotation2 = np.transpose(np.stack([r1, r2, r3], axis=0))
    newx = np.matmul(newx, rotation2)

    return newx.reshape([-1, 66])
