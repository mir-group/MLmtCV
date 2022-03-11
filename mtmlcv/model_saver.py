from sklearn import manifold

"""
functions to save and load neural network model

TO DO:
    save neural network model
      strip the label and potential energy prediction part
      utomatically add jacobian
      add name for the output and jacobian
      type the PLUMED line arguments
    load neural network model
"""
import numpy as np
import copy

# from mtmlcv.separate_pe_models import JointAE

import matplotlib.pyplot as plt

plt.switch_backend("agg")


def load_net(net, modelfile):
    """
    load NN parameters from a previous model

    :param data: string, filename where the model was saved
    :return: JointAE object
    """

    pass

    # state_dict = torch.load(modelfile)

    # for k in ["log_freq", "patience", "lr", "bn", "dropoutrate",
    #         "n_latent", "n_features", "shuffle", "n_sample",
    #         "num_epochs", "batch_size", "target_label", "input_label",
    #         "filename", "use_pe_loss", "use_label_loss",
    #         "use_reconst_loss", "pe_arch", "labels_arch",
    #         "decoder_arch", "encoder_arch", "timestamp"]:
    #     if (k not in state_dict.keys()):
    #         raise AssertionError("the key {} is not stored in the file {}".format(k, modelfile))

    # # network parameter
    # encoder_arch = state_dict['encoder_arch']
    # decoder_arch = state_dict['decoder_arch']
    # n_features = state_dict['n_features']
    # n_latent = state_dict['n_latent']
    # bn = state_dict['bn']
    # dropoutrate = state_dict['dropoutrate']

    # use_reconst_loss = state_dict['use_reconst_loss']
    # use_label_loss = state_dict['use_label_loss']
    # use_pe_loss = state_dict['use_pe_loss']
    # labels_arch = state_dict['labels_arch']
    # pe_arch = state_dict['pe_arch']
    # joint_arch = state_dict['joint_arch']

    # filename = state_dict['filename']
    # input_label = state_dict['input_label']
    # target_label = state_dict['target_label']
    # batch_size = state_dict['batch_size']
    # num_epochs = state_dict['num_epochs']
    # n_sample = state_dict['n_sample']
    # shuffle = state_dict['shuffle']

    # # optimizer parameters
    # lr = state_dict['lr']
    # patience = state_dict['patience']

    # # output parameters
    # log_freq = state_dict['log_freq']

    # for k in ["log_freq", "patience", "lr", "bn", "dropoutrate",
    #         "n_latent", "n_features", "shuffle", "n_sample",
    #         "num_epochs", "batch_size", "target_label", "input_label",
    #         "filename", "use_pe_loss", "use_label_loss",
    #         "use_reconst_loss", "pe_arch", "labels_arch",
    #         "decoder_arch", "encoder_arch", "timestamp", "ae_init_weights", "joint_arch"]:
    #     state_dict.pop(k, None)

    # net.load_state_dict(state_dict, strict=False)

    # return None


def create_net_from_file(modelfile):
    """
    load NN parameters from a previous model

    :param data: string, filename where the model was saved
    :return: JointAE object
    """

    pass

    # state_dict = torch.load(modelfile)

    # for k in ["log_freq", "patience", "lr", "bn", "dropoutrate",
    #         "n_latent", "n_features", "shuffle", "n_sample",
    #         "num_epochs", "batch_size", "target_label", "input_label",
    #         "filename", "use_pe_loss", "use_label_loss",
    #         "use_reconst_loss", "pe_arch", "labels_arch",
    #         "decoder_arch", "encoder_arch", "timestamp"]:
    #     if (k not in state_dict.keys()):
    #         raise AssertionError("the key {} is not stored in the file {}".format(k, modelfile))

    # # network parameter
    # encoder_arch = state_dict['encoder_arch']
    # decoder_arch = state_dict['decoder_arch']
    # n_features = state_dict['n_features']
    # n_latent = state_dict['n_latent']
    # bn = state_dict['bn']
    # dropoutrate = state_dict['dropoutrate']

    # use_reconst_loss = state_dict['use_reconst_loss']
    # use_label_loss = state_dict['use_label_loss']
    # use_pe_loss = state_dict['use_pe_loss']
    # labels_arch = state_dict['labels_arch']
    # pe_arch = state_dict['pe_arch']
    # # joint_arch = state_dict['joint_arch']

    # filename = state_dict['filename']
    # input_label = state_dict['input_label']
    # target_label = state_dict['target_label']
    # batch_size = state_dict['batch_size']
    # num_epochs = state_dict['num_epochs']
    # n_sample = state_dict['n_sample']
    # shuffle = state_dict['shuffle']

    # # optimizer parameters
    # lr = state_dict['lr']
    # patience = state_dict['patience']

    # # output parameters
    # log_freq = state_dict['log_freq']

    # # set up nn

    # net = JointAE(n_features=n_features,
    #               n_latent=n_latent,
    #               encoder_arch=encoder_arch,
    #               decoder_arch=decoder_arch,
    #               dropoutrate=dropoutrate,
    #               bn=bn,
    #               use_reconst_loss=use_reconst_loss,
    #               use_labels=use_label_loss,
    #               use_pe=use_pe_loss,
    #               labels_arch=labels_arch,
    #               pe_arch=pe_arch)

    # for k in ["log_freq", "patience", "lr", "bn", "dropoutrate",
    #         "n_latent", "n_features", "shuffle", "n_sample",
    #         "num_epochs", "batch_size", "target_label", "input_label",
    #         "filename", "use_pe_loss", "use_label_loss",
    #         "use_reconst_loss", "pe_arch", "labels_arch",
    #         "decoder_arch", "encoder_arch", "timestamp", "ae_init_weights", "joint_arch"]:
    #     state_dict.pop(k, None)

    # net.load_state_dict(state_dict)

    # return net


def load_net_data(modelfile):
    """
    load NN parameters from a previous model

    :param data: string, filename where the model was saved
    :return: JointAE object, and the loader for training and test data
    """

    pass
    # state_dict = torch.load(modelfile)
    # filename = state_dict['filename']
    # input_label = state_dict['input_label']
    # target_label = state_dict['target_label']
    # batch_size = state_dict['batch_size']
    # num_epochs = state_dict['num_epochs']
    # n_sample = state_dict['n_sample']
    # shuffle = state_dict['shuffle']

    # # net = load_net(modelfile)
    # net = create_net_from_file(modelfile)

    # input_dim, output_dim, output_type, training_set, target_set, train_loader, test_loader= load_data(filename=filename,
    #                                                                                                    shuffle=shuffle,
    #                                                                                                    input_label=input_label,
    #                                                                                                    target_label=target_label,
    #                                                                                                    n_sample=n_sample,
    #                                                                                                    batch_size=batch_size)
    # return net, train_loader, test_loader
