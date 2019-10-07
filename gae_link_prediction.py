import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl import DGLGraph
from dgl.data import citation_graph as citegrh

from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from sklearn.cluster import KMeans

from tqdm import tqdm

from warnings import simplefilter

from fcts.layer import GCN, InnerProductDecoder, VGAE
from fcts.optimizer import loss_function
from fcts.clustering import bench_k_means

from preprocessing import *


########################################################################################################################
# CONFIG

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# determine directory of this script
cur_dir = os.path.dirname(os.path.abspath(__file__))

# training
n_epochs = 200
vis_steps = 40.

hidden_dim = 32
latent_dim = 16
lr = 1e-2

########################################################################################################################
# DATA

def load_cora_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    mask = th.ByteTensor(data.train_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(g.selfloop_edges())
    g = DGLGraph(g)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask


"""
CORA
features: 1433
nodes: 2708
classes: 7
mask: defines of which nodes labels are used for training
"""

# init
g, features, labels, train_mask = load_cora_data()
in_dim = features.size()[1]

# adjacency matrix
adj = g.adjacency_matrix().to_dense()

########################################################################################################################
# PREPROCESSING

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj

# random split in train, test and validation set
adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train
adj += th.eye(adj.size()[0])

# weights for BCE loss
weight_mask = adj_train.view(-1) == 1
weight_tensor = th.ones(weight_mask.size(0))
weight_tensor[weight_mask] = float(adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()

# update adjacency matrix in graph g
g.from_scipy_sparse_matrix(sp.coo_matrix(adj))

########################################################################################################################
# TRAIN


# train function
def train(net, optim, features, param):
    # store info
    loss_total = []
    acc_total = []
    mu_total = []

    use_features = param['use_features']
    with tqdm(range(n_epochs), desc="Training", unit='epoch', leave=False) as t:
        net.train()
        # store info
        loss_ = []
        val_roc = []
        val_ap = []

        for epoch in t:
            # decide if features are used
            if use_features:
                features_ = features
            else:
                features_ = th.zeros_like(features)

            # forward
            recovered, mu, logvar = net(g, features_)

            # decide if weights are used in loss
            loss = loss_function(recon_x=recovered, x=adj, mu=mu, log_var=logvar,
                                 factors=True, weight=weight_tensor)

            optim.zero_grad()
            loss.backward()
            optim.step()

            # update description and store test loss
            loss_.append(loss.detach().numpy())
            t.set_description("Loss {:.4f}".format(loss_[-1]))

            # get scores on validation set
            if (epoch+1)%(int(n_epochs/vis_steps)) == 0:
                roc, ap = get_scores(val_edges, val_edges_false, recovered)
                val_roc.append(roc)
                val_ap.append(ap)

        # save info
        loss_total.append(loss_)
        mu_total.append(mu.detach().numpy())

        test_scores = get_scores(test_edges, test_edges_false, recovered)

    return np.array(loss_total), np.array(acc_total), \
           np.array(mu_total), np.array([val_roc, val_ap]), test_scores


def get_scores(edges_pos, edges_neg, adj_rec):

    # Predict on test set of edges
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]].item())
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]].item())
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = metrics.roc_auc_score(labels_all, preds_all)
    ap_score = metrics.average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

########################################################################################################################
# VISUALIZE


def score_vis(val_roc, val_ap, param):

    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.xlabel('epoch')
    plt.ylabel('roc score')
    plt.text(0.04, 0.84, 'dropout: {} \nxavier init: {} \nactivations: {} '
                         '\nfeatures used: {} \ndim: {} \nlr: {}'
             .format(param['dropout'], param['xavier_init'], param['act'],
                     param['use_features'], param['dim'], param['lr']),
             verticalalignment='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.6", edgecolor='black', facecolor='white', alpha=0.5))
    for i in range(len(val_roc)):
        # plot
        plt.ylim(0.4, 0.9)
        plt.plot(np.arange(val_roc[i].shape[0]) * n_epochs / vis_steps, val_roc[i], label='roc {}'.format(i))
        plt.legend(loc='lower right')
    plt.subplot(122)
    plt.xlabel('epoch')
    plt.ylabel('average precision')
    for i in range(len(val_ap)):
        plt.ylim(0.4, 0.9)
        plt.plot(np.arange(val_ap[i].shape[0]) * n_epochs / vis_steps, val_ap[i], label='ap {}'.format(i))
        plt.legend(loc='lower right')
    # save
    path = '{}-dropout_{}-xavier_{}-{}-{}-act_{}-feat_{}-{}-dim_{}-lr'.format(
        param['dropout'], param['xavier_init'], *param['act'],
        param['use_features'], *param['dim'], param['lr']
    )
    plt.savefig(os.path.join(cur_dir, './plots/link_prediction/{}.png'.format(path)))
    # clean up
    plt.close('all')


def latent_vis(mu, param):
    # K-Means clustering
    kmeans = KMeans(init='k-means++', n_clusters=7, n_init=10).fit(mu)

    # clustering scores
    score = bench_k_means(labels, kmeans)

    # plot init
    plt.figure(figsize=(6, 6))

    # plot data
    plt.title('GAE latent space \n', size=12, weight='bold')
    plt.text(0.04, 0.84, 'dropout: {} \nxavier init: {} \nactivations: {} \nfeatures used: {} '
                         '\ndim: {} \nlr: {} \nmean clustering score: {:.2f}'
             .format(param['dropout'], param['xavier_init'], param['act'], param['use_features'],
                     param['dim'], param['lr'], score),
             verticalalignment='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.6", edgecolor='black', facecolor='white', alpha=0.5))
    plt.scatter(x=mu[:, 0], y=mu[:, 1], c=labels, alpha=0.25)
    # save
    path = '{}-dropout_{}-xavier_{}-{}-{}-act_{}-feat_{}-{}-dim_{}-lr'.format(
        param['dropout'], param['xavier_init'], *param['act'],
        param['use_features'], *param['dim'], param['lr']
    )
    plt.savefig(os.path.join(cur_dir, './plots/link_prediction/latent_vis/{}.png'.format(path)))

    # clean up
    plt.close('all')

########################################################################################################################
# HYPER-PARAMETER


def basic_hyperparameter_search(n_routines):
    # hyper-parameter dictionary
    parameters = {'dropout': [0., 0.25], 'xavier_init': [False, True],
                  'act': [['relu', 'linear', 'sigmoid']],
                  'use_features': [False, True],
                  'dim': [[16, 2], [32, 16]],
                  'lr': [1e-2, 1e-3]
                  }

    # actual hyper-parameter search
    for param in tqdm(ParameterGrid(parameters), desc='hyper-parameter search'):

        # score lists
        val_roc = []
        val_ap = []

        # train
        for _ in tqdm(range(n_routines), desc='Routines', unit='routine', leave=False):
            # init
            model = VGAE(features.size()[1], param['dim'][0], param['dim'][1], dropout=param['dropout'],
                         xavier_init=param['xavier_init'], activations=param['act'])
            optimizer = th.optim.Adam(model.parameters(), lr=param['lr'])
            # train
            _, _, mus, val_scores, _ = train(model, optimizer, features, param)
            # store scores
            val_roc.append(val_scores[0])
            val_ap.append(val_scores[1])

        # visualize
        score_vis(val_roc, val_ap, param)
        if param['dim'][1] == 2:
            latent_vis(mus[-1], param)


def feature_setup_evaluation(n_routines):
    # different feature setups
    features_w_train_labels = th.cat((features, F.one_hot(labels * train_mask.long()).float()), dim=1)  # add train labels
    features_w_all_labels = th.cat((features, F.one_hot(labels).float()), dim=1)  # add all labels
    features_wo_labels = th.cat((features, F.one_hot(labels * 0).float()), dim=1)  # add 0s

    features_dict = {'simple': features, 'train labels': features_w_train_labels,
                     'all labels': features_w_all_labels, 'added zeros': features_wo_labels}

    # basic parameters
    param = {'dropout': 0., 'xavier_init': False, 'act': ['relu', 'linear', 'sigmoid'],
             'use_features': True, 'dim': [32, 16], 'lr': 1e-2}

    # train on different feature sets
    val_ap_dict = {}
    test_ap_dict = {}
    for feature_key in tqdm(features_dict, desc='Different Features', unit='features', leave=False):

        features_ = features_dict[feature_key]
        val_ap_dict[feature_key] = []
        test_ap_dict[feature_key] = []

        for _ in tqdm(range(n_routines), desc='routine', leave=False):
            # init
            model = VGAE(features_.size()[1], param['dim'][0], param['dim'][1], dropout=param['dropout'],
                         xavier_init=param['xavier_init'], activations=param['act'])
            optimizer = th.optim.Adam(model.parameters(), lr=param['lr'])
            # train
            loss_av, _, _, val_scores, test_scores = train(model, optimizer, features_, param)
            val_ap_dict[feature_key].append(val_scores[1])
            test_ap_dict[feature_key].append(test_scores[1])


    # visualize validation scores over epochs
    plt.figure(figsize=(10,6))
    plt.text(0.04, 0.84, 'dropout: {} \nxavier init: {} \nactivations: {} '
                         '\nfeatures used: {} \ndim: {} \nlr: {}'
             .format(param['dropout'], param['xavier_init'], param['act'],
                     param['use_features'], param['dim'], param['lr']),
             verticalalignment='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.6", edgecolor='black', facecolor='white', alpha=0.5))
    for feature_key in features_dict.keys():
        print(feature_key)
        ap_mean = np.array(val_ap_dict[feature_key]).mean(axis=0)
        plt.plot(np.arange(ap_mean.shape[0]) * n_epochs / vis_steps, ap_mean, label=feature_key)
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('average precision')
    # save
    path = 'vgae_feature_setup_comparison'
    plt.savefig(os.path.join(cur_dir, './plots/link_prediction/{}.png'.format(path)))
    # clean up
    plt.close('all')

    # print test scores
    for feature_key in features_dict:
        print('test ap using {} setup: {:.1f} +- {:.2f}'.format(feature_key, 100 * np.mean(test_ap_dict[feature_key]),
                                                                100 * np.var(test_ap_dict[feature_key])))


basic_hyperparameter_search(3)
feature_setup_evaluation(3)
