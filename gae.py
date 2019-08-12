import os
import numpy as np
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

from fcts.layer import GCN, InnerProductDecoder
from fcts.optimizer import loss_function
from fcts.clustering import bench_k_means


########################################################################################################################
# CONFIG

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# determine directory of this script
cur_dir = os.path.dirname(os.path.abspath(__file__))

# training
n_epochs = 150
n_routines = 5

# activations
acts = {'relu': F.relu, 'linear': lambda x: x, 'sigmoid': th.sigmoid, 'binary step': lambda x: th.sign(F.relu(x))}


########################################################################################################################
# MODEL

class Net(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim, latent_dim, dropout=0., xavier_init=False,
                 activations=['relu', 'linear', 'sigmoid']):
        super(Net, self).__init__()
        global acts
        self.gcn1 = GCN(input_feat_dim, hidden_dim, acts[activations[0]], dropout=dropout, xavier_init=xavier_init)
        self.gcn2 = GCN(hidden_dim, latent_dim, acts[activations[1]], dropout=dropout, xavier_init=xavier_init)
        self.gcn3 = GCN(hidden_dim, latent_dim, acts[activations[1]], dropout=dropout, xavier_init=xavier_init)
        self.dc = InnerProductDecoder(activation=acts[activations[2]], dropout=dropout)

    def encode(self, g, features):
        hidden = self.gcn1(g, features)
        return self.gcn2(g, hidden), self.gcn3(g, hidden)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = th.exp(logvar)
            eps = th.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, g, features):
        mu, logvar = self.encode(g, features)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


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
features: 1433
nodes: 2708
classes: 7
mask: defines of which nodes labels are used for training
"""

# init
g, features, labels, _ = load_cora_data()

# adjacency matrix
adj = g.adjacency_matrix().to_dense()

########################################################################################################################
# VISUALIZATION


def latent_vis(mu, path, params):
    # K-Means clustering
    kmeans = KMeans(init='k-means++', n_clusters=7, n_init=10).fit(mu)

    # clustering scores
    score = bench_k_means(labels, kmeans)

    # plot init
    plt.figure(figsize=(6, 6))

    # plot data
    plt.title('GAE latent space \n', size=12, weight='bold')
    plt.text(0.04, 0.84, 'dropout: {} \nxavier init: {} \nfeatures used: {} \nactivation 1: {} \nactivation 2: {} '
                         '\nactivation 3: {} \nmean clustering score: {:.2f}'.format(*params, score),
             verticalalignment='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.6", edgecolor='black', facecolor='white', alpha=0.5))
    plt.scatter(x=mu[:, 0], y=mu[:, 1], c=labels, alpha=0.25)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(os.path.join(cur_dir, path))

    # clean up
    plt.close('all')


########################################################################################################################
# TRAIN

# train function
def train(net, optim, use_features):
    # store info
    loss_total = []
    acc_total = []
    mu_total = []

    for _ in tqdm(range(n_routines), desc='Training', unit='routine', leave=False):

        with tqdm(range(n_epochs), unit='epoch', leave=False) as t:
            # store info
            loss_ = []
            acc_ = []

            for _ in t:
                # decide if features are used
                if use_features:
                    features_ = features
                else:
                    features_ = th.zeros_like(features)

                recovered, mu, logvar = net(g, features_)
                loss = loss_function(recon_x=recovered, x=adj, mu=mu, log_var=logvar)

                # get predictions in 1/0 format
                pred = th.round(recovered).detach().numpy()

                optim.zero_grad()
                loss.backward()
                optim.step()

                # measures
                acc = metrics.accuracy_score(adj.numpy().flatten(), pred.flatten())  # accuracy
                loss = loss.detach().numpy()

                # append info
                acc_.append(acc)
                loss_.append(loss)

                # update description and store test loss
                t.set_description("Loss {:.4f} | Accuracy {:.2f}".format(loss, acc))

            # save info
            loss_total.append(loss_)
            acc_total.append(acc_)
            # get and save mu of latent space
            net.eval()
            _, mu, _ = net(g, features)
            mu = mu.detach().numpy()
            mu_total.append(mu)

    return np.mean(np.array(loss_total), axis=0), np.mean(np.array(acc_total), axis=0), \
           np.mean(np.array(mu_total), axis=0)


# try different hyper-parameter for training
parameters = {'dropout': [0., 0.25], 'xavier_init': [False, True],
              'act': [['relu', 'linear', 'binary step'], ['relu', 'linear', 'sigmoid']],
              'use_features': [False, True]}

# store best parameters
best_min_loss = 1000
best_param = []
best_loss = np.array([])
best_acc = np.array([])

# actual hyper-parameter search
for param in tqdm(ParameterGrid(parameters), desc='hyper-parameter search'):
    # init
    model = Net(1433, 16, 2, dropout=param['dropout'], xavier_init=param['xavier_init'], activations=param['act'])
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
    # parameters as strings
    act = param['act']
    del param['act']
    param_list = list(param.values())
    [param_list.append(a) for a in act]
    # train
    loss_av, acc_av, mu_av = train(model, optimizer, param['use_features'])
    # update best model
    if np.min(loss_av) < best_min_loss:
        best_min_loss = np.min(loss_av)
        best_param = param_list
        best_loss = loss_av
        best_acc = acc_av
    # visualize
    latent_vis(mu_av, './plots/gae/{}_{}_{}_{}_{}_{}.png'.format(*param_list), param_list)


########################################################################################################################
# VISUALIZATION

# loss over epochs
plt.figure(figsize=(8, 6))

# plot data
plt.title('GAE loss over epochs \n', size=12, weight='bold')
plt.text(0.728, 0.7875, 'dataset: Cora \noptimizer: Adam \ninitial lr: 1e-3 \nloss: KL + BCE \ndropout: {} '
                        '\nxavier init: {} \nfeatures used: {} \nactivation 1: {} \nactivation 2: {} '
                        '\nactivation 3: {} '.format(*best_param),
         verticalalignment='center',
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.6", edgecolor='black', facecolor='white', alpha=0.5))
plt.plot(np.arange(n_epochs), best_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(os.path.join(cur_dir, './plots/gae_loss.png'))

# clean up
plt.close('all')

# accuracy over epochs
plt.figure(figsize=(8, 6))

# plot data
plt.title('GAE accuracy over epochs \n', size=12, weight='bold')
plt.text(0.735, 0.7875, 'dataset: Cora \noptimizer: Adam \ninitial lr: 1e-3 \nloss: KL + BCE \ndropout: {} '
                        '\nxavier init: {} \nfeatures used: {} \nactivation 1: {} \nactivation 2: {} '
                        '\nactivation 3: {}'.format(*best_param),
         verticalalignment='center',
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.6", edgecolor='black', facecolor='white', alpha=0.5))
plt.plot(np.arange(n_epochs), best_acc)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig(os.path.join(cur_dir, './plots/gae_acc.png'))

# clean up
plt.close('all')