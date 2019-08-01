# Credits: https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html#sphx-glr-tutorials-models-1-gnn-1-gcn-py

"""
ATTENTION:

To be able to run the code, you have to manually extract the Cora dataset into the data folder and rename it as 'cora'.
The dataset can be downloaded following this link:

https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/cora_raw.zip

Additionally the CoraDataset class in ../advanced_ml_project/venv/lib/python3.7/site-packages/dgl/data/citation_graph.py
has to be overwritten. The init function has to be replaced as it follows:

def __init__(self):
    self.dir = os.path.dirname(os.path.abspath(__file__)).split('venv')[0] + 'data'
    self._load()

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl import DGLGraph
from dgl.data import citation_graph as citegrh

from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from sklearn.cluster import KMeans

from warnings import simplefilter

########################################################################################################################
# CONFIG

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# determine directory of this script
cur_dir = os.path.dirname(os.path.abspath(__file__))

########################################################################################################################
# MODEL

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 16, F.relu)
        self.gcn2 = GCN(16, 7, F.relu)

    def forward(self, g, features):
        x_hidden = self.gcn1(g, features)
        x = self.gcn2(g, x_hidden)
        return x, x_hidden


net = Net()
print('Network architecture: \n', net)


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


########################################################################################################################
# TRAIN

"""
features: 1433
nodes: 2708
classes: 7
mask: defines of which nodes labels are used for training
"""

# init
g, features, labels, mask = load_cora_data()
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
n_epochs = 300

# details about train/test split
num_train = list(mask.numpy()).count(1)
print("Part of train data: {}%".format(int((num_train/2708)*100)))

# store info
hidden_activations = []
pred_labels = []
loss = []

# train
with tqdm(range(n_epochs), unit='epoch') as t:
    for epoch in t:
        logits, hidden = net(g, features)
        logp = F.log_softmax(logits, 1)
        loss_train = F.nll_loss(logp[mask], labels[mask])
        loss_test = F.nll_loss(logp[~mask], labels[~mask])

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # append activations and predicted labels
        hidden_activations.append(hidden.detach().numpy())
        pred_labels.append(np.argmax(logits.detach().numpy(), axis=1))

        # update description and store test loss
        t.set_description("Train Loss {:.4f} | Test Loss {:.4f}".format(loss_train.item(), loss_test.item()))
        loss.append(loss_test.item())


########################################################################################################################
# VISUALIZATION

# loss over epochs
plt.figure(figsize=(8, 6))

# plot data
plt.title('GCN test loss over epochs \n', size=12, weight='bold')
plt.text(0.78, 0.875, 'dataset: Cora \noptimizer: Adam \ninitial lr: 1e-3 \nloss: NLL \ntest part: 5% ',
         verticalalignment='center',
         transform=plt.gca().transAxes,
         bbox=dict(boxstyle="round,pad=0.6", edgecolor='black', facecolor='white', alpha=0.5))
plt.plot(np.arange(n_epochs), loss)
plt.xlabel('epoch')
plt.ylabel('test loss')
plt.savefig(os.path.join(cur_dir, './plots/gcn_loss.png'))

# clean up
plt.close('all')

# hidden layer activations of best model, along with predicted labels
index_best_model = np.argmin(loss)
pred_labels_best = pred_labels[index_best_model]
hidden_activations_best = hidden_activations[index_best_model]

# t-SNE embedding; trying different hyper-parameters
parameters = {'perplexity': [20, 35], 'learning_rate': [50, 125, 200],
              'metric': ['euclidean', 'cityblock', 'cosine', 'correlation']}


# helper functions
def bench_k_means(estimator, name):
    # metrics
    metrs = [metrics.homogeneity_score, metrics.completeness_score, metrics.v_measure_score,
             metrics.adjusted_rand_score, metrics.adjusted_mutual_info_score]
    # scores
    scores = [m(pred_labels_best, estimator.labels_) for m in metrs]

    return np.mean(scores)


def activations_vis(data, tsne_param, path):
    # t-SNE
    embedded_activations = TSNE(n_components=2, **tsne_param).fit_transform(data)

    # K-Means clustering
    kmeans = KMeans(init='k-means++', n_clusters=7, n_init=10).fit(embedded_activations)

    # clustering scores
    score = bench_k_means(kmeans, name="t-SNE")

    # plot init
    plt.figure(figsize=(6, 6))

    # plot data
    plt.title('t-SNE embedding of hidden layer activations \n', size=12, weight='bold')
    plt.text(0.04, 0.895, 'learning rate: {} \nmetric: {} \nperplexity: {} \nmean clustering score: {:.2f}'
             .format(*tsne_param.values(), score), verticalalignment='center', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round,pad=0.6", edgecolor='black', facecolor='white', alpha=0.5))
    plt.scatter(x=embedded_activations[:, 0], y=embedded_activations[:, 1], c=pred_labels_best, alpha=0.25)
    plt.xticks(())
    plt.yticks(())
    plt.savefig(os.path.join(cur_dir, path))

    # clean up
    plt.close('all')


# actual visualization for different t-SNE hyper-parameters
for param in tqdm(ParameterGrid(parameters), desc='t-SNE hyper-parameter search'):
    activations_vis(hidden_activations_best, param, './plots/gcn_tsne/{}_{}_{}.png'.format(*param.values()))
