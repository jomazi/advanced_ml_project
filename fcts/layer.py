import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation, xavier_init=False):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation
        if xavier_init:
            self.param_init()

    def param_init(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class GCN(nn.Module):
    """Graph Convolutional Network"""

    def __init__(self, in_feats, out_feats, activation, dropout=0., xavier_init=False):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation, xavier_init=xavier_init)
        self.dropout = dropout

    def forward(self, g, feature):
        feature = F.dropout(feature, self.dropout, self.training)
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class InnerProductDecoder(nn.Module):
    """Decoder using inner product for prediction."""

    def __init__(self, activation, dropout=0.):
        super(InnerProductDecoder, self).__init__()
        self.activation = activation
        self.dropout = dropout

    def forward(self, z):
        z = F.dropout(z, self.dropout, self.training)
        adj = self.activation(th.mm(z, z.t()))
        return adj