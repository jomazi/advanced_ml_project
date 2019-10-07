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


# activations
acts = {'relu': F.relu, 'linear': lambda x: x, 'sigmoid': th.sigmoid, 'binary step': lambda x: th.sign(F.relu(x))}


class VGAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim, latent_dim, dropout=0., xavier_init=False,
                 activations=['relu', 'linear', 'sigmoid']):
        super(VGAE, self).__init__()
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