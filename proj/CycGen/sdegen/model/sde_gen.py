import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse

import sys
sys.path.append('/mnt/c/Official/WCODE/proj/CycGen')
from sdegen import utils, layers
from sdegen.utils import GaussianFourierProjection
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 定义模型
class TransformerFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, num_heads=8):
        super(TransformerFeatureExtractor, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.position_encoding = PositionalEncoding(hidden_size)
        encoder_layers = TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 256)  # 输出维度为128

    def forward(self, x):
        # 输入形状为 [batch_size, seq_length, input_size]
        x = self.embedding(x)  # [batch_size, seq_length, hidden_size]
        x = x + self.position_encoding(x)  # 加上位置编码
        x = x.transpose(0, 1)  # 将序列长度移到第一维
        x = self.transformer_encoder(x)  # [seq_length, batch_size, hidden_size]
        x = x.transpose(0, 1)  # 将序列长度移到第二维
        x = self.fc(x)  # [batch_size, seq_length, 128]
        return x


def repeat_sequence(sequence, num_repeats):
    return torch.cat([sequence[:, i].unsqueeze(1).repeat(1, num_repeats, 1) for i in range(1, sequence.shape[1])], dim=1)


class SDE(torch.nn.Module):

    def __init__(self, config):
        super(SDE, self).__init__()
        self.config = config
        self.anneal_power = self.config.train.anneal_power
        self.hidden_dim = self.config.model.hidden_dim
        self.order = self.config.model.order
        self.noise_type = self.config.model.noise_type

        self.feature_extraction = TransformerFeatureExtractor(20, 128, 3, 8)

        # self.gpus = gpus
        #self.device = config.train.device
        # torch.device(gpus[0]) if len(gpus) > 0 else torch.device('cpu')
        # self.config = config

        self.node_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.edge_emb = torch.nn.Embedding(100, self.hidden_dim)
        self.input_mlp = layers.MultiLayerPerceptron(1, [self.hidden_dim, self.hidden_dim], activation=self.config.model.mlp_act)
        self.output_mlp = layers.MultiLayerPerceptron(self.hidden_dim, \
                                [self.hidden_dim, self.hidden_dim // 2, 1], activation=self.config.model.mlp_act)
        self.model = layers.GraphIsomorphismNetwork(hidden_dim=self.hidden_dim, \
                                 num_convs=self.config.model.num_convs, \
                                 activation=self.config.model.gnn_act, \
                                 readout="sum", short_cut=self.config.model.short_cut, \
                                 concat_hidden=self.config.model.concat_hidden)

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_noise_level)), dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False) # (num_noise_level)

        # time step embedding for continous SDE
        assert self.config.scheme.time_continuous
        if self.config.scheme.time_continuous:
            self.t_embed = nn.Sequential(GaussianFourierProjection(embed_dim=self.hidden_dim),
                                    nn.Linear(self.hidden_dim, self.hidden_dim))
            self.dense1 = nn.Linear(self.hidden_dim, 1)

    def marginal_prob_std(self, t, sigma, device='cuda'):
        """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

        Args:
          t: A vector of time steps.
          sigma: The $\sigma$ in our SDE.

        Returns:
          The standard deviation.
        """
        # t = torch.tensor(t, device=device)
        return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))

    @torch.no_grad()
    # extend the edge on the fly, second order: angle, third order: dihedral
    def extend_graph(self, data: Data, order=3):

        def binarize(x):
            return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

        def get_higher_order_adj_matrix(adj, order):
            """
            Args:
                adj:        (N, N)
                type_mat:   (N, N)
            """
            adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), \
                        binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

            for i in range(2, order+1):
                adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
            order_mat = torch.zeros_like(adj)

            for i in range(1, order+1):
                order_mat += (adj_mats[i] - adj_mats[i-1]) * i

            return order_mat

        num_types = len(utils.BOND_TYPES)

        N = data.num_nodes
        adj = to_dense_adj(data.edge_index).squeeze(0)
        adj_order = get_higher_order_adj_matrix(adj, order)  # (N, N)

        type_mat = to_dense_adj(data.edge_index, edge_attr=data.edge_type).squeeze(0)   # (N, N)
        type_highorder = torch.where(adj_order > 1, num_types + adj_order - 1, torch.zeros_like(adj_order))
        assert (type_mat * type_highorder == 0).all()
        type_new = type_mat + type_highorder

        new_edge_index, new_edge_type = dense_to_sparse(type_new)
        _, edge_order = dense_to_sparse(adj_order)

        data.bond_edge_index = data.edge_index  # Save original edges
        data.edge_index, data.edge_type = coalesce(new_edge_index, new_edge_type.long(), N, N) # modify data
        edge_index_1, data.edge_order = coalesce(new_edge_index, edge_order.long(), N, N) # modify data
        data.is_bond = (data.edge_type < num_types)
        assert (data.edge_index == edge_index_1).all()

        return data

    @torch.no_grad()
    def convert_score_d(self, score_d, pos, edge_index, edge_length):
        dd_dr = (1. / edge_length) * (pos[edge_index[0]] - pos[edge_index[1]])   # (num_edge, 3)
        score_pos = scatter_add(dd_dr * score_d, edge_index[0], dim=0)

        return score_pos

    # decouple the noise generator
    def noise_generator(self, data, noise_type):
        # node2graph = data.batch
        # edge2graph = node2graph[data.edge_index[0]]
        # if noise_type == 'symmetry':
        #     num_nodes = scatter_add(torch.ones(data.num_nodes, dtype=torch.long, device=self.Device), node2graph) # (num_graph)
        #     num_cum_nodes = num_nodes.cumsum(0) # (num_graph)
        #     node_offset = num_cum_nodes - num_nodes # (num_graph)
        #     edge_offset = node_offset[edge2graph] # (num_edge)
        #
        #     num_nodes_square = num_nodes**2 # (num_graph)
        #     num_nodes_square_cumsum = num_nodes_square.cumsum(-1) # (num_graph)
        #     edge_start = num_nodes_square_cumsum - num_nodes_square # (num_graph)
        #     edge_start = edge_start[edge2graph]
        #
        #     all_len = num_nodes_square_cumsum[-1]
        #
        #     node_index = data.edge_index.t() - edge_offset.unsqueeze(-1)
        #     #node_in, node_out = node_index.t()
        #     node_large = node_index.max(dim=-1)[0]
        #     node_small = node_index.min(dim=-1)[0]
        #     undirected_edge_id = node_large * (node_large + 1) + node_small + edge_start
        #     symm_noise = torch.Tensor(all_len.detach().cpu().numpy(), device='cpu').normal_()
        #     symm_noise = symm_noise.to(self.Device)
        #     d_noise = symm_noise[undirected_edge_id].unsqueeze(-1) # (num_edge, 1)
        # elif noise_type == 'rand':
        #     d = data.edge_length
        #     d_noise = torch.randn_like(d)
        d_noise = torch.randn_like(data)
        return d_noise

    def forward(self, data, device):
        """
        Input:
            data: torch geometric batched data object
        Output:
            loss
        """
        # a workaround to get the current device, we assume all tensors in a model are on the same device.
        self.Device = self.sigmas.device
        pad_seqs, pad_bonds, pad_angles, pad_dihedrals = data
        # print(pad_seqs.shape, pad_bonds.shape, pad_angles.shape, pad_dihedrals.shape)

        #dihedral diffusion
        eps = 1e-5
        random_t = torch.rand(len(pad_seqs), device=self.Device) * (1. - eps) + eps #(batch_size)
        dihedral_z = self.noise_generator(pad_dihedrals, self.noise_type).unsqueeze(-1).cuda()
        dihedral_std = self.marginal_prob_std(random_t, sigma=25).unsqueeze(1).repeat(1, dihedral_z.shape[1]).cuda()   #num_edge, f(x)=\sigma^t

        perturbed_dihedral = pad_dihedrals.unsqueeze(-1) + dihedral_z * dihedral_std[:,:,None] # [batch_size, seq_lenth, 1]
        d_emb = self.input_mlp(perturbed_dihedral) # (num_edge, hidden)
        # print(d_emb.shape)

        if self.config.scheme.time_continuous:
            # time embedding
            t_embedding = self.t_embed(random_t)             #(batch_dim, hidden_dim) = (128, 256)
            t_embedding = self.dense1(t_embedding)    #(batch_dim, 1) = (128,1)
            d_emb = d_emb + t_embedding.repeat(1, dihedral_z.shape[1]).unsqueeze(-1)

        # print(pad_seqs.shape)
        pad_seqs = repeat_sequence(pad_seqs, 3)
        seq_attr = self.feature_extraction(pad_seqs)

        seq_attr = seq_attr.reshape([-1, 256])
        d_emb = d_emb.reshape([-1, 256])
        # print(seq_attr.shape, d_emb.shape)

        seq_attr = d_emb * seq_attr # (num_edge, hidden)
        # output = self.model(data, node_attr, edge_attr)
        # h_row, h_col = output["node_feature"][data.edge_index[0]], output["node_feature"][data.edge_index[1]] # (num_edge, hidden)
        # distance_feature = torch.cat([h_row*h_col, edge_attr], dim=-1) # (num_edge, 2 * hidden)
        scores = self.output_mlp(seq_attr) # (num_edge, 1)
        scores = scores.view(-1)

        random_t_expand = random_t.unsqueeze(1).repeat(1, len(scores)//len(random_t)).reshape(-1)
        scores = scores * (1. / self.marginal_prob_std(random_t_expand, sigma=25)) # f_theta_sigma(x) =  f_theta(x) / sigma, (num_edge, 1)

        loss =  0.5 * ((scores[:,] * dihedral_std.reshape(-1)[:,] + dihedral_z.reshape(-1)) ** 2) # (num_edge)
        # loss = scatter_add(loss, edge2graph) # (num_graph)
        loss = loss.reshape(len(random_t), -1).sum(dim=1)
        return loss

    @torch.no_grad()
    def get_score(self, data, d, t):
        '''
        Args:
            data: the molecule_batched data, which provides edge_index and the information of other features
            d: input the reconsturcted data from sample process, e.g.torch.size[22234,1]
            t: input the SDE equation time with batch_size, e.g.torch.size[128]

        Returns:
            scores shape=(num_edges)
        '''
        node2graph = data.batch
        edge2graph = node2graph[data.edge_index[0]]
        node_attr = self.node_emb(data.atom_type)
        edge_attr = self.edge_emb(data.edge_type)
        d_emb = self.input_mlp(d)

        t_embedding = self.t_embed(t)
        t_embedding = self.dense1(t_embedding)
        d_emb = d_emb + t_embedding[edge2graph]

        edge_attr = d_emb * edge_attr
        output = self.model(data, node_attr, edge_attr)

        h_row, h_col = output["node_feature"][data.edge_index[0]], output["node_feature"][data.edge_index[1]]
        distance_feature = torch.cat([h_row * h_col, edge_attr], dim=-1)
        scores = self.output_mlp(distance_feature)
        scores = scores.view(-1)
        scores = scores * (1. / self.marginal_prob_std(t[edge2graph], sigma=25))
        return scores