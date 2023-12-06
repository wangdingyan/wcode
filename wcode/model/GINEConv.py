import torch
from torch import FloatTensor, LongTensor
import torch_geometric.data
from torch import Tensor
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.nn.conv import MessagePassing
from wcode.model.linear import Linear
from typing import Union, Tuple, Optional
from torch_geometric.typing import Adj, OptPairTensor, OptTensor
from torch_scatter import scatter_sum, scatter_mean, scatter_max
from torch_geometric.data import Batch as PyGBatch, Data as PyGData


class GraphEmbeddingModel(nn.Module):
    def __init__(self,
                 node_input_dim: int,
                 edge_input_dim: int,
                 global_input_dim: Optional[int] = 0,
                 hidden_dim: int = 128,
                 graph_vector_dim: Optional[int] = 0,
                 n_block: int = 2,
                 dropout: float = 0.0):
        super(GraphEmbeddingModel, self).__init__()

        global_input_dim = global_input_dim or 0
        graph_vector_dim = graph_vector_dim or 0

        self.node_embedding = Linear(
            input_dim=node_input_dim + global_input_dim,
            output_dim=hidden_dim,
            activation='SiLU'
        )

        self.edge_embedding = Linear(
            input_dim=edge_input_dim,
            output_dim=hidden_dim,
            activation='SiLU'
        )

        self.blocks = nn.ModuleList([
            ResidualBlock(
                node_dim=hidden_dim, edge_dim=hidden_dim,
                activation='SiLU', layer_norm=None, dropout=dropout)
            for _ in range(n_block)
        ])

        self.final_node_embedding = Linear(
            input_dim=hidden_dim + node_input_dim,
            output_dim=hidden_dim,
            activation='SiLU',
            dropout=dropout,
        )
        if graph_vector_dim > 0:
            self.readout = Readout(
                node_dim=hidden_dim,
                hidden_dim=graph_vector_dim,
                output_dim=graph_vector_dim,
                global_input_dim=global_input_dim,
                activation='SiLU',
                dropout=dropout
            )
        else:
            self.readout = None

    def forward(
            self,
            x_inp,
            edge_index,
            edge_attr,
            global_x=None,
            node2graph=None,
    ):
        """
        Input :
            x_inp: input node feature of graph          (V, Fv)
            edge_index: edge index of graph             (2, E)
            edge_attr: input edge attr of graph         (E, Fe)
            global_x: input graph feature such as condition (optional)
                                                        (N, Fc)
            node2graph: map node to graph (optional)    (V,)

        Output:
            x_upd: updated node feature                 (V, Fh)
            Z: latent vector of graph (graph vector)    (N, Fz)
                if graph_vector_dim is 0, Z is None
        """
        x = self.concat(x_inp, global_x, node2graph)
        x_emb = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        for convblock in self.blocks:
            x_emb = convblock(x_emb, edge_index, edge_attr, node2graph)

        x_emb = torch.cat([x_emb, x_inp], dim=-1)
        x_emb = self.final_node_embedding(x_emb)
        if self.readout is not None:
            Z = self.readout(x_emb, node2graph, global_x)
        else:
            Z = None

        return x_emb, Z

    def forward_batch(self,
                      batch: Union[PyGBatch, PyGData]):  # -> Tuple[NodeVector, Optional[GraphVector]]
        if isinstance(batch, PyGBatch):
            node2graph = batch.batch
        else:
            node2graph = None

        global_x = batch.get('global_x', None)

        return self.forward(batch.x,
                            batch.edge_index,
                            batch.edge_attr,
                            global_x,
                            node2graph)

    def concat(self,
               x,
               global_x,
               node2graph: LongTensor) -> FloatTensor:

        if global_x is not None:
            if node2graph is None:
                global_x = global_x.repeat(x.size(0), 1)
            else:
                global_x = global_x[node2graph]
            x = torch.cat([x, global_x], dim=-1)
        return x


class ResidualBlock(nn.Module):
    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            activation: Optional[str] = 'SiLU',
            layer_norm: Optional[str] = None,
            dropout: float = 0.0,
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = GINEConv(node_dim, edge_dim, activation, layer_norm, dropout)
        self.graph_norm1 = pyg_nn.LayerNorm(in_channels=node_dim, mode='graph')

        self.conv2 = GINEConv(node_dim, edge_dim, activation, layer_norm, dropout)
        self.graph_norm2 = pyg_nn.LayerNorm(in_channels=node_dim, mode='graph')
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor,
                edge_index: Adj,
                edge_attr: Tensor,
                node2graph: OptTensor):
        identity = x

        out = self.conv1(x, edge_index, edge_attr=edge_attr)
        out = self.graph_norm1(out, node2graph)
        out = self.relu(out)

        out = self.conv2(x, edge_index, edge_attr=edge_attr)
        out = self.graph_norm2(out, node2graph)

        out = (out + identity) / 2
        out = self.relu(out)

        return out


class GINEConv(MessagePassing):
    def __init__(
            self,
            node_dim: Union[int, Tuple[int]],
            edge_dim: int,
            activation: Optional[str] = None,
            norm: Optional[str] = None,
            dropout: float = 0.0,
            **kwargs,
    ):
        super(GINEConv, self).__init__(**kwargs)

        if isinstance(node_dim, int):
            src_node_dim, dst_node_dim = node_dim, node_dim
        else:  # (int, int)
            src_node_dim, dst_node_dim = node_dim

        self.edge_layer = Linear(edge_dim, src_node_dim, activation, dropout=dropout)
        self.eps = torch.nn.Parameter(torch.Tensor([0.1]))
        self.nn = nn.Sequential(
            Linear(dst_node_dim, dst_node_dim, activation, norm, dropout=dropout),
            Linear(dst_node_dim, dst_node_dim, activation, norm, dropout=0.0),
        )

    def forward(self, x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                edge_attr: Tensor) -> Tensor:
        """
        x: node feature             [(V_src, Fh_src), (V_dst, Fh_dst)]
        edge_index: edge index      (2, E)
        edge_attr: edge feature     (E, Fe)
        """
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        x_src, x_dst = x

        x_dst_upd = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        if x_dst is not None:
            x_dst_upd = x_dst_upd + (1 + self.eps) * x_dst

        return self.nn(x_dst_upd)

    def message(self, x_j, edge_attr):
        # x_i: dst, x_j: src
        edge_attr = self.edge_layer(edge_attr)
        return (x_j + edge_attr).relu()  # (E, Fh_dst)

class Readout(nn.Module) :
    """
    Input
    nodes : n_node, node_dim
    global_x: n_graph, global_dim

    Output(Graph Vector)
    retval : n_graph, output_dim
    """
    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        output_dim: int,
        global_input_dim: Optional[int] = None,
        activation: Optional[str] = None,
        dropout: float = 0.0,
    ) :

        super(Readout, self).__init__()
        global_input_dim = global_input_dim or 0

        self.linear1 = Linear(node_dim, hidden_dim, None, dropout = dropout)

        self.linear2 = Linear(node_dim, hidden_dim, 'Sigmoid', dropout = 0.0)
        self.linear3 = Linear(hidden_dim*2 + global_input_dim, output_dim, activation, dropout = 0.0)

    def forward(self,
                x: FloatTensor,
                node2graph: Optional[LongTensor] = None,
                global_x: Optional[FloatTensor] = None) -> FloatTensor:
        """
        x: [V, Fh]
        node2graph: optional, [V, ]
        global_x: optional, [N, Fg]
        """
        x = self.linear1(x) * self.linear2(x)               # Similar to SiLU   SiLU(x) = x * sigmoid(x)
        if node2graph is not None :
            Z1 = scatter_sum(x, node2graph, dim=0)          # V, Fh -> N, Fz
            Z2 = scatter_mean(x, node2graph, dim=0)         # V, Fh -> N, Fz
        else :  # when N = 1
            Z1 = x.sum(dim=0, keepdim = True)               # V, Fh -> 1, Fz
            Z2 = x.mean(dim=0, keepdim = True)              # V, Fh -> 1, Fz
        if global_x is not None :
            Z = torch.cat([Z1, Z2, global_x], dim=-1)       # N, 2*Fh + Fg
        else :
            Z = torch.cat([Z1, Z2], dim=-1)                 # N, 2*Fh
        return self.linear3(Z)


if __name__ == '__main__':
    # model = GINEConv(20, 30)
    # x = torch.randn((5, 20))
    # edge_attr = torch.randn(4, 30)
    # edge_index = torch.tensor([[0,0,0,0], [1,2,3,4]])
    # data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    # print(model(x=x, edge_attr=edge_attr, edge_index=edge_index).shape)

    # from torch_geometric.datasets import TUDataset
    #
    # dataset = TUDataset(root='C:\\data\\LGDrugAI\\data\\ENZYMES', name='ENZYMES')
    # sample = dataset[0]
    # print(sample)
    # sample.edge_attr = torch.randn(168, 10)
    # model = GraphEmbeddingModel(3,
    #                             10,
    #                             0,
    #                             128,
    #                             12)
    # print(model.forward_batch(sample)[1].shape)
    import torch

    odel = GraphEmbeddingModel(3,
                               10,
                               0,
                               128,
                               12)
    g = torch.load('C:\\database\\PDBBind\\PDBBind_processed\\1a0q\\1a0q_pyg.pt')

    print(g)

