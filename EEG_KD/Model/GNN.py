import torch as tc
import torch.nn as nn
import numpy as np
import torch.nn.functional as func
from torch_geometric.nn.conv import GraphConv,GCNConv
from torch.nn.parameter import Parameter
from torch_geometric.data import Data,Batch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import global_add_pool,MLP
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import NoneType  # noqa
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value
class GATConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
            self.att_edge = Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.register_parameter('att_edge', None)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None,
                return_attention_weights=None):

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else:
            x_src, x_dst = x
            assert x_src.dim() == 2
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = torch_sparse.set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    # TODO TorchScript requires to return a tuple
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def edge_update(self, alpha_j: Tensor, alpha_i: OptTensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr is not None and self.lin_edge is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
from typing import Callable, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.inits import uniform
from torch_geometric.utils import scatter, softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes

def topk(
    x: Tensor,
    ratio: Optional[Union[float, int]],
    batch: Tensor,
    min_score: Optional[float] = None,
    tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)

    elif ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')
        batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max())

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ), -60000.0)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0), ), int(ratio))
            k = torch.min(k, num_nodes)
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        if isinstance(ratio, int) and (k == ratio).all():
            index = torch.arange(batch_size, device=x.device) * max_num_nodes
            index = index.view(-1, 1).repeat(1, ratio).view(-1)
            index += torch.arange(ratio, device=x.device).repeat(batch_size)
        else:
            # Otherwise, compute indices per graph:
            index = torch.cat([
                torch.arange(k[i], device=x.device) + i * max_num_nodes
                for i in range(batch_size)
            ], dim=0)

        perm = perm[index]

    else:
        raise ValueError("At least one of 'min_score' and 'ratio' parameters "
                         "must be specified")

    return perm


def filter_adj(
    edge_index: Tensor,
    edge_attr: Optional[Tensor],
    perm: Tensor,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index[0], edge_index[1]
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr


class TopKPooling(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        min_score: Optional[float] = None,
        multiplier: float = 1.,
        nonlinearity: Union[str, Callable] = 'tanh',
    ):
        super().__init__()

        if isinstance(nonlinearity, str):
            nonlinearity = getattr(torch, nonlinearity)

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.weight = Parameter(torch.Tensor(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        attn: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor, Tensor, Tensor]:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = (attn * self.weight).sum(dim=-1)

        if self.min_score is None:
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f'ratio={self.ratio}'
        else:
            ratio = f'min_score={self.min_score}'

        return (f'{self.__class__.__name__}({self.in_channels}, {ratio}, '
                f'multiplier={self.multiplier})')

class C_GNN(nn.Module):
    def __init__(self,in_channels,out_channels,stride):#out_channels每个节点的特征数
        super(C_GNN, self).__init__()
        self.out_channels=out_channels
        self.k = 5
        self.max = 2
        self.cnn = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=self.k,stride=stride,bias=True),
                                 nn.ReLU(),
                                 nn.AvgPool1d(kernel_size=self.max))
        self.lin = nn.Linear(in_features=out_channels,out_features=out_channels)
        self.lstm = nn.LSTM(input_size=out_channels,hidden_size=out_channels,
                            num_layers=1,batch_first=True)
        self.gcn = GATConv(in_channels=self.out_channels,out_channels=self.out_channels,edge_dim=1)

    def forward(self,x,edge_index,edge_weight):
        num_graph,num_ele,c,l = x.size()
        x = x.view(num_graph*num_ele,c,l)
        cnn_x = self.cnn(x)#(num_graph*num_ele,c,l)
        x = tc.transpose(cnn_x, 1, 2)#(num_graph*num_ele,l,c)
        x = self.lin(x)
        x,_ = self.lstm(x)
        x = x[:,-1,:]
        x,(index,score) = self.gcn(x,edge_index,edge_weight,return_attention_weights=True)

        cnn_x = cnn_x.reshape(num_graph, num_ele, cnn_x.shape[1],cnn_x.shape[2])
        return cnn_x,x,score

class NET_MODEL(nn.Module):
    def __init__(self):
        super(NET_MODEL, self).__init__()
        self.c = [8,8]
        self.cgcn1 = C_GNN(in_channels=1,out_channels=self.c[0],stride=3)
        self.dropout1 = nn.Dropout(p=0.5)
        self.cgcn2 = C_GNN(in_channels=self.c[0],out_channels=self.c[1],stride=3)
        self.dropout2 = nn.Dropout(p=0.5)

        self.pool =TopKPooling(in_channels=int(np.sum(self.c)),ratio=0.2)
        self.lin1 = MLP(in_channels=int(np.sum(self.c)),
                        out_channels=2,
                        num_layers=1)
    def forward(self,x,adj):
        #(BATCH , 128 , Sample point)
        num_graph,num_ele,f = x.size()
        x = x.cuda()
        adj = adj.cuda()
        edge_index, edge_weight = dense_to_sparse(adj)
        x = tc.unsqueeze(x, dim=2)
        cnn_x,x1,s1 = self.cgcn1(x,edge_index, edge_weight)
        cnn_x = self.dropout1(cnn_x)
        x1 = self.dropout1(x1)
        _,x2,s2 = self.cgcn2(cnn_x,edge_index, edge_weight)
        x2 = self.dropout2(x2)
        x = tc.concat((x1,x2),dim=1)
        batch = tc.repeat_interleave(tc.arange(num_graph),num_ele).cuda()
        x, edge_index, edge_attr, batch, perm, score = self.pool(x,edge_index,edge_weight,batch)
        x = global_add_pool(x,batch)
        x = self.lin1(x)
        return x,s1,s2


class LSTM_COM(nn.Module):
    def __init__(self):
        super(LSTM_COM, self).__init__()
        self.lstm = nn.LSTM(input_size=128,hidden_size=20,proj_size=1,num_layers=2,dropout=0.5)
        self.lin1 = MLP(in_channels=500,
                        hidden_channels=64,
                        out_channels=2,
                        num_layers=2)
    def forward(self, x):
        # (BATCH , 128 , Sample point)
        x = x.cuda()
        x = tc.transpose(x,1,2)#(b,500,128)
        x,_ = self.lstm(x)#(b,500,1)
        x = tc.squeeze(x)
        x = self.lin1(x)
        return x

class GNN_COM(nn.Module):
    def __init__(self):
        super(GNN_COM, self).__init__()
        self.lin = nn.Linear(in_features=500,out_features=64)
        self.gnn1 = GCNConv(in_channels=64,out_channels=32)
        self.gnn2 = GCNConv(in_channels=32, out_channels=16)
        self.lin1 = MLP(in_channels=16,
                        out_channels=2,
                        num_layers=1)
    def forward(self,x,adj):
        #(BATCH , 128 , Sample point)
        num_graph,num_ele,f = x.size()
        x = x.cuda()
        x = x.reshape(num_graph*num_ele,-1)
        adj = adj.cuda()
        edge_index, edge_weight = dense_to_sparse(adj)
        x = self.lin(x)
        x = func.relu(self.gnn1(x, edge_index, edge_weight))
        x = func.relu(self.gnn2(x, edge_index, edge_weight))
        batch = tc.repeat_interleave(tc.arange(num_graph),num_ele).cuda()
        x = global_add_pool(x,batch)
        x = self.lin1(x)
        return x
# i = tc.randn(50,128,500).cuda()
# m = LSTM_COM().cuda()
# out = m(i)
# print(out.shape)
# batch = tc.repeat_interleave(tc.arange(num_graph),128)
# self.weight = nn.Parameter(tc.FloatTensor(128, 128))
# nn.init.xavier_uniform_(self.weight,gain = math.sqrt(2.0))