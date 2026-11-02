import torch.nn as nn
from torch_geometric.nn import GINEConv, BatchNorm, Linear, GATConv, PNAConv, RGCNConv, to_hetero
from torch_geometric.nn.aggr import DegreeScalerAggregation
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import HeteroData
import torch.nn.functional as F
import torch
import logging
import numpy as np
from torch_scatter import scatter
from torch_geometric.utils import degree
from genagg import GenAgg
from genagg.MLPAutoencoder import MLPAutoencoder
import math 
import time


class DiscreteEncoder(nn.Module):
    def __init__(self, hidden_channels, max_num_features=10, max_num_values=500): #10
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(max_num_values, hidden_channels) 
                    for i in range(max_num_features)])

    def reset_parameters(self):
        for embedding in self.embeddings:
            embedding.reset_parameters()
            
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(1)
        out = 0
        for i in range(x.size(1)):
            out = out + self.embeddings[i](x[:, i])
        return out
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class TransformerAgg(nn.Module):
    def __init__(self, d_model = 66):
        super().__init__()

        self.pos_enc = PositionalEncoding(d_model=d_model, dropout=0.05, max_len=128)
        self.trans_enc = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=2, 
            dim_feedforward=d_model*4, 
            dropout=0.0,
            batch_first=True, # If True, then the input and output tensors are provided as (batch, seq, feature). 
            norm_first=False
            )

    def forward(self, x, index, timestamps_):
        # Add timestamps to the edge features to be able to sort according to them
        x = torch.cat([timestamps_.view(-1, 1), x], dim=1)
        
        s_pre = time.time()
        sort_ids = torch.argsort(index)
        dense_edge_feats, mask = to_dense_batch(x[sort_ids, :], index[sort_ids])
        sorted_dense_edge_feats, sorted_mask = self.sort_wrt_time(dense_edge_feats, mask)
        
        s_post = time.time()
        sorted_dense_edge_feats = self.pos_enc(sorted_dense_edge_feats.permute(1,0,2)).permute(1,0,2)
        sorted_dense_edge_feats = self.trans_enc(sorted_dense_edge_feats, src_key_padding_mask = ~sorted_mask)
        sorted_dense_edge_feats[~sorted_mask.unsqueeze(-1).expand(-1, -1, sorted_dense_edge_feats.shape[-1])] = 0
        s_res = time.time()

        logging.debug(f"Preprocessing Time Transformer: {s_post - s_pre} | Transformer forward pass: {s_res - s_post}")

        return sorted_dense_edge_feats.mean(dim=1).squeeze()

    def sort_wrt_time(self, matt, mask):
        first_feature = matt[:, :, 0] 
        sort_indices = torch.argsort(first_feature, dim=1)
        sorted_matt = torch.gather(matt, 1, sort_indices.unsqueeze(-1).expand(-1, -1, matt.shape[-1]))
        sorted_mask = torch.gather(mask, 1, sort_indices)
        return sorted_matt[:, :, 1:], sorted_mask

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class GRUAgg(nn.Module):
    def __init__(self, d_model = 66):
        super().__init__()

        self.gru = nn.GRU(
            d_model, 
            hidden_size=d_model, 
            num_layers=2, 
            batch_first=True
            )

    def forward(self, x, index, timestamps_):
        # Add timestamps to the edge features to be able to sort according to them
        x = torch.cat([timestamps_.view(-1, 1), x], dim=1)
        
        sort_ids = torch.argsort(index)
        dense_edge_feats, mask = to_dense_batch(x[sort_ids, :], index[sort_ids])
        sorted_dense_edge_feats, sorted_mask = self.sort_wrt_time(dense_edge_feats, mask)

        sorted_dense_edge_feats = self.gru(sorted_dense_edge_feats)[0]
        # sorted_dense_edge_feats[~sorted_mask.unsqueeze(-1).expand(-1, -1, sorted_dense_edge_feats.shape[-1])] = 0

        return sorted_dense_edge_feats.mean(dim=1).squeeze()

    def sort_wrt_time(self, matt, mask):
        first_feature = matt[:, :, 0] 
        sort_indices = torch.argsort(first_feature, dim=1)
        sorted_matt = torch.gather(matt, 1, sort_indices.unsqueeze(-1).expand(-1, -1, matt.shape[-1]))
        sorted_mask = torch.gather(mask, 1, sort_indices)
        return sorted_matt[:, :, 1:], sorted_mask

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
class PnaAgg(nn.Module):
    def __init__(self , n_hidden, deg):
        super().__init__()
        
        aggregators = ['mean', 'min', 'max', 'std']
        self.num_aggregators = len(aggregators)
        scalers = ['identity', 'amplification', 'attenuation']

        self.agg = DegreeScalerAggregation(aggregators, scalers, deg)
        self.lin = nn.Linear(len(scalers)*len(aggregators)*n_hidden, n_hidden)

    def forward(self, x, index):
        out = self.agg(x, index)
        return self.lin(out)

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                param = nn.init.kaiming_normal_(param.detach())
            elif 'bias' in name:
                param = nn.init.constant_(param.detach(), 0)
        
class IdentityAgg(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, index):
        return x

class GinAgg(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.nn = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), 
                nn.ReLU(), 
                nn.Linear(n_hidden, n_hidden)
                )
    def forward(self, x, index):
        out = scatter(x, index, dim=0, reduce='sum')
        return self.nn(out)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass

class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=True, bias=True, activation_fn="relu", LN='False'):
        super().__init__()
        n_hid = nout
        self.layers = nn.ModuleList([nn.Linear(nin if i==0 else n_hid, 
                                     n_hid if i<nlayer-1 else nout, 
                                     bias=True if (i==nlayer-1 and not with_final_activation and bias) # TODO: revise later
                                        or (not with_norm) else False) # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i<nlayer-1 else nout) if not LN else nn.LayerNorm(n_hid) if with_norm else Identity()
                                     for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin==nout) ## TODO: test whether need this
        self.act = getattr(F, activation_fn)

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = self.act(x)  

        return x


class TransactionEncoder(nn.Module):
    def __init__(self, edge_dim, n_hidden, cardinalities):
        super().__init__()
        self.edge_lin = nn.Linear(edge_dim, n_hidden)
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, n_hidden) for cardinality in cardinalities
        ])

    def forward(self, edge_attr, edge_cat):
        out = self.edge_lin(edge_attr)
        if edge_cat is None:
            raise ValueError('AML categorical features require edge_cat on every edge batch')
        for i, embedding in enumerate(self.cat_embeddings):
            out = out + embedding(edge_cat[:, i].long())
        return out
    
class AdammAgg(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        self.edge_transform = MLP(n_hidden, n_hidden, nlayer=1, with_final_activation=False)
               
    def forward(self, x, index):
        out = scatter(x, index, dim=0, reduce='sum')
        return self.edge_transform(out)

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class SumAgg(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, index):
        return scatter(x, index, dim=0, reduce='sum')

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class MultiEdgeAggModule(nn.Module):
    def __init__(self, n_hidden=None, agg_type=None, index=None,
                 conditioned_by_payment=False, num_payment_formats=None):
        super().__init__()
        self.agg_type = agg_type
        self.conditioned_by_payment = conditioned_by_payment
        self.num_payment_formats = num_payment_formats

        if self.conditioned_by_payment:
            if agg_type not in ('gin', 'sum'):
                raise ValueError('Payment-conditioned aggregation supports gin or sum')
            if not isinstance(num_payment_formats, int) or num_payment_formats <= 0:
                raise ValueError('Payment-conditioned aggregation requires a positive payment-format cardinality')

        if agg_type == 'genagg':
            self.agg = GenAgg(f=MLPAutoencoder, jit=False)
        elif agg_type == 'gin':
            self.agg = GinAgg(n_hidden=n_hidden)
        elif agg_type == 'pna':
            uniq_index, inverse_indices = torch.unique(index, return_inverse=True)
            d = degree(inverse_indices, num_nodes=uniq_index.numel(), dtype=torch.long)
            deg = torch.bincount(d, minlength=1)
            self.agg = PnaAgg(n_hidden=n_hidden, deg=deg)
        elif agg_type == 'sum':
            self.agg = SumAgg()
        elif agg_type == 'transformer':
            self.agg = TransformerAgg(d_model=n_hidden)
        elif agg_type == 'gru':
            self.agg = GRUAgg(d_model=n_hidden)
        elif agg_type == 'adamm':
            self.agg = AdammAgg(n_hidden)
        else:
            self.agg = IdentityAgg()

        if self.conditioned_by_payment:
            self.attention_dim = max(1, n_hidden // 4)
            self.context_query = nn.Linear(n_hidden, self.attention_dim, bias=False)
            self.context_key = nn.Linear(n_hidden, self.attention_dim, bias=False)
            self.context_gate = nn.Sequential(
                nn.Linear(3 * n_hidden, n_hidden),
                nn.Sigmoid(),
            )
        
    def forward(self, edge_index, edge_attr, simp_edge_batch, times, edge_cat=None):
        _, inverse_indices = torch.unique(simp_edge_batch, return_inverse=True)
        new_edge_index = scatter(edge_index, inverse_indices, dim=1, reduce='mean') if self.agg_type is not None else edge_index

        if (self.agg_type == 'transformer') or (self.agg_type == 'gru'):
            new_edge_attr = self.agg(x= edge_attr, index=inverse_indices, timestamps_=times)
        else:
            new_edge_attr = self.agg(x=edge_attr, index=inverse_indices)

        if not self.conditioned_by_payment:
            return new_edge_index, new_edge_attr, inverse_indices

        if edge_cat is None:
            raise ValueError('Payment-conditioned edge aggregation requires edge_cat')
        if edge_cat.dim() != 2 or edge_cat.size(1) != 3 or edge_cat.size(0) != edge_attr.size(0):
            raise ValueError('edge_cat must have one row per edge and three categorical columns')
        if edge_attr.size(0) == 0:
            return new_edge_index, new_edge_attr, inverse_indices, edge_attr

        payment_format = edge_cat[:, 2].long()
        pair_payment_ids = inverse_indices * self.num_payment_formats + payment_format
        group_ids, payment_inverse_indices = torch.unique(pair_payment_ids, return_inverse=True)
        pair_context = torch.index_select(new_edge_attr, 0, inverse_indices)

        # For single-format pairs the two contexts are identical, including their
        # gradients. Only mixed-format pairs need the attention and gate branch.
        pair_group_counts = torch.bincount(
            group_ids // self.num_payment_formats, minlength=new_edge_attr.size(0)
        )
        mixed_edges = (pair_group_counts[inverse_indices] > 1).nonzero(as_tuple=True)[0]
        if mixed_edges.numel() == 0:
            # Keep zero gradients (rather than None) so Adam still applies the
            # same momentum update after a previous mixed-format batch.
            zero = sum(p.reshape(-1)[0] * 0 for module in (
                self.context_query, self.context_key, self.context_gate
            ) for p in module.parameters())
            return new_edge_index, new_edge_attr, inverse_indices, pair_context + zero

        payment_edge_attr = self.agg(x=edge_attr, index=payment_inverse_indices)
        all_mixed = mixed_edges.numel() == edge_attr.size(0)
        mixed_attr = edge_attr if all_mixed else edge_attr.index_select(0, mixed_edges)
        mixed_pair_context = pair_context if all_mixed else pair_context.index_select(0, mixed_edges)
        # Each transaction queries all non-empty payment groups of its own pair.
        num_groups = payment_edge_attr.size(0)
        group_lookup = inverse_indices.new_full(
            (new_edge_attr.size(0) * self.num_payment_formats,), num_groups
        )
        group_lookup[group_ids] = torch.arange(num_groups, device=edge_attr.device)
        group_lookup = group_lookup.view(-1, self.num_payment_formats)
        candidate_groups = group_lookup.index_select(0, inverse_indices[mixed_edges])
        missing_groups = candidate_groups == num_groups
        # Masked weights stay zero. Use each transaction's own group as padding
        # instead of sending all missing slots through one shared gradient row.
        candidate_groups = torch.where(
            missing_groups, payment_inverse_indices[mixed_edges].unsqueeze(1), candidate_groups
        )
        group_values = payment_edge_attr
        keys = self.context_key(group_values)[candidate_groups]
        query = self.context_query(mixed_attr).unsqueeze(1)
        scores = (query * keys).sum(dim=-1) / math.sqrt(self.attention_dim)
        scores = scores.masked_fill(missing_groups, float('-inf'))
        weights = torch.softmax(scores, dim=1)
        # Weighted group sums without materializing an [edges, formats, hidden] tensor.
        selected_context = F.embedding_bag(
            candidate_groups, group_values, mode='sum', per_sample_weights=weights
        )
        gate = self.context_gate(torch.cat([mixed_attr, mixed_pair_context, selected_context], dim=-1))
        mixed_context = mixed_pair_context + gate * (selected_context - mixed_pair_context)
        conditioned_context = mixed_context if all_mixed else pair_context.index_copy(0, mixed_edges, mixed_context)
        return new_edge_index, new_edge_attr, inverse_indices, conditioned_context
    
    def reset_parameters(self):
        self.agg.reset_parameters()
        if self.conditioned_by_payment:
            self.context_query.reset_parameters()
            self.context_key.reset_parameters()
            for module in self.context_gate:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()

class MultiMPNN(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, n_hidden=100, 
                 edge_updates=False,edge_dim=None, final_dropout=0.5, 
                 index_ = None, deg=None, args=None):
        super().__init__()
        self.args = args
        self.n_hidden = n_hidden
        self.final_dropout = final_dropout
        self.use_aml_categorical = getattr(args, 'aml_use_categorical', False)
        self.use_aml_history = getattr(args, 'aml_use_history', False)

        self.node_emb = nn.Linear(num_features, n_hidden)
        if self.use_aml_categorical:
            self.edge_emb = TransactionEncoder(edge_dim, n_hidden, args.edge_cat_cardinalities)
        else:
            self.edge_emb = nn.Linear(edge_dim, n_hidden)

        if args.reverse_mp or args.reverse_mp_lp:
            if self.use_aml_categorical:
                self.edge_emb_rev = TransactionEncoder(edge_dim, n_hidden, args.edge_cat_cardinalities)
            else:
                self.edge_emb_rev = nn.Linear(edge_dim, n_hidden)

        if self.use_aml_history:
            self.history_encoder = nn.Sequential(
                nn.Linear(args.edge_hist_dim, n_hidden),
                nn.ReLU(),
                nn.Linear(n_hidden, n_hidden),
            )

        if args.edge_agg_type =='adamm':
            self.edge_direction_encoder = DiscreteEncoder(n_hidden, max_num_values=4)
    
        self.gnn = GnnHelper(num_gnn_layers=num_gnn_layers, n_hidden=n_hidden, edge_updates=edge_updates, final_dropout=final_dropout,
                             index_=index_, deg=deg, args=args)
        
        if args.reverse_mp or args.reverse_mp_lp:
            self.gnn = to_hetero(
                self.gnn,
                metadata=(['node'], [('node', 'to', 'node'), ('node', 'rev_to', 'node')]),
                aggr='mean',
                input_map={'x': 'node', 'edge_index': 'edge', 'edge_attr': 'edge',
                           'simp_edge_batch': 'edge', 'edge_cat': 'edge'},
            )

        if args.task == 'edge_class':
            readout_dim = n_hidden * (4 if self.use_aml_history else 3)
            self.mlp = nn.Sequential(Linear(readout_dim, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                                Linear(25, n_classes))
        elif args.task == 'node_class':
            self.mlp = nn.Sequential(Linear(n_hidden, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                                Linear(25, n_classes))
        elif args.task == 'lp':
            if args.ports_batch:
                # positive and negative edges do not contain port information, since ports are assigned after neighborhood sampling.
                self.edge_emb_new = nn.Linear(edge_dim-2, n_hidden)
            self.edge_readout = LinkPredHead(n_hidden=n_hidden, n_classes=1, final_dropout=final_dropout)
    
    def forward(self, data):
        if isinstance(data, HeteroData):

            # Initial Embedding Layers
            x_dict = {"node": self.node_emb(data['node'].x)}
            forward_edge = data['node', 'to', 'node']
            reverse_edge = data['node', 'rev_to', 'node']
            forward_attr = self.edge_emb(forward_edge.edge_attr, forward_edge.edge_cat) if self.use_aml_categorical else self.edge_emb(forward_edge.edge_attr)
            reverse_attr = self.edge_emb_rev(reverse_edge.edge_attr, reverse_edge.edge_cat) if self.use_aml_categorical else self.edge_emb_rev(reverse_edge.edge_attr)
            edge_attr_dict = {
                    ("node", 'to', 'node'): forward_attr,
                    ("node", 'rev_to', 'node'): reverse_attr,
                }
            simp_edge_batch_dict = data.simp_edge_batch_dict if self.args.flatten_edges else None
            edge_cat_dict = None
            if getattr(self.args, 'aml_conditioned_edge_agg', False):
                edge_cat_dict = {
                    ("node", 'to', 'node'): forward_edge.edge_cat,
                    ("node", 'rev_to', 'node'): reverse_edge.edge_cat,
                }

            # Message Passing Layers
            if edge_cat_dict is None:
                x_dict, edge_attr_dict = self.gnn(
                    x_dict, data.edge_index_dict, edge_attr_dict, simp_edge_batch_dict
                )
            else:
                x_dict, edge_attr_dict = self.gnn(
                    x_dict, data.edge_index_dict, edge_attr_dict, simp_edge_batch_dict, edge_cat_dict
                )
            x = x_dict['node']
            edge_attr = edge_attr_dict['node', 'to', 'node']

            # Prediction Heads
            if self.args.task == 'edge_class':
                x = x[data['node', 'to', 'node'].edge_index.T].reshape(-1, 2*self.n_hidden).relu()
                x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
                if self.use_aml_history:
                    x = torch.cat((x, self.history_encoder(forward_edge.edge_hist)), 1)
                out = self.mlp(x)
            elif self.args.task == 'node_class':
                out = self.mlp(x)
            elif self.args.task == 'lp':
                if self.args.ports_batch:
                    neg_edge_attr = self.edge_emb_new(data['node', 'to', 'node'].neg_edge_attr) 
                    pos_edge_attr = self.edge_emb_new(data['node', 'to', 'node'].pos_edge_attr)
                else:
                    neg_edge_attr = self.edge_emb(data['node', 'to', 'node'].neg_edge_attr) 
                    pos_edge_attr = self.edge_emb(data['node', 'to', 'node'].pos_edge_attr)
                out = self.edge_readout(x, data['node', 'to', 'node'].pos_edge_index, pos_edge_attr, data['node', 'to', 'node'].neg_edge_index, neg_edge_attr)

        else:
            # Initial Embedding Layers
            x = self.node_emb(data.x)
            edge_attr = self.edge_emb(data.edge_attr, data.edge_cat) if self.use_aml_categorical else self.edge_emb(data.edge_attr)
            simp_edge_batch = data.simp_edge_batch if self.args.flatten_edges else None

            if self.args.edge_agg_type =='adamm':
                edge_attr = edge_attr + self.edge_direction_encoder(data.edge_direction)

            # Message Passing Layers
            if getattr(self.args, 'aml_conditioned_edge_agg', False):
                x, edge_attr = self.gnn(x, data.edge_index, edge_attr, simp_edge_batch, data.edge_cat)
            else:
                x, edge_attr = self.gnn(x, data.edge_index, edge_attr, simp_edge_batch)

            # Prediction Heads
            if self.args.task == 'edge_class':
                x = x[data.edge_index.T].reshape(-1, 2*self.n_hidden).relu()
                x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
                if self.use_aml_history:
                    x = torch.cat((x, self.history_encoder(data.edge_hist)), 1)
                out = self.mlp(x)
            elif self.args.task == 'node_class':
                out = self.mlp(x)
            elif self.args.task == 'lp':
                if self.args.ports_batch:
                    neg_edge_attr = self.edge_emb_new(data.neg_edge_attr) 
                    pos_edge_attr = self.edge_emb_new(data.pos_edge_attr)
                else:
                    neg_edge_attr = self.edge_emb(data.neg_edge_attr) 
                    pos_edge_attr = self.edge_emb(data.pos_edge_attr)
                out = self.edge_readout(x, data.pos_edge_index, pos_edge_attr, data.neg_edge_index, neg_edge_attr)
        return out

class LinkPredHead(torch.nn.Module):
    """Readout head for link prediction.

    Parameters
    ----------
    config : GNNConfig
        Architecture configuration
    """
    def __init__(self, n_hidden, n_classes, final_dropout) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.final_dropout = final_dropout

        self.mlp = nn.Sequential(Linear(self.n_hidden*3, self.n_hidden), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(self.n_hidden, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
                            Linear(25, self.n_classes))
            
    def forward(self, x, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr):
        #reshape s.t. each row in x corresponds to the concatenated src and dst node features for each edge
        x_pos = x[pos_edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x_neg = x[neg_edge_index.T].reshape(-1, 2 * self.n_hidden).relu()

        #concatenate the node feature vector with the corresponding edge features
        x_pos = torch.cat((x_pos, pos_edge_attr.view(-1, pos_edge_attr.shape[1])), 1)
        x_neg = torch.cat((x_neg, neg_edge_attr.view(-1, neg_edge_attr.shape[1])), 1)

        return (torch.sigmoid(self.mlp(x_pos)), torch.sigmoid(self.mlp(x_neg)))

class GnnHelper(torch.nn.Module):
    def __init__(self, num_gnn_layers, n_hidden=100, edge_updates=False, 
                final_dropout=0.5, index_ = None, deg = None, args=None):
        super().__init__()

        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout
        self.flatten_edges = args.flatten_edges
        self.edge_agg_type = args.edge_agg_type
        self.args = args
        self.use_dual_endpoint_update = getattr(args, 'aml_dual_endpoint_update', False)
        self.use_conditioned_edge_agg = getattr(args, 'aml_conditioned_edge_agg', False)

        if self.use_dual_endpoint_update or self.use_conditioned_edge_agg:
            if not self.flatten_edges or not self.edge_updates:
                raise ValueError('AML structural modules require flattened edges and edge updates')
            if self.edge_agg_type not in ('gin', 'sum'):
                raise ValueError('AML structural modules support edge_agg_type gin or sum')
        if self.use_conditioned_edge_agg:
            cardinalities = getattr(args, 'edge_cat_cardinalities', [])
            if not getattr(args, 'aml_use_categorical', False) or len(cardinalities) != 3:
                raise ValueError('Conditioned aggregation requires the B2 categorical encoder and its three cardinalities')

    
        if args.node_agg_type == 'genagg':
            self.node_agg = GenAgg(f=MLPAutoencoder, jit=False)
        elif args.node_agg_type == 'sum':
            self.node_agg = 'sum'
     
        self.edge_aggrs = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            if args.model == 'gin':
                conv = GINEConv(nn.Sequential(
                    nn.Linear(self.n_hidden, self.n_hidden), 
                    nn.ReLU(), 
                    nn.Linear(self.n_hidden, self.n_hidden)
                    ), 
                    edge_dim=self.n_hidden, 
                    aggr = self.node_agg # Added New!!!
                    )
            elif args.model == 'pna':
                aggregators = ['mean', 'min', 'max', 'std']
                scalers = ['identity', 'amplification', 'attenuation']
                conv = PNAConv(in_channels=n_hidden, out_channels=n_hidden,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=n_hidden, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
                
            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear((4 if self.use_dual_endpoint_update else 3) * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            if self.flatten_edges:
                edge_agg = MultiEdgeAggModule(
                    n_hidden,
                    agg_type=args.edge_agg_type,
                    index=index_,
                    conditioned_by_payment=self.use_conditioned_edge_agg,
                    num_payment_formats=(args.edge_cat_cardinalities[2] if self.use_conditioned_edge_agg else None),
                )
                self.edge_aggrs.append(edge_agg)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))
        

        if self.edge_agg_type == 'adamm':
            self.edge_aggrs = None
            self.edge_agg = MultiEdgeAggModule(n_hidden, agg_type=args.edge_agg_type, index=index_)
        

    def forward(self, x, edge_index, edge_attr, simp_edge_batch=None, edge_cat=None):
        times= None # TODO: Not implemented yet

        if self.edge_agg_type == 'adamm':
            # Apply the flattening at the beggining only.
            edge_index, edge_attr, _ = self.edge_agg(edge_index, edge_attr, simp_edge_batch, times)
            src, dst = edge_index

            for i in range(self.num_gnn_layers):
                x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
                if self.edge_updates: 
                    edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2   
        else:
            src, dst = edge_index

            for i in range(self.num_gnn_layers):
                if self.flatten_edges:
                    if self.use_conditioned_edge_agg:
                        n_edge_index, n_edge_attr, inverse_indices, remapped_edge_attr = self.edge_aggrs[i](
                            edge_index, edge_attr, simp_edge_batch, times, edge_cat
                        )
                    else:
                        n_edge_index, n_edge_attr, inverse_indices = self.edge_aggrs[i](
                            edge_index, edge_attr, simp_edge_batch, times
                        )
                    x = (x + F.relu(self.batch_norms[i](self.convs[i](x, n_edge_index, n_edge_attr)))) / 2
                    if self.edge_updates: 
                        if not self.use_conditioned_edge_agg:
                            remapped_edge_attr = torch.index_select(n_edge_attr, 0, inverse_indices)
                        update_inputs = [x[src]]
                        if self.use_dual_endpoint_update:
                            update_inputs.append(x[dst])
                        update_inputs.extend([remapped_edge_attr, edge_attr])
                        edge_attr = edge_attr + self.emlps[i](torch.cat(update_inputs, dim=-1)) / 2
                else:
                    x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
                    if self.edge_updates: 
                        edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
                    
        return x, edge_attr

