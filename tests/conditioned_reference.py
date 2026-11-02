"""Original version-2 forward, retained for equivalence tests and benchmarks."""

import math

import torch
import torch.nn.functional as F
from torch_scatter import scatter


def original_conditioned_forward(module, edge_index, edge_attr, simp_edge_batch, times, edge_cat):
    _, inverse = torch.unique(simp_edge_batch, return_inverse=True)
    pair_index = scatter(edge_index, inverse, dim=1, reduce='mean')
    pair_attr = module.agg(x=edge_attr, index=inverse)
    if not module.conditioned_by_payment:
        return pair_index, pair_attr, inverse
    if edge_attr.size(0) == 0:
        return pair_index, pair_attr, inverse, edge_attr
    group_ids, group_inverse = torch.unique(
        inverse * module.num_payment_formats + edge_cat[:, 2].long(), return_inverse=True
    )
    payment_attr = module.agg(x=edge_attr, index=group_inverse)
    pair_context = pair_attr.index_select(0, inverse)
    num_groups = payment_attr.size(0)
    lookup = inverse.new_full((pair_attr.size(0) * module.num_payment_formats,), num_groups)
    lookup[group_ids] = torch.arange(num_groups, device=edge_attr.device)
    candidates = lookup.view(-1, module.num_payment_formats).index_select(0, inverse)
    values = torch.cat([payment_attr, edge_attr.new_zeros((1, edge_attr.size(1)))])
    keys = module.context_key(values)[candidates]
    query = module.context_query(edge_attr).unsqueeze(1)
    scores = (query * keys).sum(-1) / math.sqrt(module.attention_dim)
    weights = torch.softmax(scores.masked_fill(candidates == num_groups, float('-inf')), dim=1)
    selected = F.embedding_bag(candidates, values, mode='sum', per_sample_weights=weights)
    gate = module.context_gate(torch.cat([edge_attr, pair_context, selected], dim=-1))
    return pair_index, pair_attr, inverse, pair_context + gate * (selected - pair_context)
