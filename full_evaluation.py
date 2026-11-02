"""Complete-seed evaluation for existing AML edge-classification checkpoints."""

import hashlib
import logging

import numpy as np
import torch
from torch_geometric.data import HeteroData

FORWARD = ('node', 'to', 'node')
PROTOCOL = 'complete_forward_seeds_v1'
AML_DATASETS = ('Small_HI', 'Small_LI', 'Medium_HI', 'Medium_LI')


def forward_store(data):
    return data[FORWARD] if isinstance(data, HeteroData) else data


def complete_seed_edges(batch, data, input_ids):
    """Append missing forward seeds; keep sampled reverse edges unchanged.

    Parallel transactions are identified by integer edge ID, never endpoints.
    The added edges participate in message passing, as sampled seeds already do.
    Labels are copied for scoring only, never concatenated to model inputs.
    """
    store, source = forward_store(batch), forward_store(data)
    seeds = input_ids.detach().cpu()[store.input_id.cpu()]
    sampled_ids = store.e_id.cpu()
    if seeds.unique().numel() != seeds.numel() or sampled_ids.unique().numel() != sampled_ids.numel():
        raise ValueError('Duplicate seed/sample IDs in complete evaluation')
    missing = seeds[~torch.isin(seeds, sampled_ids)]
    if missing.numel():
        nodes = batch['node'].n_id if isinstance(batch, HeteroData) else batch.n_id
        sorted_nodes, permutation = nodes.cpu().sort()
        endpoints = source.edge_index[:, missing].cpu()
        positions = torch.searchsorted(sorted_nodes, endpoints)
        if ((positions >= len(sorted_nodes)).any()
                or not torch.equal(sorted_nodes[positions], endpoints)):
            raise ValueError('Seed endpoints missing from sampled nodes')
        local_edges = permutation[positions]
        # Explicit fields avoid confusing node and edge tensors of equal length.
        for key in ('edge_attr', 'y', 'edge_cat', 'edge_hist', 'simp_edge_batch',
                    'timestamps', 'edge_group', 'edge_time_group'):
            if key in source:
                if key not in store or store[key].size(0) != sampled_ids.numel():
                    raise ValueError(f'Misaligned sampled edge field: {key}')
                store[key] = torch.cat((store[key], source[key][missing]), dim=0)
        store.edge_index = torch.cat((store.edge_index, local_edges), dim=1)
        store.e_id = torch.cat((sampled_ids, missing))
    mask = torch.isin(store.e_id.cpu(), seeds)
    if not torch.equal(store.e_id.cpu()[mask].sort().values, seeds.sort().values):
        raise ValueError('Complete evaluation did not recover every seed edge exactly once')
    return mask, int(missing.numel())


def validate_coverage(predicted_ids, expected_ids):
    predicted = np.asarray(predicted_ids, dtype=np.int64)
    expected = np.asarray(expected_ids, dtype=np.int64)
    if (len(np.unique(predicted)) != len(predicted)
            or len(np.unique(expected)) != len(expected)
            or not np.array_equal(np.sort(predicted), np.sort(expected))):
        raise ValueError(f'Incomplete or duplicate evaluation IDs: {len(predicted)}/{len(expected)}')
    return hashlib.sha256(np.sort(predicted).astype('<i8').tobytes()).hexdigest()


@torch.no_grad()
def collect_complete_predictions(model, loader, ids, data, device, args, strip_edge_ids=False):
    from data_util import assign_ports_with_cpp
    was_training = model.training
    model.eval()
    collected = {key: [] for key in ('ids', 'logits', 'labels', 'groups', 'times')}
    inserted = 0
    try:
        for batch in loader:
            mask, count = complete_seed_edges(batch, data, ids)
            inserted += count
            store = forward_store(batch)
            collected['ids'].append(store.e_id.cpu()[mask].numpy())
            collected['labels'].append(store.y.cpu()[mask].numpy())
            for key, field in (('groups', 'edge_group'), ('times', 'edge_time_group')):
                if field in store:
                    collected[key].append(store[field].cpu()[mask].numpy())
            if strip_edge_ids:
                stores = batch.edge_stores if isinstance(batch, HeteroData) else [batch]
                for edge in stores:
                    edge.edge_attr = edge.edge_attr[:, 1:]
            if args.ports and args.ports_batch:
                assign_ports_with_cpp(batch, process_batch=isinstance(batch, HeteroData))
            logits = model(batch.to(device))[mask.to(device)]
            if not torch.isfinite(logits).all():
                raise ValueError('Non-finite logits in complete evaluation')
            collected['logits'].append(logits.cpu().numpy())
    finally:
        model.train(was_training)
    result = {key: np.concatenate(values) if values else None for key, values in collected.items()}
    if result['ids'] is None:
        raise ValueError('Empty evaluation split')
    validate_coverage(result['ids'], ids.cpu().numpy())
    order = np.argsort(result['ids'])
    result = {key: value[order] if value is not None else None for key, value in result.items()}
    logging.info('Complete evaluation scored/requested edges: %d/%d; inserted forward seeds: %d',
                 len(result['ids']), len(ids), inserted)
    return result


def complete_metrics(predictions, threshold=None):
    from train_util import compute_binary_metrics, select_f1_threshold
    logits, labels = predictions['logits'], predictions['labels']
    threshold = select_f1_threshold(logits, labels) if threshold is None else float(threshold)
    values = compute_binary_metrics(logits, labels, threshold)
    result = dict(zip(('F1', 'PR-AUC', 'AP', 'Precision', 'Recall'), map(float, values)))
    result.update(threshold=float(threshold), scored_edges=len(labels), requested_edges=len(labels),
                  ids_sha256=validate_coverage(predictions['ids'], predictions['ids']))
    return result


def evaluate_complete_legacy(loader, ids, model, data, device, args, threshold):
    from train_util import (compute_grouped_binary_metrics, compute_time_grouped_binary_metrics,
                            compute_ranking_metrics)
    pred = collect_complete_predictions(model, loader, ids, data, device, args, strip_edge_ids=True)
    metrics = complete_metrics(pred, threshold)
    grouped = compute_grouped_binary_metrics(pred['logits'], pred['labels'], pred['groups'], metrics['threshold']) if pred['groups'] is not None else {}
    if pred['times'] is not None:
        grouped.update(compute_time_grouped_binary_metrics(pred['logits'], pred['labels'], pred['times'], metrics['threshold']))
    grouped.update({f'overall/{key}': value for key, value in compute_ranking_metrics(pred['logits'], pred['labels']).items()})
    return (*(metrics[key] for key in ('F1', 'PR-AUC', 'AP', 'Precision', 'Recall')), grouped, metrics['threshold'])
