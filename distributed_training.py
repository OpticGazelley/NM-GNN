"""Single-node DDP training for the IBM AML GIN edge-classification experiments."""

from datetime import datetime, timedelta
import copy
import json
import logging
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import BatchNorm
from torch.utils.data import Sampler
import wandb

from data_util import assign_ports_with_cpp
from train_util import (AddEgoIds, compute_binary_metrics, compute_grouped_binary_metrics,
                        compute_time_grouped_binary_metrics, compute_ranking_metrics,
                        extract_param, save_model, select_f1_threshold)
from training import get_model
from util import create_parser, set_seed, validate_aml_architecture_args

FORWARD = ('node', 'to', 'node')


class GlobalBatchShard(Sampler):
    """Partition every global batch without dropping or reweighting real seeds.

    Ranks with no seeds in the final step run one dummy seed for DDP collectives;
    its loss and metrics are excluded. No training example is padded into loss.
    """

    def __init__(self, size, global_batch_size, rank, world_size, seed):
        if size < 1 or global_batch_size < world_size or not 0 <= rank < world_size:
            raise ValueError('Need non-empty training data and global batch_size >= world_size')
        self.size, self.global_batch_size = size, global_batch_size
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.local_batch_size = global_batch_size // world_size + int(rank < global_batch_size % world_size)
        self.offset = rank * (global_batch_size // world_size) + min(rank, global_batch_size % world_size)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def is_dummy(self, step):
        return step * self.global_batch_size + self.offset >= self.size

    def __len__(self):
        return math.ceil(self.size / self.global_batch_size)

    def __iter__(self):
        order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch))
        for step in range(len(self)):
            start = step * self.global_batch_size
            shard = order[start + self.offset:min(start + self.offset + self.local_batch_size,
                                                 start + self.global_batch_size, self.size)]
            yield shard.tolist() if shard.numel() else [int(order[start])]


def edge_store(data):
    return data[FORWARD] if isinstance(data, HeteroData) else data


def make_loader(data, ids, args, batch_size=None, batch_sampler=None):
    store = edge_store(data)
    index = store.edge_index[:, ids]
    index = (FORWARD, index) if isinstance(data, HeteroData) else index
    options = {'batch_sampler': batch_sampler} if batch_sampler is not None else {
        'batch_size': batch_size, 'shuffle': False}
    return LinkNeighborLoader(data, num_neighbors=args.num_neighs, edge_label_index=index,
                              edge_label=store.y[ids], transform=AddEgoIds() if args.ego else None,
                              **options)


def prepare_batch(batch, input_ids, args, device, dummy=False):
    store = edge_store(batch)
    seeds = input_ids[store.input_id.cpu()]
    mask = torch.isin(store.e_id.cpu(), seeds)
    if dummy:
        mask.zero_()
    identifiers = store.e_id.cpu()[mask]
    groups = store.edge_group.cpu()[mask] if 'edge_group' in store else None
    times = store.edge_time_group.cpu()[mask] if 'edge_time_group' in store else None
    if args.ports and args.ports_batch:
        assign_ports_with_cpp(batch, process_batch=isinstance(batch, HeteroData))
    batch.to(device)
    mask = mask.to(device)
    return batch, mask, edge_store(batch).y[mask], identifiers, groups, times


def global_weighted_loss(logits, labels, class_weights, group=None):
    """DDP averages gradients, so scale local summed CE by world/global weight."""
    denominator = class_weights[labels].sum().detach()
    if group is None:
        dist.all_reduce(denominator)
    else:
        cpu_denominator = denominator.cpu()
        dist.all_reduce(cpu_denominator, group=group)
        denominator = cpu_denominator.to(logits.device)
    numerator = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights, reduction='sum')
    return numerator * dist.get_world_size() / denominator.clamp_min(1), denominator


def empty_predictions():
    return {'logits': [], 'labels': [], 'ids': [], 'groups': [], 'times': []}


def append_predictions(results, logits, labels, identifiers, groups, times):
    results['logits'].append(logits.detach().cpu().numpy())
    results['labels'].append(labels.detach().cpu().numpy())
    results['ids'].append(identifiers.numpy())
    if groups is not None:
        results['groups'].append(groups.numpy())
    if times is not None:
        results['times'].append(times.numpy())


def gather_predictions(results, control_group=None):
    local = {key: np.concatenate(values) if values else None for key, values in results.items()}
    gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(local, gathered, dst=0, group=control_group)
    if gathered is None:
        return None
    merged = {}
    for key in results:
        arrays = [item[key] for item in gathered if item[key] is not None and len(item[key])]
        merged[key] = np.concatenate(arrays) if arrays else None
    return merged


def summarize_predictions(results, threshold=None):
    if results['labels'] is None:
        raise ValueError('No seed edges were scored; check data and neighbor sampling')
    order = np.argsort(results['ids'], kind='stable')
    results = {key: value[order] if value is not None else None for key, value in results.items()}
    if len(np.unique(results['ids'])) != len(results['ids']):
        raise ValueError('Duplicate seed edges in distributed metrics')
    logits, labels = results['logits'], results['labels']
    threshold = select_f1_threshold(logits, labels) if threshold is None else threshold
    values = compute_binary_metrics(logits, labels, threshold)
    metrics = dict(zip(('F1', 'PR-AUC', 'AP', 'Precision', 'Recall'), map(float, values)))
    metrics.update(threshold=float(threshold), scored_edges=len(labels))
    groups = compute_grouped_binary_metrics(logits, labels, results['groups'], threshold) if results['groups'] is not None else {}
    if results['times'] is not None:
        groups.update(compute_time_grouped_binary_metrics(logits, labels, results['times'], threshold))
    groups.update({f'overall/{key}': value for key, value in compute_ranking_metrics(logits, labels).items()})
    metrics['groups'] = groups
    return metrics


def broadcast_metrics(results, threshold=None, control_group=None, expected_ids=None):
    # Return root-side metric errors to every rank before raising, avoiding a wait.
    message = [None]
    if dist.get_rank() == 0:
        try:
            if expected_ids is not None:
                from full_evaluation import validate_coverage
                validate_coverage(results['ids'], expected_ids.cpu().numpy())
            message[0] = {'metrics': summarize_predictions(results, threshold)}
        except Exception as exc:
            message[0] = {'error': str(exc)}
    dist.broadcast_object_list(message, src=0, group=control_group)
    if 'error' in message[0]:
        raise RuntimeError(message[0]['error'])
    return message[0]['metrics']


@torch.no_grad()
def evaluate_distributed(model, data, ids, args, device, threshold=None, control_group=None):
    # Local BatchNorm statistics differ during training. Evaluate all shards
    # using rank zero's buffers, which are also the buffers saved in checkpoints.
    if getattr(args, 'gradient_sync', 'ddp') == 'gloo':
        synchronize_cpu_buffers(model, control_group)
    else:
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
    model.eval()
    local_ids = ids[dist.get_rank()::dist.get_world_size()]
    results = empty_predictions()
    if local_ids.numel():
        loader = make_loader(data, local_ids, args, batch_size=max(1, args.batch_size // dist.get_world_size()))
        for batch in loader:
            from full_evaluation import complete_seed_edges
            complete_seed_edges(batch, data, local_ids)
            batch, mask, labels, identifiers, groups, times = prepare_batch(batch, local_ids, args, device)
            append_predictions(results, model(batch)[mask], labels, identifiers, groups, times)
    metrics = broadcast_metrics(gather_predictions(results, control_group), threshold, control_group, expected_ids=ids)
    metrics['requested_edges'] = len(ids)
    model.train()
    return metrics


def report_metrics(metrics, split, epoch):
    if dist.get_rank() != 0:
        return
    for key in ('F1', 'Precision', 'Recall', 'PR-AUC', 'AP'):
        logging.info('%s %s: %.4f', split, key, metrics[key])
    if split != 'Train':
        logging.info('%s threshold: %.6f', split, metrics['threshold'])
    logging.info('%s scored/requested edges: %d/%d', split, metrics['scored_edges'], metrics['requested_edges'])
    if wandb.run is not None:
        wandb.log({f'{split}/{key}': value for key, value in metrics.items() if key != 'groups'}, step=epoch)
        wandb.log({f'group/{split}/{key}': value for key, value in metrics['groups'].items()}, step=epoch)


def startup_stage(message, device=None):
    if device is not None and device.type == 'cuda':
        torch.cuda.synchronize(device)
    print(f'[rank {dist.get_rank()}] {message}', flush=True)


def synchronize_initial_cpu_state(model, control_group):
    """Copy rank zero's CPU parameters and buffers without NCCL initialization."""
    state = [None]
    if dist.get_rank() == 0:
        state[0] = {
            'parameters': {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
            },
            'buffers': {
                name: buffer.detach().cpu().clone()
                for name, buffer in model.named_buffers()
            },
        }
    dist.broadcast_object_list(state, src=0, group=control_group)
    if dist.get_rank() == 0:
        return
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(state[0]['parameters'][name])
        for name, buffer in model.named_buffers():
            buffer.copy_(state[0]['buffers'][name])


def synchronize_cpu_buffers(model, control_group):
    """Make evaluation use rank zero's running buffers without NCCL broadcasts."""
    state = [None]
    if dist.get_rank() == 0:
        state[0] = {name: buffer.detach().cpu().clone() for name, buffer in model.named_buffers()}
    dist.broadcast_object_list(state, src=0, group=control_group)
    if dist.get_rank() == 0:
        return
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            buffer.copy_(state[0][name])


def synchronize_gradients(model, control_group):
    """Average dense gradients through one CPU/Gloo reduction per dtype."""
    if dist.get_world_size() == 1:
        return
    by_dtype = {}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        if parameter.grad.is_sparse:
            raise ValueError('Gloo gradient synchronization does not support sparse gradients')
        by_dtype.setdefault(parameter.grad.dtype, []).append(parameter)
    with torch.no_grad():
        for parameters in by_dtype.values():
            flat = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
            reduced = flat.detach().cpu()
            dist.all_reduce(reduced, group=control_group)
            reduced.div_(dist.get_world_size())
            flat.copy_(reduced.to(device=flat.device, dtype=flat.dtype))
            offset = 0
            for parameter in parameters:
                size = parameter.numel()
                parameter.grad.copy_(flat[offset:offset + size].view_as(parameter))
                offset += size


def assert_parameters_equal(model, control_group):
    local = torch.cat([parameter.detach().cpu().reshape(-1) for parameter in model.parameters()])
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local, group=control_group)
    for value in gathered[1:]:
        torch.testing.assert_close(value, gathered[0])


def wrap_distributed_model(model, device, control_group, gradient_sync='ddp'):
    # Check names/order/shapes on the CPU before initializing the NCCL reducer.
    layout = {
        'state': [(name, tuple(value.shape), tuple(value.stride()), str(value.dtype))
                  for name, value in model.state_dict().items()],
        'trainable': [name for name, parameter in model.named_parameters() if parameter.requires_grad],
    }
    layouts = [None] * dist.get_world_size()
    dist.all_gather_object(layouts, layout, group=control_group)
    if any(value != layouts[0] for value in layouts):
        raise ValueError('Model state layouts differ across ranks; aborting before DDP initialization')
    for layer in model.modules():
        if isinstance(layer, BatchNorm):
            layer.allow_single_element = True
    synchronize_initial_cpu_state(model, control_group)
    startup_stage('initial CPU model state synchronized from rank 0')
    startup_stage(f'model layout verified ({sum(p.numel() for p in model.parameters())} parameters); moving model to {device}')
    model.to(device)
    if gradient_sync == 'gloo':
        startup_stage('model transfer complete; using manual CPU/Gloo gradient synchronization', device)
        return model
    startup_stage('model transfer complete; constructing DDP', device)
    # DDP's default init_sync performs a coalesced NCCL model-state broadcast.
    # The state is already synchronized above through the CPU/Gloo control group.
    # Disable per-forward buffer broadcasts as well; rank zero's BatchNorm buffers
    # are explicitly used for evaluation and checkpointing.
    model = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
                                    find_unused_parameters=True, init_sync=False,
                                    broadcast_buffers=False)
    startup_stage('DDP construction complete', device)
    return model


def train_distributed(model, datasets, indices, args, config, device, data_config, control_group=None):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if control_group is None and dist.get_backend() == 'nccl':
        control_group = dist.new_group(backend='gloo', timeout=timedelta(hours=2))
    tr_data, val_data, te_data = datasets
    tr_ids, val_ids, te_ids = [ids.cpu() for ids in indices]
    gradient_sync = getattr(args, 'gradient_sync', 'ddp')
    ddp = wrap_distributed_model(model, device, control_group, gradient_sync)
    optimizer = torch.optim.Adam(ddp.parameters(), lr=config.lr)
    weights = torch.tensor([config.w_ce1, config.w_ce2], device=device)
    sampler = GlobalBatchShard(len(tr_ids), args.batch_size, rank, world_size, args.seed)
    loader = make_loader(tr_data, tr_ids, args, batch_sampler=sampler)
    best_f1, best_epoch, best_threshold, best_state = -1., -1, 0.5, None
    set_seed(args.seed + rank)
    for epoch in range(config.epochs):
        epoch_started = time.perf_counter()
        sampler.set_epoch(epoch)
        ddp.train()
        if rank == 0:
            logging.info('---------- Epoch %d ----------', epoch)
        results = empty_predictions()
        for step, batch in enumerate(loader):
            batch, mask, labels, identifiers, groups, times = prepare_batch(
                batch, tr_ids, args, device, dummy=sampler.is_dummy(step))
            optimizer.zero_grad(set_to_none=True)
            logits = ddp(batch)[mask]
            loss, denominator = global_weighted_loss(
                logits, labels, weights, control_group if gradient_sync == 'gloo' else None
            )
            loss.backward()
            if gradient_sync == 'gloo':
                synchronize_gradients(ddp, control_group)
            if denominator.item() > 0:
                optimizer.step()
            append_predictions(results, logits, labels, identifiers, groups, times)
        train_metrics = broadcast_metrics(gather_predictions(results, control_group), threshold=0.5,
                                          control_group=control_group)
        train_metrics['requested_edges'] = len(tr_ids)
        report_metrics(train_metrics, 'Train', epoch)
        val_metrics = evaluate_distributed(model, val_data, val_ids, args, device, control_group=control_group)
        report_metrics(val_metrics, 'Val', epoch)
        if val_metrics['F1'] > best_f1:
            best_f1, best_epoch, best_threshold = val_metrics['F1'], epoch, val_metrics['threshold']
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if rank == 0 and args.save_model:
                # Save the underlying model, without DDP's module. key prefix.
                save_model(model, optimizer, epoch, args, data_config, best_threshold)
        dist.barrier(group=control_group)
        if rank == 0:
            logging.info('Epoch wall time: %.2fs', time.perf_counter() - epoch_started)
    model.load_state_dict(best_state)
    test_metrics = evaluate_distributed(model, te_data, te_ids, args, device, best_threshold, control_group)
    if rank == 0:
        logging.info('Best validation epoch: %d', best_epoch)
        logging.info('Validation-selected threshold: %.6f', best_threshold)
    report_metrics(test_metrics, 'Test', config.epochs)
    return {'best_epoch': best_epoch, 'threshold': best_threshold, 'test': test_metrics}


def validate_distributed_args(args, world_size):
    validate_aml_architecture_args(args)
    if args.task != 'edge_class' or args.data not in ('Small_HI', 'Small_LI', 'Medium_HI', 'Medium_LI'):
        raise ValueError('DDP currently supports IBM AML edge classification')
    if args.model != 'gin' or args.node_agg_type != 'sum':
        raise ValueError('DDP currently supports --model gin --node_agg_type sum')
    if args.flatten_edges and args.edge_agg_type not in ('gin', 'sum'):
        raise ValueError('DDP flattened-edge aggregation supports gin or sum')
    if args.inference or args.finetune or args.reverse_mp_lp or args.adamm_hetero:
        raise ValueError('Use the single-GPU entry for inference/finetune; DDP currently starts fresh AML training')
    if args.batch_size < world_size or args.n_epochs < 1:
        raise ValueError('Require batch_size >= GPU count and n_epochs >= 1')
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError('--lr must be finite and positive')
    if args.run_name and (args.run_name in ('.', '..') or '/' in args.run_name or '\\' in args.run_name):
        raise ValueError('--run_name must be a single directory name')


def communication_check(device, control_group):
    """Exercise NCCL collectives and DDP without loading data or creating a run."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    for count in (1, 65536):
        probe = torch.full((count,), float(rank + 1), device=device)
        dist.all_reduce(probe)
        torch.testing.assert_close(probe, torch.full_like(probe, world_size * (world_size + 1) / 2))
    model = DistributedDataParallel(torch.nn.Linear(16, 2).to(device), device_ids=[device.index])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        model(torch.full((4, 16), float(rank + 1), device=device)).square().mean().backward()
        optimizer.step()
    weights = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    copies = [torch.empty_like(weights) for _ in range(world_size)]
    dist.all_gather(copies, weights)
    for copied in copies:
        torch.testing.assert_close(copied, weights)
    dist.barrier(group=control_group)
    print(f'[rank {rank}] DISTRIBUTED CHECK PASSED: GPU collectives and synchronized optimizer steps', flush=True)


def model_config_values(args):
    values = {key: extract_param(key, args) for key in (
        'lr', 'n_hidden', 'n_gnn_layers', 'final_dropout', 'w_ce1', 'w_ce2')}
    if args.lr is not None:
        values['lr'] = args.lr
    values['epochs'] = args.n_epochs
    return values


def model_distributed_check(args, device, control_group):
    """Check the selected GNN architecture on a small synthetic multigraph."""
    if args.ports or args.ports_batch or args.tds:
        raise ValueError('The synthetic model check does not support ports or time-delta features')
    from data_loading import _resolve_aml_feature_flags
    from history_features import HISTORY_FEATURE_NAMES
    check_args = copy.copy(args)
    _resolve_aml_feature_flags(check_args)
    check_args.edge_cat_cardinalities = [16, 16, 7] if check_args.aml_use_categorical else []
    check_args.edge_hist_dim = (len(HISTORY_FEATURE_NAMES) if check_args.aml_use_history_gaps else 6) if check_args.aml_use_history else 0
    shared = [model_config_values(args) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(shared, src=0, group=control_group)
    config = SimpleNamespace(**shared[0])
    set_seed(args.seed if args.seed != 1 else 42)
    edge_ids = torch.arange(64)
    source = edge_ids % 8
    edge_dim = (2 if check_args.aml_use_semantic else 4) + (3 if check_args.aml_use_transaction_indicators else 0)
    graph = Data(x=torch.ones(16, 2 if args.ego else 1),
                 edge_index=torch.stack([source, source + 8]),
                 edge_attr=torch.randn(64, edge_dim), simp_edge_batch=source,
                 y=(edge_ids % 5 == 0).long())
    if check_args.aml_use_categorical:
        graph.edge_cat = torch.stack([edge_ids % 16, (edge_ids + 1) % 16, (edge_ids // 8) % 7], dim=1)
    if check_args.aml_use_history:
        graph.edge_hist = torch.randn(64, check_args.edge_hist_dim)
    if args.reverse_mp:
        hetero = HeteroData()
        hetero['node'].x = graph.x
        for relation in ('to', 'rev_to'):
            for key in ('edge_index', 'edge_attr', 'simp_edge_batch', 'edge_cat', 'edge_hist'):
                if key in graph:
                    hetero['node', relation, 'node'][key] = graph[key].clone()
        hetero['node', 'rev_to', 'node'].edge_index = graph.edge_index.flip(0)
        hetero[FORWARD].y = graph.y
        graph = hetero
    startup_stage('building selected GNN on synthetic graph: 16 nodes, 64 edges')
    model = get_model(graph, config, check_args)
    ddp = wrap_distributed_model(model, device, control_group, args.gradient_sync)
    batch = graph.to(device)
    optimizer = torch.optim.Adam(ddp.parameters(), lr=config.lr)
    weights = torch.tensor([config.w_ce1, config.w_ce2], device=device)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        startup_stage(f'GNN check step {step}: forward start')
        logits = ddp(batch)
        startup_stage(f'GNN check step {step}: forward complete; backward start', device)
        loss, _ = global_weighted_loss(
            logits, edge_store(batch).y, weights,
            control_group if args.gradient_sync == 'gloo' else None,
        )
        loss.backward()
        if args.gradient_sync == 'gloo':
            synchronize_gradients(ddp, control_group)
        startup_stage(f'GNN check step {step}: backward complete; optimizer start', device)
        optimizer.step()
        startup_stage(f'GNN check step {step}: optimizer complete', device)
        if not torch.isfinite(loss).item():
            raise ValueError('Non-finite loss during the model check')
    if args.gradient_sync == 'gloo':
        assert_parameters_equal(model, control_group)
    else:
        parameters = torch.cat([value.detach().flatten() for value in model.parameters()])
        reference = parameters.clone()
        dist.broadcast(reference, src=0)
        torch.testing.assert_close(parameters, reference)
    dist.barrier(group=control_group)
    startup_stage('MODEL DDP CHECK PASSED: selected GNN forward/backward and optimizer steps')


def main():
    parser = create_parser()
    parser.add_argument('--lr', type=float, help='Override the configured learning rate for this DDP run')
    parser.add_argument('--run_name', help='Unique run directory name (default: timestamp)')
    parser.add_argument('--checkpoint_dir', help='Root directory for this run, overriding model_to_save')
    parser.add_argument('--data_config', default='data_config.json', help='Data/path config for this DDP run')
    parser.add_argument('--gradient_sync', choices=('ddp', 'gloo'), default='ddp',
                        help='Gradient synchronization backend: native DDP/NCCL or CPU/Gloo compatibility mode')
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument('--check_distributed', action='store_true', help='Only test communication and a small linear model')
    checks.add_argument('--check_model_distributed', action='store_true', help='Test the selected GNN on synthetic data without loading the AML dataset')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        parser.error('Start with python train_multi_gpu.py --gpus ...')
    world_size, rank, local_rank = (int(os.environ[key]) for key in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK'))
    try:
        validate_distributed_args(args, world_size)
    except ValueError as exc:
        parser.error(str(exc))
    args.num_neighs = [int(value) for value in args.num_neighs]
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    timeout = timedelta(seconds=120) if args.check_distributed or args.check_model_distributed else timedelta(hours=2)
    print(f'[rank {rank}] pid={os.getpid()} local_device={device} GPU={torch.cuda.get_device_name(device)}; initializing NCCL', flush=True)
    dist.init_process_group('nccl', device_id=device, timeout=timeout)
    try:
        # Host-side preprocessing and Python objects must not occupy a GPU with
        # NCCL wait kernels while another rank is busy reading the CSV/cache.
        control_group = dist.new_group(backend='gloo', timeout=timeout)
        if args.check_distributed:
            communication_check(device, control_group)
            return
        if args.check_model_distributed:
            model_distributed_check(args, device, control_group)
            return
        probe = torch.ones(1, device=device)
        dist.all_reduce(probe)
        torch.testing.assert_close(probe, torch.full_like(probe, world_size))
        print(f'[rank {rank}] communication ready; loading data', flush=True)
        shared = [None]
        if rank == 0:
            with open(args.data_config, encoding='utf-8') as stream:
                data_config = json.load(stream)
            values = model_config_values(args)
            seed = int(torch.randint(2, 256000, ()).item()) if args.seed == 1 else args.seed
            run_name = args.run_name or datetime.now().strftime('ddp_%Y%m%d_%H%M%S_%f')
            if args.checkpoint_dir:
                data_config['paths']['model_to_save'] = args.checkpoint_dir
            shared[0] = (data_config, values, seed, run_name)
        dist.broadcast_object_list(shared, src=0, group=control_group)
        data_config, values, args.seed, args.unique_name = shared[0]
        config = SimpleNamespace(**values)
        args.ddp_world_size = world_size
        run_dir = Path(data_config['paths']['model_to_save']) / args.unique_name
        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=False)
            logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] %(message)s',
                                handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / 'training.log')])
            with open(run_dir / 'run_config.json', 'w', encoding='utf-8') as stream:
                json.dump({'args': vars(args), 'model': values, 'data': data_config,
                           'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                           'nccl_environment': {key: value for key, value in os.environ.items() if key.startswith('NCCL_')},
                           'global_batch_size': args.batch_size, 'world_size': world_size}, stream, indent=2)
            logging.info('DDP GPUs=%s, global batch=%d, per-rank batches=%s, lr=%s',
                         os.environ.get('CUDA_VISIBLE_DEVICES'), args.batch_size,
                         [args.batch_size // world_size + int(i < args.batch_size % world_size) for i in range(world_size)], config.lr)
            logging.info('Run directory: %s', run_dir.resolve())
            wandb.init(mode='disabled' if args.testing else 'online', project='project_name',
                       name=args.unique_name, config={**values, 'world_size': world_size, 'batch_size': args.batch_size})
        else:
            logging.disable(logging.CRITICAL)
        set_seed(args.seed)
        from data_loading import get_data
        # Rank zero builds any missing history cache before other ranks read it.
        if rank == 0:
            loaded = get_data(args, data_config)
        dist.barrier(group=control_group)
        if rank != 0:
            loaded = get_data(args, data_config)
        print(f'[rank {rank}] data loaded; synchronizing on CPU', flush=True)
        dist.barrier(group=control_group)
        datasets, indices = loaded[:3], loaded[3:]
        sample = [None]
        if rank == 0:
            startup_stage('creating model sample batch on CPU')
            loader = make_loader(datasets[0], indices[0].cpu()[:min(256, len(indices[0]))], args, batch_size=256)
            sample[0] = next(iter(loader))
            if args.ports and args.ports_batch:
                assign_ports_with_cpp(sample[0], process_batch=isinstance(sample[0], HeteroData))
        dist.broadcast_object_list(sample, src=0, group=control_group)
        startup_stage('sample batch received; building model on CPU')
        model = get_model(sample[0], config, args)
        startup_stage('CPU model built; preparing distributed training')
        result = train_distributed(model, datasets, indices, args, config, device, data_config, control_group)
        if rank == 0:
            with open(run_dir / 'results.json', 'w', encoding='utf-8') as stream:
                json.dump(result, stream, indent=2)
    finally:
        if rank == 0 and wandb.run is not None:
            wandb.finish()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
