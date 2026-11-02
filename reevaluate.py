"""Select a saved AML checkpoint on complete validation seeds, then test once."""

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data_loading import get_data
from distributed_training import make_loader
from full_evaluation import (AML_DATASETS, PROTOCOL, collect_complete_predictions,
                             complete_metrics, complete_seed_edges)
from training import get_model
from train_util import extract_param, get_aml_feature_config, _normalize_aml_feature_config
from util import create_parser, set_seed, validate_aml_architecture_args, validate_aml_architecture_checkpoint


def parse_options(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--run_dir', type=Path)
    known, _ = pre.parse_known_args(argv)
    saved = None
    if known.run_dir is not None:
        config_path = known.run_dir / 'run_config.json'
        if not config_path.is_file():
            pre.error('run_config.json not found; use --checkpoints with the original model/feature flags instead')
        saved = json.loads(config_path.read_text(encoding='utf-8'))
    parser = create_parser()
    parser.add_argument('--run_dir', type=Path, help='Restore model/feature settings and evaluate saved epoch_*.tar files')
    parser.add_argument('--checkpoints', nargs='+', type=Path, help='Explicit candidate checkpoints from one experiment')
    parser.add_argument('--output_dir', type=Path, help='New directory; existing directories are never overwritten')
    parser.add_argument('--data_config', type=Path, help='Override dataset paths')
    parser.add_argument('--eval_seed', type=int, default=42, help='Fixed neighbor sampling seed for all compared models')
    if saved is not None:
        allowed = {action.dest for action in parser._actions}
        defaults = {key: value for key, value in saved['args'].items() if key in allowed}
        # Physical GPU allocation is deliberately chosen for this evaluation.
        defaults.pop('device', None)
        defaults.pop('data_config', None)
        parser.set_defaults(**defaults)
        for action in parser._actions:
            if action.dest in ('data', 'model'):
                action.required = False
    args = parser.parse_args(argv)
    if args.data not in AML_DATASETS or args.model != 'gin' or args.task != 'edge_class':
        parser.error('Complete reevaluation currently supports IBM AML GIN edge classification')
    if args.batch_size < 1 or args.reverse_mp_lp or args.adamm_hetero:
        parser.error('Require positive batch_size and the ordinary AML edge-classification model')
    if args.inference or args.finetune:
        parser.error('reevaluate.py already loads checkpoints; do not pass --inference/--finetune')
    validate_aml_architecture_args(args)
    args.num_neighs = [int(value) for value in args.num_neighs]
    candidates = args.checkpoints
    if not candidates and args.run_dir is not None:
        candidates = sorted(args.run_dir.glob('epoch_*.tar'),
                            key=lambda path: int(path.stem.split('_')[-1]))
    if not candidates or any(not path.is_file() for path in candidates):
        parser.error('Supply existing --checkpoints files or a --run_dir containing epoch_*.tar')
    candidates = list(dict.fromkeys(path.resolve() for path in candidates))
    return args, saved, candidates


def load_checked_checkpoint(path, model, args):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    validate_aml_architecture_checkpoint(checkpoint, args)
    features = checkpoint.get('aml_feature_config')
    if features is not None and _normalize_aml_feature_config(features) != get_aml_feature_config(args):
        raise ValueError(f'Checkpoint feature settings differ from evaluation settings: {path}')
    if features is None:
        logging.warning('Checkpoint has no feature metadata; verify original CLI settings: %s', path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return checkpoint


def log_metrics(split, metrics):
    for key in ('F1', 'Precision', 'Recall', 'PR-AUC', 'AP'):
        logging.info('%s %s: %.4f', split, key, metrics[key])
    logging.info('%s threshold: %.6f', split, metrics['threshold'])
    logging.info('%s scored/requested edges: %d/%d', split, metrics['scored_edges'], metrics['requested_edges'])


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run_reevaluation(args, saved, candidates):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is unavailable; pass --device cpu explicitly for CPU evaluation')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    output = args.output_dir or Path('evaluations') / datetime.now().strftime('complete_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / 'evaluation.log')], force=True)
    if args.data_config is not None or saved is None:
        with (args.data_config or Path('data_config.json')).open(encoding='utf-8') as stream:
            data_config = json.load(stream)
    else:
        data_config = saved['data']
    keys = ('lr', 'n_hidden', 'n_gnn_layers', 'final_dropout', 'w_ce1', 'w_ce2')
    values = {key: saved['model'][key] if saved and key in saved['model'] else extract_param(key, args)
              for key in keys}
    config = SimpleNamespace(**values)
    set_seed(args.seed)
    _, val_data, test_data, _, val_ids, test_ids = get_data(args, data_config)
    val_ids, test_ids = val_ids.cpu(), test_ids.cpu()
    loader = make_loader(val_data, val_ids, args, batch_size=args.batch_size)
    sample = next(iter(loader))
    complete_seed_edges(sample, val_data, val_ids)
    if args.ports and args.ports_batch:
        from data_util import assign_ports_with_cpp
        from torch_geometric.data import HeteroData
        assign_ports_with_cpp(sample, process_batch=isinstance(sample, HeteroData))
    model = get_model(sample, config, args).to(device)
    del sample, loader
    manifest = {'protocol': PROTOCOL, 'args': vars(args), 'model': values, 'data': data_config,
                'candidates': [str(path) for path in candidates],
                'torch': torch.__version__,
                'selection': 'highest complete-validation F1; threshold from validation only',
                'scope': 'only supplied/saved checkpoints, not every training epoch',
                'context': 'sampled static graph plus missing forward seeds; sampled reverse edges unchanged'}
    (output / 'evaluation_config.json').write_text(json.dumps(manifest, default=str, indent=2), encoding='utf-8')
    logging.info('Protocol %s; %d candidate checkpoints; output %s', PROTOCOL, len(candidates), output.resolve())
    rows, best = [], None
    for path in candidates:
        checkpoint = load_checked_checkpoint(path, model, args)
        set_seed(args.eval_seed)
        loader = make_loader(val_data, val_ids, args, batch_size=args.batch_size)
        predictions = collect_complete_predictions(model, loader, val_ids, val_data, device, args)
        metrics = complete_metrics(predictions)
        row = {'checkpoint': str(path), 'sha256': checkpoint_digest(path),
               'saved_epoch': checkpoint.get('epoch'), 'validation': metrics}
        rows.append(row)
        logging.info('Validation candidate: %s (saved epoch %s)', path, row['saved_epoch'])
        log_metrics('Val', metrics)
        if best is None or metrics['F1'] > best['validation']['F1']:
            best = row
            np.savez_compressed(output / 'validation_predictions.npz',
                                **{key: value for key, value in predictions.items() if value is not None})
        (output / 'validation_candidates.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        del checkpoint, predictions, loader
    selected = Path(best['checkpoint'])
    if checkpoint_digest(selected) != best['sha256']:
        raise RuntimeError('Selected checkpoint changed during evaluation')
    load_checked_checkpoint(selected, model, args)
    set_seed(args.eval_seed + 1)
    loader = make_loader(test_data, test_ids, args, batch_size=args.batch_size)
    predictions = collect_complete_predictions(model, loader, test_ids, test_data, device, args)
    test = complete_metrics(predictions, threshold=best['validation']['threshold'])
    np.savez_compressed(output / 'test_predictions.npz',
                        **{key: value for key, value in predictions.items() if value is not None})
    result = {'protocol': PROTOCOL, 'selected': best, 'test': test}
    (output / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    logging.info('Selected checkpoint: %s; saved epoch %s', selected, best['saved_epoch'])
    log_metrics('Test', test)
    logging.info('Evaluation saved in %s', output.resolve())
    return result


if __name__ == '__main__':
    run_reevaluation(*parse_options())
