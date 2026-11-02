import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'genagg'), str(ROOT / 'tests')]

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.data import HeteroData

from distributed_training import evaluate_distributed, make_loader
from full_evaluation import (FORWARD, collect_complete_predictions, complete_metrics,
                             complete_seed_edges, forward_store, validate_coverage)
from test_aml_architecture import make_args, make_graph, make_model
from train_util import add_arange_ids, evaluate_hetero, evaluate_homo


def full_evaluation_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{directory}/rendezvous',
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        for reverse in (False, True):
            torch.manual_seed(42)
            args = make_args(dual=True, conditioned=True, reverse=reverse)
            args.num_neighs, args.batch_size = [0, 0], 4
            args.ego = args.ports = args.ports_batch = False
            args.gradient_sync = 'gloo'
            model, graph = make_model(args), make_graph(reverse)
            for ids in (torch.tensor([1]), torch.arange(7)):
                metrics = evaluate_distributed(model, graph, ids, args, torch.device('cpu'), threshold=0.5)
                assert metrics['scored_edges'] == metrics['requested_edges'] == len(ids)
    finally:
        dist.destroy_process_group()


class FullEvaluationTests(unittest.TestCase):
    def args(self, reverse):
        args = make_args(dual=True, conditioned=True, reverse=reverse)
        args.num_neighs = [0, 0]  # Every seed is deliberately absent from neighbors.
        args.ego = args.ports = args.ports_batch = args.tqdm = False
        return args

    def test_missing_parallel_edges_keep_attributes_and_local_endpoints(self):
        for reverse in (False, True):
            graph = make_graph(reverse)
            source = forward_store(graph)
            source.edge_hist = torch.arange(21).reshape(7, 3).float()
            source.edge_group = torch.arange(7)
            source.edge_time_group = torch.arange(7) % 3
            source.timestamps = torch.arange(7).float()
            args = self.args(reverse)
            ids = torch.tensor([6, 0, 2, 4])
            batch = next(iter(make_loader(graph, ids, args, batch_size=4)))
            old = batch.clone()
            mask, inserted = complete_seed_edges(batch, graph, ids)
            self.assertEqual(inserted, 4)
            store = forward_store(batch)
            self.assertEqual(int(mask.sum()), 4)
            nodes = batch['node'].n_id if reverse else batch.n_id
            torch.testing.assert_close(nodes[store.edge_index], source.edge_index[:, store.e_id])
            for key in ('edge_attr', 'edge_cat', 'edge_hist', 'edge_group', 'edge_time_group', 'timestamps', 'y', 'simp_edge_batch'):
                torch.testing.assert_close(store[key], source[key][store.e_id])
            self.assertEqual(complete_seed_edges(batch, graph, ids)[1], 0)
            if reverse:
                for key, value in old['node', 'rev_to', 'node'].items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(batch['node', 'rev_to', 'node'][key], value)

    def test_full_predictions_cover_all_edges_and_preserve_source(self):
        for reverse in (False, True):
            args, graph = self.args(reverse), make_graph(reverse)
            original = graph.clone()
            model = make_model(args)
            ids = torch.tensor([6, 3, 2, 0, 4])
            pred = collect_complete_predictions(model, make_loader(graph, ids, args, batch_size=2),
                                                ids, graph, torch.device('cpu'), args)
            np.testing.assert_array_equal(pred['ids'], np.sort(ids.numpy()))
            np.testing.assert_array_equal(pred['labels'], forward_store(graph).y[pred['ids']].numpy())
            self.assertTrue(model.training)
            torch.testing.assert_close(forward_store(graph).edge_index, forward_store(original).edge_index)
            # Changing target labels must not change model inputs/logits.
            changed = graph.clone()
            forward_store(changed).y = 1 - forward_store(changed).y
            other = collect_complete_predictions(model, make_loader(changed, ids, args, batch_size=2),
                                                 ids, changed, torch.device('cpu'), args)
            np.testing.assert_allclose(pred['logits'], other['logits'])

    def test_legacy_evaluators_use_the_same_complete_protocol(self):
        for reverse in (False, True):
            args, graph = self.args(reverse), make_graph(reverse)
            ids, model = torch.arange(7), make_model(args)
            pred = collect_complete_predictions(model, make_loader(graph, ids, args, batch_size=3),
                                                ids, graph, torch.device('cpu'), args)
            golden = complete_metrics(pred)
            legacy = graph.clone()
            add_arange_ids([legacy])
            evaluate = evaluate_hetero if reverse else evaluate_homo
            actual = evaluate(make_loader(legacy, ids, args, batch_size=3), ids, model, legacy,
                              torch.device('cpu'), args)
            np.testing.assert_allclose(actual[:5], [golden[key] for key in ('F1', 'PR-AUC', 'AP', 'Precision', 'Recall')])
            self.assertEqual(actual[-1], golden['threshold'])

    def test_coverage_rejects_equal_count_wrong_ids_and_duplicates(self):
        for predicted in ([0, 1, 1], [0, 1, 3], [0, 1]):
            with self.assertRaises(ValueError):
                validate_coverage(predicted, [0, 1, 2])
        self.assertEqual(validate_coverage([2, 0, 1], [0, 1, 2]),
                         validate_coverage([0, 1, 2], [2, 1, 0]))

    def test_two_rank_complete_evaluation_with_empty_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(full_evaluation_worker, args=(directory,), nprocs=2, join=True)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gpu_complete_evaluation(self):
        for reverse in (False, True):
            args, graph = self.args(reverse), make_graph(reverse)
            model, ids = make_model(args), torch.arange(7)
            cpu = collect_complete_predictions(model, make_loader(graph, ids, args, batch_size=3),
                                               ids, graph, torch.device('cpu'), args)
            model.cuda()
            gpu = collect_complete_predictions(model, make_loader(graph, ids, args, batch_size=3),
                                               ids, graph, torch.device('cuda:0'), args)
            np.testing.assert_array_equal(cpu['ids'], gpu['ids'])
            np.testing.assert_allclose(cpu['logits'], gpu['logits'], rtol=1e-4, atol=1e-5)

    def test_existing_seed_edges_are_not_duplicated(self):
        args, graph = self.args(False), make_graph(False)
        args.num_neighs = [-1, -1]
        ids = torch.arange(7)
        batch = next(iter(make_loader(graph, ids, args, batch_size=7)))
        old = batch.clone()
        mask, inserted = complete_seed_edges(batch, graph, ids)
        self.assertEqual(inserted, 0)
        self.assertEqual(int(mask.sum()), 7)
        torch.testing.assert_close(batch.edge_index, old.edge_index)

    def test_cli_run_directory_selects_validation_and_saves_complete_results(self):
        import pandas as pd
        from data_loading import get_data
        from training import get_model
        from train_util import get_aml_feature_config
        from util import create_parser, get_aml_architecture_config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / 'data' / 'Small_HI'
            data_dir.mkdir(parents=True)
            rows = []
            for day in range(6):
                for i in range(12):
                    rows.append({'EdgeID': len(rows), 'from_id': i % 4, 'to_id': (i + 1) % 4,
                                 'Timestamp': day * 86400 + i * 60, 'Amount Sent': i + 1 + day,
                                 'Amount Received': i + 1 + day, 'Sent Currency': i % 2,
                                 'Received Currency': i % 2, 'Payment Format': i % 3,
                                 'Is Laundering': int(i % 5 == 0)})
            pd.DataFrame(rows).to_csv(data_dir / 'formatted_transactions.csv', index=False)
            args = create_parser().parse_args(['--data', 'Small_HI', '--model', 'gin', '--reverse_mp',
                '--emlps', '--ego', '--flatten_edges', '--edge_agg_type', 'gin',
                '--aml_semantic_features', '--aml_categorical_features', '--aml_conditioned_edge_agg',
                '--aml_dual_endpoint_update', '--batch_size', '8', '--num_neighs', '0', '0',
                '--seed', '42', '--aml_history_cache_dir', str(root / 'cache')])
            args.num_neighs = [0, 0]
            data_config = {'paths': {'aml_data': str(root / 'data')}}
            loaded = get_data(args, data_config)
            values = dict(lr=0.0003, n_hidden=8, n_gnn_layers=2, final_dropout=0., w_ce1=1., w_ce2=6.)
            sample = next(iter(make_loader(loaded[1], loaded[4], args, batch_size=8)))
            complete_seed_edges(sample, loaded[1], loaded[4])
            model = get_model(sample, SimpleNamespace(**values), args)
            run = root / 'run'
            run.mkdir()
            (run / 'run_config.json').write_text(json.dumps({'args': vars(args), 'model': values, 'data': data_config}))
            for epoch in (1, 2):
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'decision_threshold': 0.99999,
                            'aml_feature_config': get_aml_feature_config(args),
                            'aml_architecture_config': get_aml_architecture_config(args)}, run / f'epoch_{epoch}.tar')
            env = os.environ.copy()
            env['PYTHONPATH'] = str(ROOT / 'genagg') + os.pathsep + env.get('PYTHONPATH', '')
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            output = root / 'evaluation'
            command = [sys.executable, str(ROOT / 'reevaluate.py'), '--run_dir', str(run),
                       '--device', 'cpu', '--output_dir', str(output)]
            completed = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, timeout=90)
            self.assertEqual(completed.returncode, 0, completed.stdout)
            result = json.loads((output / 'results.json').read_text())
            candidates = json.loads((output / 'validation_candidates.json').read_text())
            self.assertEqual(len(candidates), 2)
            self.assertEqual(candidates[0]['validation'], candidates[1]['validation'])
            self.assertEqual(result['selected']['saved_epoch'], 1)
            self.assertEqual(result['test']['threshold'], result['selected']['validation']['threshold'])
            self.assertEqual(completed.stdout.count('Test F1:'), 1)
            self.assertEqual(result['test']['scored_edges'], len(loaded[5]))
            for split, ids in (('validation', loaded[4]), ('test', loaded[5])):
                pred = np.load(output / f'{split}_predictions.npz')
                validate_coverage(pred['ids'], ids.numpy())
            self.assertNotEqual(result['test']['threshold'], 0.99999)
            self.assertEqual(len(list(run.glob('epoch_*.tar'))), 2)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
