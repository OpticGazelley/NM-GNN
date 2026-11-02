import copy
from datetime import timedelta
import itertools
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'genagg'), str(ROOT / 'tests')]

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from distributed_training import GlobalBatchShard, global_weighted_loss, train_distributed, wrap_distributed_model
from test_aml_architecture import make_args, make_graph, make_model
from train_multi_gpu import configure_nccl_compatibility, parse_gpu_ids, selected_visibility


def distributed_worker(rank, world_size, directory, reverse, backend, single_node=False):
    torch.set_num_threads(1)
    device = torch.device('cuda', rank) if backend == 'nccl' else torch.device('cpu')
    if backend == 'nccl':
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, init_method=f'file://{directory}/rendezvous', rank=rank,
                            world_size=world_size, timeout=timedelta(seconds=120))
    try:
        if world_size > 1:
            try:
                wrap_distributed_model(torch.nn.Linear(2, rank + 2), device, None)
            except ValueError as exc:
                assert 'layouts differ' in str(exc)
            else:
                raise AssertionError('Different model layouts were not rejected')
        initial = torch.nn.Linear(2, 2)
        with torch.no_grad():
            initial.weight.fill_(rank + 1)
            initial.bias.fill_(rank + 1)
        synchronized = wrap_distributed_model(initial, device, None)
        expected = torch.ones_like(synchronized.module.weight)
        torch.testing.assert_close(synchronized.module.weight, expected)
        assert not synchronized.broadcast_buffers
        # Match a single global weighted CE even when ranks have unequal label counts.
        for dummy_tail in (False, True):
            torch.manual_seed(7)
            linear = torch.nn.Linear(2, 2).to(device)
            reference = copy.deepcopy(linear)
            ddp = DistributedDataParallel(linear, device_ids=[rank] if backend == 'nccl' else None)
            x = torch.tensor([[1., 2.], [3., -1.], [-1., 4.]], device=device)
            y = torch.tensor([0, 0, 1], device=device)
            weights = torch.tensor([1., 6.], device=device)
            ids = torch.arange(3, device=device) if world_size == 1 or (dummy_tail and rank == 0) else (
                torch.tensor([], device=device, dtype=torch.long) if dummy_tail else torch.arange(rank, 3, world_size, device=device))
            logits = ddp(x[ids]) if ids.numel() else ddp(x[:1])[:0]
            loss, _ = global_weighted_loss(logits, y[ids], weights)
            loss.backward()
            torch.nn.functional.cross_entropy(reference(x), y, weight=weights).backward()
            for parameter, golden in zip(linear.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter.grad, golden.grad, rtol=1e-5, atol=1e-6)
        args = make_args(dual=True, conditioned=True, reverse=reverse)
        args.batch_size, args.seed, args.num_neighs = 4, 42, [-1, -1]
        args.ego, args.ports, args.ports_batch = False, False, False
        args.save_model, args.unique_name = True, 'run'
        args.ddp_world_size, args.gradient_sync = world_size, 'gloo'
        graph = make_graph(reverse)
        if single_node:
            graph.x = graph.x[:1]
            graph.edge_index.zero_()
            graph.simp_edge_batch.zero_()
        train_ids = torch.tensor([0, 2, 4, 5, 6])
        val_ids = torch.tensor([1])  # One rank has no validation batches.
        test_ids = torch.arange(7)
        config = SimpleNamespace(epochs=2, lr=0.0003, w_ce1=1., w_ce2=6.)
        torch.manual_seed(42)
        result = train_distributed(make_model(args), [graph, graph.clone(), graph.clone()],
                                   [train_ids, val_ids, test_ids], args, config, device,
                                   {'paths': {'model_to_save': directory}})
        assert result['test']['scored_edges'] == 7, result
        gathered = [None] * world_size
        dist.all_gather_object(gathered, result)
        assert all(item == gathered[0] for item in gathered)
        if rank == 0:
            checkpoints = list((Path(directory) / 'run').glob('epoch_*.tar'))
            assert 1 <= len(checkpoints) <= 2
            checkpoint = torch.load(checkpoints[0], map_location='cpu', weights_only=True)
            assert not any(key.startswith('module.') for key in checkpoint['model_state_dict'])
            fresh = make_model(args)
            fresh.load_state_dict(checkpoint['model_state_dict'], strict=True)
    finally:
        dist.destroy_process_group()


class DistributedTests(unittest.TestCase):
    def run_launcher(self, command, env, cwd=ROOT):
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            output, _ = process.communicate(timeout=100)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                output, _ = process.communicate()
            self.fail('Distributed launcher timed out:\n' + output)
        self.assertEqual(process.returncode, 0, output)
        return output

    def test_gpu_selection_and_inherited_mask(self):
        self.assertEqual(parse_gpu_ids(['0', '2']), [0, 2])
        self.assertEqual(parse_gpu_ids(['0,2']), [0, 2])
        self.assertEqual(selected_visibility([0, 2], 4), '0,2')
        self.assertEqual(selected_visibility([1, 0], 2, '2,3'), '3,2')
        for values in (['0', '0'], ['-1'], ['0,a']):
            with self.assertRaises(ValueError):
                parse_gpu_ids(values)
        with self.assertRaises(ValueError):
            selected_visibility([2], 2)
        environment = {'CUDA_VISIBLE_DEVICES': '0,2', 'NCCL_P2P_DISABLE': '0'}
        configure_nccl_compatibility(environment)
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '0,2')
        self.assertEqual(environment['NCCL_P2P_DISABLE'], '1')
        self.assertEqual(environment['NCCL_CUMEM_HOST_ENABLE'], '0')

    def test_global_batch_shards_have_no_duplicate_real_seeds(self):
        for size, batch, world in itertools.product((1, 5, 8, 11, 17), (4, 7), (2, 3, 4)):
            samplers = [GlobalBatchShard(size, batch, rank, world, 42) for rank in range(world)]
            for epoch in (0, 1):
                for sampler in samplers:
                    sampler.set_epoch(epoch)
                shards = [list(sampler) for sampler in samplers]
                self.assertEqual(len({len(value) for value in shards}), 1)
                seen = []
                for step in range(len(shards[0])):
                    seeds = [item for rank in range(world) if not samplers[rank].is_dummy(step)
                             for item in shards[rank][step]]
                    self.assertEqual(len(seeds), min(batch, size - step * batch))
                    self.assertEqual(len(set(seeds)), len(seeds))
                    seen.extend(seeds)
                self.assertEqual(sorted(seen), list(range(size)))

    def test_two_rank_cpu_training_and_validation(self):
        for reverse, single_node in ((False, False), (True, False), (False, True)):
            with self.subTest(reverse=reverse, single_node=single_node), tempfile.TemporaryDirectory() as directory:
                mp.spawn(distributed_worker, args=(2, directory, reverse, 'gloo', single_node), nprocs=2, join=True)

    @unittest.skipUnless(torch.cuda.is_available() and dist.is_nccl_available(), 'CUDA/NCCL unavailable')
    def test_single_rank_nccl_training(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=(1, directory, True, 'nccl'), nprocs=1, join=True)

    @unittest.skipUnless(torch.cuda.is_available() and dist.is_nccl_available(), 'CUDA/NCCL unavailable')
    def test_communication_check_needs_no_dataset_or_config(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT / 'genagg') + os.pathsep + env.get('PYTHONPATH', '')
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        command = [sys.executable, str(ROOT / 'train_multi_gpu.py'), '--gpus', '0', '--nccl_compat',
                   '--data', 'Small_HI', '--model', 'gin', '--check_distributed']
        with tempfile.TemporaryDirectory() as directory:
            output = self.run_launcher(command, env, cwd=directory)
            self.assertIn('DISTRIBUTED CHECK PASSED', output)
            self.assertEqual(list(Path(directory).iterdir()), [])

    @unittest.skipUnless(torch.cuda.is_available() and dist.is_nccl_available(), 'CUDA/NCCL unavailable')
    def test_selected_gnn_diagnostic(self):
        env = os.environ.copy()
        env['PYTHONPATH'] = str(ROOT / 'genagg') + os.pathsep + env.get('PYTHONPATH', '')
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        command = [sys.executable, str(ROOT / 'train_multi_gpu.py'), '--gpus', '0', '--nccl_compat',
                   '--data', 'Small_HI', '--model', 'gin', '--emlps', '--reverse_mp', '--ego',
                   '--flatten_edges', '--edge_agg_type', 'gin', '--aml_categorical_features',
                   '--aml_conditioned_edge_agg', '--check_model_distributed', '--lr', '0.0003', '--seed', '42']
        output = self.run_launcher(command, env)
        self.assertIn('DDP construction complete', output)
        self.assertIn('MODEL DDP CHECK PASSED', output)
        self.assertNotIn('Raw AML CSV detected', output)

    @unittest.skipUnless(torch.cuda.is_available() and dist.is_nccl_available(), 'CUDA/NCCL unavailable')
    def test_gpu_launcher_end_to_end(self):
        import pandas as pd
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
            paths = {'aml_data': str(root / 'data'), 'model_to_save': str(root / 'checkpoints')}
            config_file = root / 'data_config.json'
            config_file.write_text(json.dumps({'paths': paths}), encoding='utf-8')
            command = [sys.executable, str(ROOT / 'train_multi_gpu.py'), '--gpus', '0',
                       '--data', 'Small_HI', '--model', 'gin', '--task', 'edge_class',
                       '--emlps', '--reverse_mp', '--ego', '--flatten_edges', '--edge_agg_type', 'gin',
                       '--aml_categorical_features', '--aml_dual_endpoint_update', '--aml_conditioned_edge_agg',
                       '--batch_size', '8', '--num_neighs', '-1', '-1', '--n_epochs', '1', '--seed', '42',
                       '--lr', '0.0003', '--run_name', 'launcher_test', '--testing', '--save_model',
                       '--data_config', str(config_file), '--aml_history_cache_dir', str(root / 'cache')]
            env = os.environ.copy()
            env['PYTHONPATH'] = str(ROOT / 'genagg') + os.pathsep + env.get('PYTHONPATH', '')
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            output = self.run_launcher(command, env)
            self.assertIn('data loaded; synchronizing on CPU', output)
            run_dir = root / 'checkpoints' / 'launcher_test'
            self.assertTrue((run_dir / 'results.json').exists(), output)
            self.assertTrue((run_dir / 'training.log').exists())
            saved_config = json.loads((run_dir / 'run_config.json').read_text())
            self.assertEqual(saved_config['global_batch_size'], 8)
            self.assertEqual(saved_config['model']['lr'], 0.0003)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
