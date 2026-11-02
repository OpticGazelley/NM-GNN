import copy
import itertools
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'genagg')]

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData

from models import GnnHelper, MultiEdgeAggModule, MultiMPNN
from conditioned_reference import original_conditioned_forward
from util import (create_parser, get_aml_architecture_config,
                  validate_aml_architecture_args, validate_aml_architecture_checkpoint)


def make_args(dual=False, conditioned=False, reverse=False):
    return SimpleNamespace(
        data='Small_HI', model='gin', task='edge_class', emlps=True,
        node_agg_type='sum', edge_agg_type='gin', flatten_edges=True,
        reverse_mp=reverse, reverse_mp_lp=False, aml_use_categorical=True,
        aml_categorical_features=True, aml_use_history=False,
        edge_cat_cardinalities=[3, 3, 5],
        aml_dual_endpoint_update=dual, aml_conditioned_edge_agg=conditioned,
    )


def make_graph(reverse=False, empty_reverse=False):
    graph = Data(
        x=torch.tensor([[1., 0.], [1., 1.], [1., 0.], [1., 1.]]),
        edge_index=torch.tensor([[0, 0, 0, 2, 2, 1, 3], [1, 1, 1, 3, 3, 0, 0]]),
        edge_attr=torch.tensor([[1., 0.], [-1., 2.], [2., 1.], [0., 3.],
                                [1., 2.], [2., 0.], [0., 1.]]),
        edge_cat=torch.tensor([[0, 1, 0], [1, 1, 1], [0, 2, 1], [2, 0, 0],
                               [2, 1, 2], [1, 0, 4], [0, 0, 1]]),
        simp_edge_batch=torch.tensor([90, 90, 90, 15, 15, 32, 7]),
        y=torch.tensor([0, 1, 0, 0, 1, 0, 1]),
    )
    if not reverse:
        return graph
    hetero = HeteroData()
    hetero['node'].x = graph.x
    for key in ('edge_index', 'edge_attr', 'edge_cat', 'simp_edge_batch', 'y'):
        hetero['node', 'to', 'node'][key] = graph[key]
    # Reverse neighbor sampling need not return the same edges, order, or size.
    selection = torch.tensor([], dtype=torch.long) if empty_reverse else torch.tensor([4, 0, 6, 1, 3])
    hetero['node', 'rev_to', 'node'].edge_index = graph.edge_index.flip(0)[:, selection]
    for key in ('edge_attr', 'edge_cat', 'simp_edge_batch'):
        hetero['node', 'rev_to', 'node'][key] = graph[key][selection]
    return hetero


def make_model(args):
    return MultiMPNN(2, 2, n_hidden=8, edge_updates=True, edge_dim=2,
                     final_dropout=0., args=args)


class ConditionedAggregationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.graph = make_graph()
        self.module = MultiEdgeAggModule(2, 'sum', conditioned_by_payment=True,
                                         num_payment_formats=5)

    def call(self, graph, module=None):
        return (module or self.module)(graph.edge_index, graph.edge_attr,
                                      graph.simp_edge_batch, None, graph.edge_cat)

    def test_query_selects_other_payment_group(self):
        graph = self.graph
        graph.edge_index = graph.edge_index[:, :2]
        graph.edge_attr = graph.edge_attr[:2]
        graph.edge_cat = graph.edge_cat[:2]
        graph.simp_edge_batch = graph.simp_edge_batch[:2]
        with torch.no_grad():
            self.module.context_query.weight.copy_(torch.tensor([[1., 0.]]))
            self.module.context_key.weight.copy_(torch.tensor([[0., 1.]]))
            self.module.context_gate[0].weight.zero_()
            self.module.context_gate[0].bias.zero_()
        _, pair, inverse, context = self.call(graph)
        weights = torch.softmax(torch.tensor([[0., 2.], [0., -2.]]), dim=1)
        selected = weights @ graph.edge_attr
        torch.testing.assert_close(pair, graph.edge_attr.sum(0, keepdim=True))
        torch.testing.assert_close(context, (pair[inverse] + selected) / 2)
        self.assertFalse(torch.allclose(context[0], context[1]))
        # The first transaction attends more to a different payment format.
        self.assertGreater(weights[0, 1].item(), weights[0, 0].item())

    def test_single_format_and_singleton_equal_original_sum(self):
        self.graph.edge_cat[:, 2] = 4
        _, pair, inverse, context = self.call(self.graph)
        torch.testing.assert_close(context, pair[inverse], rtol=0, atol=0)
        expected = self.graph.edge_attr[:3].sum(0)
        torch.testing.assert_close(context[0], expected)

    def test_edge_permutation_and_pair_isolation(self):
        original = self.call(self.graph)
        perm = torch.tensor([4, 2, 0, 6, 3, 1, 5])
        permuted = self.graph.clone()
        permuted.edge_index = permuted.edge_index[:, perm]
        for key in ('edge_attr', 'edge_cat', 'simp_edge_batch'):
            permuted[key] = permuted[key][perm]
        actual = self.call(permuted)
        torch.testing.assert_close(original[0], actual[0])
        torch.testing.assert_close(original[1], actual[1])
        torch.testing.assert_close(original[3][perm], actual[3])
        changed = self.graph.clone()
        changed.edge_attr[3:] += 100
        torch.testing.assert_close(original[3][:3], self.call(changed)[3][:3])

    def test_empty_edges_and_missing_categories_are_finite(self):
        self.assertTrue(torch.isfinite(self.call(self.graph)[3]).all())
        empty = self.graph.clone()
        empty.edge_index = empty.edge_index[:, :0]
        for key in ('edge_attr', 'edge_cat', 'simp_edge_batch'):
            empty[key] = empty[key][:0]
        result = self.call(empty)
        self.assertEqual(tuple(result[0].shape), (2, 0))
        self.assertEqual(tuple(result[3].shape), (0, 2))

    def test_query_key_and_gate_receive_gradients(self):
        self.call(self.graph)[3].square().sum().backward()
        for name, parameter in self.module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def check_original_equivalence(self, device):
        for aggregation, composition in itertools.product(('sum', 'gin'), ('single', 'mixed', 'all_mixed')):
            with self.subTest(device=device, aggregation=aggregation, composition=composition):
                graph = make_graph().to(device)
                if composition == 'single':
                    graph.edge_cat[:, 2] = 0
                elif composition == 'all_mixed':
                    graph.edge_index = graph.edge_index[:, :5]
                    for key in ('edge_attr', 'edge_cat', 'simp_edge_batch'):
                        graph[key] = graph[key][:5]
                x = graph.edge_attr.clone().requires_grad_()
                old_x = graph.edge_attr.clone().requires_grad_()
                module = MultiEdgeAggModule(2, aggregation, conditioned_by_payment=True,
                                            num_payment_formats=5).to(device)
                original = copy.deepcopy(module)
                arguments = (graph.simp_edge_batch, None, graph.edge_cat)
                actual = module(graph.edge_index, x, *arguments)
                expected = original_conditioned_forward(original, graph.edge_index, old_x, *arguments)
                for new, old in zip(actual, expected):
                    torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
                (actual[1].square().sum() + actual[3].square().sum()).backward()
                (expected[1].square().sum() + expected[3].square().sum()).backward()
                torch.testing.assert_close(x.grad, old_x.grad, rtol=1e-5, atol=1e-6)
                for name, parameter in module.named_parameters():
                    reference = dict(original.named_parameters())[name]
                    grad = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
                    old_grad = reference.grad if reference.grad is not None else torch.zeros_like(reference)
                    torch.testing.assert_close(grad, old_grad, rtol=1e-5, atol=1e-6, msg=name)
                # The version-2 state_dict is still loadable without conversion.
                module.load_state_dict(original.state_dict(), strict=True)

    def test_optimized_context_matches_original_cpu(self):
        self.check_original_equivalence('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_optimized_context_matches_original_cuda(self):
        self.check_original_equivalence('cuda:0')

    def test_single_format_batch_preserves_adam_momentum_updates(self):
        module = copy.deepcopy(self.module)
        reference = copy.deepcopy(self.module)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.0003)
        old_optimizer = torch.optim.Adam(reference.parameters(), lr=0.0003)
        graph = make_graph()
        for single_format in (False, True, False):
            if single_format:
                graph.edge_cat[:, 2] = 0
            else:
                graph = make_graph()
            optimizer.zero_grad(set_to_none=True)
            old_optimizer.zero_grad(set_to_none=True)
            actual = self.call(graph, module)[3]
            expected = original_conditioned_forward(reference, graph.edge_index, graph.edge_attr,
                                                     graph.simp_edge_batch, None, graph.edge_cat)[3]
            actual.square().sum().backward()
            expected.square().sum().backward()
            optimizer.step()
            old_optimizer.step()
            for name, parameter in module.named_parameters():
                torch.testing.assert_close(parameter, dict(reference.named_parameters())[name],
                                           rtol=1e-5, atol=1e-7, msg=name)


class ArchitectureTests(unittest.TestCase):
    def exercise_variants(self, device):
        for reverse, dual, conditioned in itertools.product((False, True), repeat=3):
            with self.subTest(device=device, reverse=reverse, dual=dual, conditioned=conditioned):
                torch.manual_seed(42)
                args = make_args(dual, conditioned, reverse)
                model = make_model(args).to(device)
                batch = make_graph(reverse).to(device)
                labels = batch['node', 'to', 'node'].y if reverse else batch.y
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
                for _ in range(2):
                    optimizer.zero_grad()
                    out = model(batch)
                    self.assertEqual(tuple(out.shape), (7, 2))
                    loss = F.cross_entropy(out, labels)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    optimizer.step()
                if conditioned:
                    aggr = model.gnn.edge_aggrs[0]
                    aggr = aggr['node__to__node'] if reverse else aggr
                    self.assertIsNotNone(aggr.context_query.weight.grad)

    def test_all_variants_cpu(self):
        self.exercise_variants('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_all_variants_cuda(self):
        self.exercise_variants('cuda:0')

    def test_empty_reverse_relation(self):
        for dual, conditioned in itertools.product((False, True), repeat=2):
            model = make_model(make_args(dual, conditioned, True))
            out = model(make_graph(True, empty_reverse=True))
            self.assertTrue(torch.isfinite(out).all())
            out.sum().backward()

    def test_receiver_reaches_each_dual_update(self):
        helper = GnnHelper(2, n_hidden=2, edge_updates=True, args=make_args(dual=True))
        graph = make_graph()
        updates, node_outputs = [], []
        handles = []
        # Recover each layer's updated nodes from the normalization output.
        for norm, update in zip(helper.batch_norms, helper.emlps):
            handles.append(norm.register_forward_hook(lambda module, inputs, output: node_outputs.append(output)))
            handles.append(update.register_forward_pre_hook(lambda module, inputs: updates.append(inputs[0])))
        x = graph.x
        helper(x, graph.edge_index, graph.edge_attr, graph.simp_edge_batch)
        for i, tensor in enumerate(updates):
            x = (x + F.relu(node_outputs[i])) / 2
            torch.testing.assert_close(tensor[:, :2], x[graph.edge_index[0]])
            torch.testing.assert_close(tensor[:, 2:4], x[graph.edge_index[1]])
        for handle in handles:
            handle.remove()

    def test_default_switches_match_explicit_baseline(self):
        implicit = make_args(reverse=True)
        del implicit.aml_dual_endpoint_update
        del implicit.aml_conditioned_edge_agg
        torch.manual_seed(42)
        baseline = make_model(implicit).eval()
        torch.manual_seed(42)
        explicit = make_model(make_args(reverse=True)).eval()
        self.assertEqual(set(baseline.state_dict()), set(explicit.state_dict()))
        graph = make_graph(True)
        torch.testing.assert_close(baseline(graph), explicit(graph), rtol=0, atol=0)

    def test_checkpoint_roundtrip_and_variant_mismatch(self):
        from train_util import load_model, save_model
        for dual, conditioned in itertools.product((False, True), repeat=2):
            args = make_args(dual, conditioned, reverse=True)
            args.unique_name = 'test_run'
            model = make_model(args).eval()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
            with tempfile.TemporaryDirectory() as directory:
                save_model(model, optimizer, 2, args,
                           {'paths': {'model_to_save': directory}}, decision_threshold=0.37)
                checkpoint = torch.load(Path(directory) / 'test_run' / 'epoch_3.tar',
                                        map_location='cpu', weights_only=True)
                self.assertEqual(checkpoint['epoch'], 3)
                self.assertEqual(checkpoint['decision_threshold'], 0.37)
                self.assertEqual(checkpoint['aml_architecture_config'], get_aml_architecture_config(args))
                validate_aml_architecture_checkpoint(checkpoint, args)
                with patch.dict(os.environ, {'MEGAGNN_CHECKPOINT': str(Path(directory) / 'test_run' / 'epoch_3.tar')}):
                    reloaded, _ = load_model(make_model(args).eval(), torch.device('cpu'), args,
                                             SimpleNamespace(lr=0.0003),
                                             {'paths': {'model_to_load': directory}})
                torch.testing.assert_close(model(make_graph(True)), reloaded(make_graph(True)))
                wrong = copy.copy(args)
                wrong.aml_dual_endpoint_update = not dual
                with self.assertRaisesRegex(ValueError, 'architecture'):
                    validate_aml_architecture_checkpoint(checkpoint, wrong)
        validate_aml_architecture_checkpoint({}, make_args())
        with self.assertRaisesRegex(ValueError, 'metadata'):
            validate_aml_architecture_checkpoint({}, make_args(conditioned=True))

    def test_sampled_categories_stay_aligned(self):
        from torch_geometric.loader import LinkNeighborLoader
        from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE
        if not (WITH_PYG_LIB or WITH_TORCH_SPARSE):
            self.skipTest('Neighbor-sampling backend unavailable')
        graph = make_graph(reverse=True)
        loader = LinkNeighborLoader(
            graph, num_neighbors=[2, 2], batch_size=2, shuffle=False,
            edge_label_index=(('node', 'to', 'node'), graph['node', 'to', 'node'].edge_index),
        )
        model = make_model(make_args(dual=True, conditioned=True, reverse=True))
        for batch in loader:
            for edge_type in graph.edge_types:
                storage = batch[edge_type]
                torch.testing.assert_close(storage.edge_cat, graph[edge_type].edge_cat[storage.e_id])
                torch.testing.assert_close(storage.simp_edge_batch, graph[edge_type].simp_edge_batch[storage.e_id])
            out = model(batch)
            self.assertTrue(torch.isfinite(out).all())
            out.square().mean().backward()

    def test_invalid_combinations_fail_early(self):
        for changes in ({'flatten_edges': False}, {'emlps': False},
                        {'edge_agg_type': None}, {'edge_agg_type': 'gru'},
                        {'aml_categorical_features': False}, {'task': 'lp'}):
            args = make_args(conditioned=True)
            vars(args).update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_aml_architecture_args(args)
        for agg_type in (None, 'gru', 'pna', 'genagg'):
            with self.assertRaises(ValueError):
                MultiEdgeAggModule(8, agg_type, conditioned_by_payment=True, num_payment_formats=5)
        args = make_args(dual=True)
        with self.assertRaises(ValueError):
            GnnHelper(2, n_hidden=8, edge_updates=False, args=args)

    def test_cli_switches_are_independent(self):
        base = ['--data', 'Small_HI', '--model', 'gin', '--flatten_edges',
                '--edge_agg_type', 'gin', '--emlps', '--aml_categorical_features']
        for dual, conditioned in itertools.product((False, True), repeat=2):
            flags = (['--aml_dual_endpoint_update'] if dual else [])
            flags += ['--aml_conditioned_edge_agg'] if conditioned else []
            args = create_parser().parse_args(base + flags)
            validate_aml_architecture_args(args)
            self.assertEqual(args.aml_dual_endpoint_update, dual)
            self.assertEqual(args.aml_conditioned_edge_agg, conditioned)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
