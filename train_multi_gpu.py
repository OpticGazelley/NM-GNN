"""Launch one AML training job on explicitly selected, currently visible GPUs."""

import argparse
import os
from pathlib import Path
import sys


def parse_gpu_ids(values):
    try:
        ids = [int(item) for value in values for item in value.split(',')]
    except ValueError as exc:
        raise ValueError('--gpus requires integer GPU indices, e.g. --gpus 0 2') from exc
    if not ids or min(ids) < 0 or len(set(ids)) != len(ids):
        raise ValueError('--gpus must contain distinct non-negative GPU indices')
    return ids


def selected_visibility(ids, visible_count, inherited=None):
    if max(ids) >= visible_count:
        raise ValueError(f'Requested GPUs {ids}, but only {visible_count} GPUs are visible')
    available = inherited.split(',') if inherited else [str(i) for i in range(visible_count)]
    if len(available) < visible_count:
        raise ValueError('CUDA_VISIBLE_DEVICES does not match the available GPU count')
    return ','.join(available[i].strip() for i in ids)


def configure_nccl_compatibility(environment):
    for name in ('NCCL_P2P_DISABLE', 'NCCL_IB_DISABLE'):
        environment[name] = '1'
    for name in ('NCCL_CUMEM_ENABLE', 'NCCL_CUMEM_HOST_ENABLE'):
        environment[name] = '0'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', nargs='+', required=True, help='Visible GPU indices, e.g. 0 2 or 0,2')
    parser.add_argument('--nccl_compat', action='store_true', help='Diagnostic fallback: disable NCCL P2P, IB, and cuMem transports')
    args, training_args = parser.parse_known_args()
    if training_args[:1] == ['--']:
        training_args = training_args[1:]
    if any(value == '--device' or value.startswith('--device=') for value in training_args):
        parser.error('Use --gpus instead of --device for distributed training')
    if 'LOCAL_RANK' in os.environ:
        parser.error('Run this launcher directly with python, not inside torchrun')
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    import torch
    try:
        ids = parse_gpu_ids(args.gpus)
        visibility = selected_visibility(ids, torch.cuda.device_count(), os.environ.get('CUDA_VISIBLE_DEVICES'))
    except ValueError as exc:
        parser.error(str(exc))
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = visibility
    environment.setdefault('OMP_NUM_THREADS', '1')
    if args.nccl_compat:
        configure_nccl_compatibility(environment)
        print('NCCL compatibility mode: P2P/IB/cuMem disabled for this job', flush=True)
    worker = Path(__file__).resolve().with_name('distributed_training.py')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
               f'--nproc_per_node={len(ids)}', str(worker), *training_args]
    print(f'Selected GPUs: {ids}; child CUDA_VISIBLE_DEVICES={visibility}; workers={len(ids)}', flush=True)
    # torchrun owns worker startup, error propagation, and termination.
    os.execvpe(command[0], command, environment)


if __name__ == '__main__':
    main()
