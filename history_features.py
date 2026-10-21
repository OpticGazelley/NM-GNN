import hashlib
import logging
import os
from collections import deque

import numpy as np


HISTORY_FEATURE_NAMES = [
    'pair_seen_before',
    'log_pair_count_before',
    'log_pair_last_gap_minutes',
    'pair_history_missing',
    'log_sender_in_count_24h',
    'log_sender_out_count_24h',
    'log_sender_last_in_gap_minutes',
    'sender_in_history_missing',
]

HISTORY_FEATURE_VERSION = 'v2-strict-deque-full-digest'


def _update_array_digest(digest, values, chunk_bytes=8 * 1024 * 1024):
    array = np.ascontiguousarray(values, dtype=np.int64)
    raw = memoryview(array).cast('B')
    for start in range(0, len(raw), chunk_bytes):
        digest.update(raw[start:start + chunk_bytes])


def _cache_path(cache_dir, dataset_name, src, dst, timestamps, window_minutes):
    os.makedirs(cache_dir, exist_ok=True)
    digest = hashlib.sha256()
    digest.update(HISTORY_FEATURE_VERSION.encode('utf-8'))
    digest.update(dataset_name.encode('utf-8'))
    digest.update(str(int(window_minutes)).encode('utf-8'))
    _update_array_digest(digest, src)
    _update_array_digest(digest, dst)
    _update_array_digest(digest, timestamps)
    return os.path.join(cache_dir, f'{dataset_name}_{digest.hexdigest()[:20]}.npz')


def compute_aml_history_features(src, dst, timestamps, dataset_name='aml', window_minutes=1440,
                                 cache_dir='cache/aml_history'):
    """Build label-free transaction history features.

    History is strict: a transaction can only see events with an earlier timestamp.
    Transactions tied at the same timestamp are queried before any of them update the
    running state, so CSV row order cannot leak artificial within-minute ordering.
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    cache_file = _cache_path(cache_dir, dataset_name, src, dst, timestamps, window_minutes)
    if os.path.exists(cache_file):
        logging.info(f'Loading AML history features from {cache_file}')
        with np.load(cache_file) as cached:
            return cached['features'].astype(np.float32), list(cached['names'])

    logging.info('Computing AML history features with strict timestamp history')
    n = len(timestamps)
    features = np.zeros((n, len(HISTORY_FEATURE_NAMES)), dtype=np.float32)
    order = np.lexsort((np.arange(n), timestamps))

    pair_count = {}
    pair_last_time = {}
    sender_last_in = {}
    sender_in_window = {}
    sender_out_window = {}
    window_seconds = int(window_minutes) * 60

    pos = 0
    while pos < n:
        time_value = int(timestamps[order[pos]])
        end = pos + 1
        while end < n and int(timestamps[order[end]]) == time_value:
            end += 1

        batch_ids = order[pos:end]
        for edge_id in batch_ids:
            u = int(src[edge_id])
            v = int(dst[edge_id])
            pair_key = (u, v)

            count_before = pair_count.get(pair_key, 0)
            features[edge_id, 0] = 1.0 if count_before > 0 else 0.0
            features[edge_id, 1] = np.log1p(count_before)
            if count_before > 0:
                gap_minutes = max((time_value - pair_last_time[pair_key]) / 60.0, 0.0)
                features[edge_id, 2] = np.log1p(gap_minutes)
            else:
                features[edge_id, 3] = 1.0

            in_times = sender_in_window.setdefault(u, deque())
            out_times = sender_out_window.setdefault(u, deque())
            cutoff = time_value - window_seconds
            while in_times and in_times[0] < cutoff:
                in_times.popleft()
            while out_times and out_times[0] < cutoff:
                out_times.popleft()
            features[edge_id, 4] = np.log1p(len(in_times))
            features[edge_id, 5] = np.log1p(len(out_times))

            if u in sender_last_in:
                gap_minutes = max((time_value - sender_last_in[u]) / 60.0, 0.0)
                features[edge_id, 6] = np.log1p(gap_minutes)
            else:
                features[edge_id, 7] = 1.0

        for edge_id in batch_ids:
            u = int(src[edge_id])
            v = int(dst[edge_id])
            pair_key = (u, v)
            pair_count[pair_key] = pair_count.get(pair_key, 0) + 1
            pair_last_time[pair_key] = time_value
            if u != v:
                sender_out_window.setdefault(u, deque()).append(time_value)
                sender_in_window.setdefault(v, deque()).append(time_value)
                sender_last_in[v] = time_value

        pos = end

    np.savez_compressed(cache_file, features=features, names=np.array(HISTORY_FEATURE_NAMES))
    logging.info(f'Saved AML history features to {cache_file}')
    return features, HISTORY_FEATURE_NAMES

