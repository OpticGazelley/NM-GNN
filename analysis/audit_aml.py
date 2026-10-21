"""Full-data, read-only audit of IBM AML CSVs. Requires numpy and polars.

Usage: python analysis/audit_aml.py HI-Small_Trans.csv LI-Small_Trans.csv
All history features use timestamps strictly earlier than each query.
Labels are only used to aggregate diagnostic results, never as history input.
"""
from pathlib import Path
import gc
import itertools
import json
import sys
import time

import numpy as np
import polars as pl

COLS = ['timestamp', 'from_bank', 'from_account', 'to_bank', 'to_account',
        'amount_received', 'receiving_currency', 'amount_paid',
        'payment_currency', 'payment_format', 'label']
OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(parents=True, exist_ok=True)


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def summarize(values):
    values = np.asarray(values)
    if values.size == 0:
        return {'n': 0}
    q = np.quantile(values, [0, .25, .5, .75, .9, .95, .99, 1])
    return dict(zip(['min', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'max'],
                    [float(v) for v in q]), n=int(values.size), mean=float(values.mean()))


def feature_groups(df, columns):
    return (df.group_by(columns).agg(pl.len().alias('n'),
                                    pl.col('label').sum().alias('positive'))
            .with_columns((pl.col('positive') / pl.col('n')).alias('positive_rate'))
            .sort(columns).to_dicts())


class HistoryIndex:
    """Integer-key index, preserving all transactions tied at the same minute."""
    def __init__(self, groups, minutes, width):
        self.width = int(width)
        self.keys = np.sort(np.asarray(groups, dtype=np.int64) * self.width + minutes)

    def query(self, groups, minutes, windows=(60, 1440, 4320)):
        base = np.asarray(groups, dtype=np.int64) * self.width
        keys = base + minutes
        stop = np.searchsorted(self.keys, keys, side='left')
        start = np.searchsorted(self.keys, base, side='left')
        counts = stop - start
        gaps = np.full(len(keys), -1, dtype=np.int64)
        valid = counts > 0
        if len(self.keys):
            gaps[valid] = keys[valid] - self.keys[stop[valid] - 1]
        result = {'count': counts, 'gap_minutes': gaps}
        for window in windows:
            lower = np.maximum(base, keys - window)
            result[f'count_{window}m'] = stop - np.searchsorted(self.keys, lower, side='left')
        return result


def verify_history_index():
    index = HistoryIndex(np.array([0, 0, 0, 1]), np.array([0, 2, 2, 1]), 20)
    result = index.query(np.array([0, 0, 0, 1, 2]), np.array([0, 2, 3, 2, 3]), (1,))
    assert result['count'].tolist() == [0, 1, 3, 1, 0]
    assert result['gap_minutes'].tolist() == [-1, 2, 1, 1, -1]
    assert result['count_1m'].tolist() == [0, 0, 2, 1, 0]


def audit(path):
    name = path.stem.replace('_Trans', '')
    result = {'dataset': name, 'file_bytes': path.stat().st_size,
              'scope': 'full CSV; diagnostic label-conditioned statistics, no model training'}
    log(f'{name}: reading full CSV')
    string_cols = [c for c in COLS if c not in ['amount_received', 'amount_paid', 'label']]
    df = pl.read_csv(path, new_columns=COLS,
                     schema_overrides={**{c: pl.String for c in string_cols},
                                       'amount_received': pl.Float64,
                                       'amount_paid': pl.Float64, 'label': pl.UInt8})
    result['null_counts'] = df.null_count().to_dicts()[0]
    assert sum(result['null_counts'].values()) == 0, 'Missing values require explicit handling'
    assert set(df['label'].unique().to_list()) <= {0, 1}
    for c in ['amount_received', 'amount_paid']:
        assert df[c].is_finite().all(), f'Nonfinite amounts: {c}'
    df = df.with_columns(
        (pl.col('timestamp').str.strptime(pl.Datetime, '%Y/%m/%d %H:%M')
         .dt.epoch('s') // 60).alias('t'),
        pl.concat_str(['from_bank', 'from_account'], separator='|').alias('src_key'),
        pl.concat_str(['to_bank', 'to_account'], separator='|').alias('dst_key'))
    origin = int(df['t'].min())
    df = df.with_columns((pl.col('t') - origin).cast(pl.Int32),
                         (pl.col('from_bank') == pl.col('to_bank')).alias('same_bank'),
                         (pl.col('receiving_currency') != pl.col('payment_currency')).alias('cross_currency'),
                         (pl.col('amount_received') != pl.col('amount_paid')).alias('different_amount'))
    nodes = pl.concat([df.select(pl.col('src_key').alias('key')),
                       df.select(pl.col('dst_key').alias('key'))]).unique().with_row_index('id')
    # Verify that the original formatter's bank+account concatenation does not collide on these files.
    raw_keys = pl.concat([
        df.select(pl.col('src_key').alias('safe'), pl.concat_str(['from_bank', 'from_account']).alias('legacy')),
        df.select(pl.col('dst_key').alias('safe'), pl.concat_str(['to_bank', 'to_account']).alias('legacy'))
    ]).unique()
    result['legacy_key_collision_groups'] = int(raw_keys.group_by('legacy').len().filter(pl.col('len') > 1).height)
    del raw_keys
    df = (df.join(nodes.rename({'key': 'src_key', 'id': 'src'}), on='src_key', how='left')
          .join(nodes.rename({'key': 'dst_key', 'id': 'dst'}), on='dst_key', how='left')
          .with_columns((pl.col('src') == pl.col('dst')).alias('self_loop'),
                        (pl.col('t') // 1440).alias('day')))
    result['basic'] = {'rows': df.height, 'positives': int(df['label'].sum()),
                       'positive_rate': float(df['label'].mean()), 'accounts': nodes.height,
                       'banks': pl.concat([df['from_bank'], df['to_bank']]).n_unique(),
                       'start': df['timestamp'].min(), 'end': df['timestamp'].max(),
                       'span_minutes': int(df['t'].max()),
                       'currencies': pl.concat([df['receiving_currency'], df['payment_currency']]).n_unique(),
                       'payment_formats': df['payment_format'].n_unique(),
                       'file_order_time_decreases': int((df['t'].diff() < 0).sum())}
    # Match get_data's approximate 60/20/20 day-boundary splitting rule.
    day_rows = np.bincount(df['day'].to_numpy())
    cuts = min(itertools.combinations(range(len(day_rows)), 2),
               key=lambda ij: max(abs(v / len(df) - target) / target
                   for v, target in zip([day_rows[:ij[0]].sum(), day_rows[ij[0]:ij[1]].sum(),
                                         day_rows[ij[1]:].sum()], [.6, .2, .2])))
    df = df.with_columns(pl.when(pl.col('day') < cuts[0]).then(pl.lit('train'))
                         .when(pl.col('day') < cuts[1]).then(pl.lit('validation'))
                         .otherwise(pl.lit('test')).alias('split'))
    result['split_day_boundaries'] = list(cuts)
    result['daily'] = feature_groups(df, ['day', 'split'])
    result['splits'] = feature_groups(df, ['split'])
    result['payment_format'] = feature_groups(df, ['split', 'payment_format'])
    result['currency_pair'] = feature_groups(df, ['payment_currency', 'receiving_currency'])
    result['flags'] = {flag: feature_groups(df, ['split', flag]) for flag in
                        ['self_loop', 'same_bank', 'cross_currency', 'different_amount']}
    result['amounts'] = []
    for split in ['train', 'validation', 'test']:
        for label in [0, 1]:
            subset = df.filter((pl.col('split') == split) & (pl.col('label') == label))
            result['amounts'].append({'split': split, 'label': label,
                                     'received': summarize(subset['amount_received'].to_numpy()),
                                     'paid': summarize(subset['amount_paid'].to_numpy())})
    # Amounts here are descriptive across currencies, not a meaningful currency-normalized total.
    result['amount_note'] = 'Raw amount quantiles mix currencies; do not interpret as comparable money values.'
    result['nonpositive_amounts'] = {c: int((df[c] <= 0).sum()) for c in ['amount_received', 'amount_paid']}
    log(f'{name}: {df.height:,} rows, {int(df["label"].sum()):,} positives; pair statistics')
    # Remove high-memory raw account and bank strings after their diagnostics.
    df = df.drop(['from_account', 'to_account', 'src_key', 'dst_key', 'from_bank', 'to_bank', 'timestamp'])
    for c in ['payment_currency', 'receiving_currency', 'payment_format', 'split']:
        df = df.with_columns(pl.col(c).cast(pl.Categorical))
    pair = df.group_by(['src', 'dst']).agg(pl.len().alias('pair_n'), pl.col('label').sum().alias('pair_pos'))
    result['pairs'] = {'unique_pairs': pair.height,
                       'multiplicity': summarize(pair['pair_n'].to_numpy()),
                       'mixed_label_pairs': pair.filter((pl.col('pair_pos') > 0) & (pl.col('pair_pos') < pl.col('pair_n'))).height}
    df = df.join(pair, on=['src', 'dst'], how='left').with_columns(
        (pl.col('pair_n') > 1).alias('repeated_pair_full_graph'),
        ((pl.col('pair_pos') > 0) & (pl.col('pair_pos') < pl.col('pair_n'))).alias('mixed_label_pair_full_graph'))
    result['pairs']['by_label'] = df.group_by('label').agg(
        pl.len().alias('n'), pl.col('repeated_pair_full_graph').mean(),
        pl.col('mixed_label_pair_full_graph').mean()).sort('label').to_dicts()
    result['pairs']['note'] = 'Full-graph multiplicity/mixed labels are diagnostic only; never use as causal model inputs.'
    del pair
    # Historical pair counts: account-pair codes compressed to dense group ids.
    pair_ids = df.select(['src', 'dst']).unique().with_row_index('pair_id')
    df = df.join(pair_ids, on=['src', 'dst'], how='left')
    del pair_ids
    t = df['t'].to_numpy().astype(np.int64)
    src, dst = df['src'].to_numpy(), df['dst'].to_numpy()
    external = src != dst
    width = int(t.max()) + 1
    pair_index = HistoryIndex(df['pair_id'].to_numpy(), t, width)
    pair_history = pair_index.query(df['pair_id'].to_numpy(), t, ())
    df = df.with_columns(pl.Series('prior_pair_count', pair_history['count']),
                         pl.Series('prior_pair_gap', pair_history['gap_minutes']))
    del pair_index, pair_history
    log(f'{name}: strict historical incoming/outgoing coverage (self-transfers excluded)')
    for direction, group in [('incoming', dst), ('outgoing', src)]:
        index = HistoryIndex(group[external], t[external], width)
        values = index.query(src, t)
        df = df.with_columns([pl.Series(f'{direction}_{k}', v) for k, v in values.items()])
        del index, values
    result['history'] = []
    numeric_history = ['prior_pair_count', 'incoming_count', 'outgoing_count',
                       'incoming_count_60m', 'incoming_count_1440m', 'incoming_count_4320m',
                       'outgoing_count_60m', 'outgoing_count_1440m', 'outgoing_count_4320m']
    for split in ['train', 'validation', 'test']:
        for label in [0, 1]:
            subset = df.filter((pl.col('split') == split) & (pl.col('label') == label))
            item = {'split': split, 'label': label, 'n': subset.height}
            for c in numeric_history:
                arr = subset[c].to_numpy()
                item[c] = dict(summarize(arr), fraction_nonzero=float((arr > 0).mean()))
            for c in ['incoming_gap_minutes', 'outgoing_gap_minutes', 'prior_pair_gap']:
                arr = subset[c].to_numpy()
                item[c] = summarize(arr[arr >= 0])
            result['history'].append(item)
    # Conditional tables disentangle broad payment-format / self-loop effects.
    df = df.with_columns((pl.col('incoming_count_1440m') > 0).alias('prior_in_24h'),
                         (pl.col('prior_pair_count') > 0).alias('prior_pair'),
                         pl.when(pl.col('incoming_count_1440m') == 0).then(pl.lit('0'))
                         .when(pl.col('incoming_count_1440m') <= 2).then(pl.lit('1-2'))
                         .when(pl.col('incoming_count_1440m') <= 10).then(pl.lit('3-10'))
                         .otherwise(pl.lit('11+')).alias('in_24h_bucket'))
    result['history_by_format'] = feature_groups(df, ['split', 'payment_format', 'prior_in_24h'])
    result['history_buckets'] = feature_groups(df, ['split', 'in_24h_bucket'])
    result['history_by_external'] = feature_groups(df, ['split', 'self_loop', 'prior_in_24h'])
    result['prior_pair'] = feature_groups(df, ['split', 'prior_pair'])
    # Count simultaneous counterparties without imposing fabricated within-minute order.
    incoming_events = df.filter(pl.col('src') != pl.col('dst')).group_by(['dst', 't']).len()
    result['same_minute_external_incoming'] = {
        'account_minute_groups': incoming_events.height,
        'groups_with_multiple_transactions': incoming_events.filter(pl.col('len') > 1).height,
        'transactions_in_tied_groups': int(incoming_events.filter(pl.col('len') > 1)['len'].sum())}
    del incoming_events
    train = df.filter(pl.col('split') == 'train')
    train_nodes = np.unique(np.concatenate([train['src'].to_numpy(), train['dst'].to_numpy()]))
    train_pairs = np.unique(train['pair_id'].to_numpy())
    df = df.with_columns(pl.Series('both_accounts_seen_train', np.isin(src, train_nodes) & np.isin(dst, train_nodes)),
                         pl.Series('pair_seen_train', np.isin(df['pair_id'].to_numpy(), train_pairs)))
    result['seen_training'] = {flag: feature_groups(df, ['split', flag])
                               for flag in ['both_accounts_seen_train', 'pair_seen_train']}
    # Small diagnostic feature table, excluding raw identifiers, for reproducible follow-up checks.
    audit_cols = ['label', 'split', 'payment_format', 'self_loop', 'cross_currency',
                  'same_bank', 'prior_pair_count', 'incoming_count', 'outgoing_count',
                  'incoming_gap_minutes', 'outgoing_gap_minutes',
                  'incoming_count_60m', 'incoming_count_1440m', 'incoming_count_4320m',
                  'outgoing_count_60m', 'outgoing_count_1440m', 'outgoing_count_4320m']
    df.select(audit_cols).write_parquet(OUT / f'{name}_diagnostic_features.parquet')
    assert sum(x['n'] for x in result['splits']) == result['basic']['rows']
    assert sum(x['positive'] for x in result['splits']) == result['basic']['positives']
    assert df['incoming_gap_minutes'].filter(df['incoming_gap_minutes'] >= 0).min() > 0
    with (OUT / f'{name}_audit.json').open('w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    log(f'{name}: saved audit JSON and diagnostic parquet')
    del df, train, nodes
    gc.collect()
    return result['basic']


if __name__ == '__main__':
    verify_history_index()
    for argument in sys.argv[1:]:
        print(json.dumps(audit(Path(argument)), ensure_ascii=False), flush=True)
