"""Train-only smoothed frequency probes; evaluates validation only, no test selection.
These are diagnostic tables, NOT MEGA-GNN training or architecture ablations.
"""
from pathlib import Path
import json
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent / 'results'


def ap_from_groups(groups):
    ranked = (groups.group_by('score').agg(pl.col('n').sum(), pl.col('positive').sum())
              .sort('score', descending=True))
    n = ranked['n'].to_numpy().astype(float)
    p = ranked['positive'].to_numpy().astype(float)
    return float(np.sum(p / p.sum() * np.cumsum(p) / np.cumsum(n)))


def evaluate(df, keys, strength=1000):
    train = df.filter(pl.col('split') == 'train')
    val = df.filter(pl.col('split') == 'validation')
    priors = train.group_by('payment_format').agg(pl.col('label').mean().alias('prior'))
    table = (train.group_by(keys).agg(pl.len().alias('train_n'), pl.col('label').sum().alias('train_pos'))
             .join(priors, on='payment_format')
             .with_columns(((pl.col('train_pos') + strength * pl.col('prior')) /
                            (pl.col('train_n') + strength)).alias('score')))
    groups = (val.group_by(keys).agg(pl.len().alias('n'), pl.col('label').sum().alias('positive'))
              .join(table.select(keys + ['score']), on=keys, how='left')
              .join(priors, on='payment_format', how='left')
              .with_columns(pl.col('score').fill_null(pl.col('prior'))))
    assert groups['score'].null_count() == 0
    return {'keys': keys, 'smoothing_strength': strength,
            'train_cells': table.height, 'validation_AP': ap_from_groups(groups),
            'validation_n': val.height, 'validation_positive': int(val['label'].sum())}


def bucket(c):
    return (pl.when(pl.col(c) == 0).then(pl.lit('0'))
            .when(pl.col(c) <= 2).then(pl.lit('1-2'))
            .when(pl.col(c) <= 8).then(pl.lit('3-8'))
            .when(pl.col(c) <= 32).then(pl.lit('9-32')).otherwise(pl.lit('33+')))


assert abs(ap_from_groups(pl.DataFrame({'score':[0.1], 'n':[100], 'positive':[10]})) - .1) < 1e-12
assert ap_from_groups(pl.DataFrame({'score':[1.,0.], 'n':[10,90], 'positive':[10,0]})) == 1.
for path in sorted(ROOT.glob('*_diagnostic_features.parquet')):
    df = pl.read_parquet(path).with_columns(
        (pl.col('prior_pair_count') > 0).alias('prior_pair'),
        (pl.col('incoming_count_60m') > 0).alias('in_1h'),
        bucket('prior_pair_count').alias('pair_bin'),
        bucket('incoming_count_1440m').alias('in_24h_bin'),
        bucket('outgoing_count_1440m').alias('out_24h_bin'))
    configs = [['payment_format'], ['payment_format', 'prior_pair'],
               ['payment_format', 'pair_bin'], ['payment_format', 'in_1h'],
               ['payment_format', 'in_24h_bin'],
               ['payment_format', 'pair_bin', 'in_24h_bin', 'out_24h_bin']]
    probes = [evaluate(df, keys) for keys in configs]
    conditional = (df.filter(pl.col('split') == 'train').group_by(['payment_format', 'prior_pair'])
                   .agg(pl.len().alias('n'), pl.col('label').sum().alias('positive'))
                   .with_columns((pl.col('positive') / pl.col('n')).alias('positive_rate'))
                   .sort(['payment_format', 'prior_pair']).to_dicts())
    name = path.name.replace('_diagnostic_features.parquet', '')
    output = {'dataset': name, 'method': 'train-only frequency tables; shrinkage to payment-format rate with 1000 pseudo-observations; validation AP with tied scores grouped',
              'probes': probes, 'train_prior_pair_by_format': conditional}
    (ROOT / f'{name}_probes.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(name, json.dumps(output, ensure_ascii=False), flush=True)
