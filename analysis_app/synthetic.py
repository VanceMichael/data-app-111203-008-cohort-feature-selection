"""合成泄漏演示：重复设备如何欺骗随机拆分，以及选择边界如何挡住穿透。

每个设备有独立的特征偏移量（同设备各行互为近重复），标签是设备级硬币，
与偏移量无关：随机行拆分让记忆型模型在测试集看到同设备的近重复行而得分虚高，
按设备/批次分组的评估则回到机会水平。附加列演示两类穿透：
postmortem 只在晚到记录里复制标签（时间规则必须排除），constant_channel 恒为常数。
"""

import random

from .ml import auc, fit_model, predict_proba
from .train import BASE_FEATURES, build_instances, outer_evaluate

SYNTHETIC_EXTRAS = ['postmortem', 'constant_channel', 'sparse_channel']
DEMO_SEED = 20261112


def make_leakage_rows(seed=DEMO_SEED, batches=8, devices_per_batch=4, cycles=10):
    """生成确定性的合成台账行（含标准字段与三个附加列）。"""
    rng = random.Random(seed)
    rows = []
    for b in range(batches):
        for d in range(devices_per_batch):
            device = f'SD{b}-{d}'
            offset = rng.gauss(0.0, 1.0)
            label = 1 if rng.random() < 0.5 else 0
            late = d == devices_per_batch - 1
            for c in range(cycles):
                rows.append({
                    'observation_id': f'syn-{b:02d}-{d}-{c:02d}',
                    'device': device,
                    'batch': f'SB{b}',
                    'cycle': c,
                    'vibration': offset + rng.gauss(0.0, 0.05),
                    'temperature': 40.0 + 3.0 * offset + rng.gauss(0.0, 0.1),
                    'load': 50 + (c % 5),
                    'failed': label,
                    'recorded': c + (2 if late else 0),
                    'postmortem': (99.0 if label else -99.0) if late else rng.gauss(0.0, 1.0),
                    'constant_channel': 7.0,
                    'sparse_channel': None if rng.random() < 0.2 else offset + rng.gauss(0.0, 0.1),
                })
    return rows


def _knn_auc(train_rows, test_rows, features, k=3):
    """k-NN 在标准化特征上的 AUC；标准化只在训练侧拟合。"""
    means = {f: sum(r[f] for r in train_rows) / len(train_rows) for f in features}
    scales = {}
    for f in features:
        var = sum((r[f] - means[f]) ** 2 for r in train_rows) / len(train_rows)
        scales[f] = var ** 0.5 or 1.0
    scale = lambda r: {f: (r[f] - means[f]) / scales[f] for f in features}
    model, _ = fit_model({'model': 'knn', 'k': k},
                         [scale(r) for r in train_rows], features,
                         [r['failed'] for r in train_rows])
    scores = predict_proba(model, [scale(r) for r in test_rows])
    return auc(scores, [r['failed'] for r in test_rows])


def random_vs_grouped(rows, seed=DEMO_SEED, features=('vibration', 'temperature')):
    """同一批可用行上的两种评估：随机行拆分 vs 按设备分组拆分。"""
    usable = [r for r in rows if r['recorded'] <= r['cycle']]
    rng = random.Random(seed)
    idx = list(range(len(usable)))
    rng.shuffle(idx)
    half = len(idx) // 2
    random_auc = _knn_auc([usable[i] for i in idx[:half]], [usable[i] for i in idx[half:]], features)
    devices = sorted({r['device'] for r in usable})
    rng.shuffle(devices)
    held = set(devices[: len(devices) // 2])
    grouped_auc = _knn_auc([r for r in usable if r['device'] not in held],
                           [r for r in usable if r['device'] in held], features)
    return {'random_split_auc': random_auc, 'device_grouped_auc': grouped_auc}


def leakage_demo(seed=DEMO_SEED, miss_cost=5.0):
    """完整演示：随机拆分虚高、分组评估回落，且三条边界均不被穿透。"""
    rows = make_leakage_rows(seed)
    whitelist = list(BASE_FEATURES) + SYNTHETIC_EXTRAS
    instances = build_instances(rows, whitelist)
    late = [inst for inst in instances if not inst['usable']]
    split = random_vs_grouped(rows, seed)
    folds = outer_evaluate(instances, whitelist, miss_cost=miss_cost,
                           lambdas=(0.1,), knn_ks=(3,))
    selected = [tuple(fold['selected']['features']) for fold in folds]
    constant_selected = any('constant_channel' in names for names in selected)
    batch_devices = {}
    for inst in instances:
        batch_devices.setdefault(inst['batch'], set()).add(inst['device'])
    overlap = any(
        set(fold['test_devices'])
        & set().union(*(devs for batch, devs in batch_devices.items() if batch != fold['batch']))
        for fold in folds)
    usable_labels = [inst['label'] for inst in instances if inst['usable']]
    return {'product': 'cohort-feature-selection', 'report': 'demo', 'schema_version': 1,
            'seed': seed, 'rows': len(rows), 'usable_rows': len(usable_labels),
            'random_split_auc': split['random_split_auc'],
            'device_grouped_auc': split['device_grouped_auc'],
            'boundary_checks': {
                'late_rows': len(late),
                'late_rows_all_excluded': all(inst['refuse_reason'] for inst in late),
                'constant_feature_never_selected': not constant_selected,
                'outer_device_overlap': overlap,
            },
            'outer_selected_features': [list(names) for names in selected]}
