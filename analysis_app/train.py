"""嵌套分组评估与特征选择流程。

外层按 batch 留出整个批次，device 是不可拆分单元；内层只做三件事：
选特征子集、选缺值策略、选模型参数，并按预设漏报成本在内层样本外分数上定阈值。
recorded 晚于 cycle 的记录在预测时刻不可见，不进入特征，也不进入内层任何拟合。
"""

import csv
import math
import random
from collections import Counter
from pathlib import Path

from .io import fingerprint, load_records
from .ml import (MODEL_RANK, apply_imputer, apply_standardizer, auc, fit_imputer,
                 fit_model, fit_standardizer, predict_proba, select_threshold)
from .schema import ENTITY, GROUP, PRODUCT

LABEL = 'failed'
CYCLE = 'cycle'
RECORDED = 'recorded'
BASE_FEATURES = ['vibration', 'temperature', 'load']
FORBIDDEN_FEATURES = {'observation_id', 'device', 'batch', 'failed', 'recorded'}
TIME_RULE = {'rule': 'recorded <= cycle', 'cycle_field': CYCLE, 'recorded_field': RECORDED}
REFUSE_LATE = '记录晚于预测周期，特征不可用'
DEFAULT_MISS_COST = 5.0
DEFAULT_INNER_FOLDS = 3
DEFAULT_SEED = 20261112
LOGISTIC_LAMBDAS = (0.01, 1.0)
KNN_KS = (3, 7)
IMPUTATION_STRATEGIES = ('mean', 'median')
CALIBRATION_BINS = 5


def check_whitelist(extra_fields):
    """附加字段不能是标签、主体、维度或可用性元数据。"""
    bad = sorted(set(extra_fields) & FORBIDDEN_FEATURES)
    if bad:
        raise ValueError('字段不能作为特征：' + ','.join(bad))
    return list(BASE_FEATURES) + [name for name in extra_fields if name not in BASE_FEATURES]


def read_extra_fields(path, extra_fields):
    """读取白名单附加列。返回 ({observation_id: {字段: 值}}, {observation_id: 拒绝原因})。"""
    values, problems = {}, {}
    if not extra_fields:
        return values, problems
    with Path(path).open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        header = set(reader.fieldnames or [])
        rows = list(reader)
    missing = sorted(name for name in extra_fields if name not in header)
    if missing:
        reason = '缺少白名单字段：' + ','.join(missing)
        for raw in rows:
            problems[raw['observation_id']] = reason
        return values, problems
    for raw in rows:
        oid = raw['observation_id']
        parsed = {}
        for name in extra_fields:
            cell = raw.get(name)
            if cell is None or cell == '':
                parsed[name] = None
                continue
            try:
                value = float(cell)
            except ValueError:
                problems[oid] = f'附加字段不是有限数：{name}'
                break
            if not math.isfinite(value):
                problems[oid] = f'附加字段不是有限数：{name}'
                break
            parsed[name] = value
        else:
            values[oid] = parsed
    return values, problems


def build_instances(records, whitelist):
    """把记录变成预测实例；recorded 晚于 cycle 的记录标记为不可用。"""
    instances = []
    for row in sorted(records, key=lambda item: item['observation_id']):
        late = row[RECORDED] > row[CYCLE]
        instances.append({
            'observation_id': row['observation_id'],
            'device': row[ENTITY],
            'batch': row[GROUP],
            'cycle': row[CYCLE],
            'recorded': row[RECORDED],
            'label': row[LABEL],
            'features': {name: row.get(name) for name in whitelist},
            'usable': not late,
            'refuse_reason': REFUSE_LATE if late else None,
        })
    return instances


def load_instances(path, whitelist, extra_fields=(), strict=True):
    """加载记录并合并白名单附加列；strict 时附加列解析失败直接报错。"""
    records = load_records(path)
    extras, problems = read_extra_fields(path, extra_fields)
    if problems and strict:
        oid, reason = sorted(problems.items())[0]
        raise ValueError(f'附加字段解析失败（{oid}）：{reason}')
    for row in records:
        row.update(extras.get(row['observation_id'], {}))
    instances = build_instances(records, whitelist)
    for inst in instances:
        problem = problems.get(inst['observation_id'])
        if problem and inst['usable']:
            inst['usable'] = False
            inst['refuse_reason'] = problem
    return instances, problems


def _corr(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((v - mx) ** 2 for v in x)
    syy = sum((v - my) ** 2 for v in y)
    if sxx == 0 or syy == 0:
        return 0.0
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sxx * syy)


def screen_features(filled_rows, features, labels):
    """单变量筛选：常数特征直接出局，其余按 |点二列相关| 降序，同分按字段名。"""
    scored = []
    for name in features:
        col = [row[name] for row in filled_rows]
        mean = sum(col) / len(col)
        if sum((v - mean) ** 2 for v in col) == 0:
            continue
        scored.append((abs(_corr(col, labels)), name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scored]


def candidate_configs(order, lambdas=LOGISTIC_LAMBDAS, knn_ks=KNN_KS,
                      strategies=IMPUTATION_STRATEGIES):
    """候选 = 筛选序的前 k 个子集 × 缺值策略 × 模型参数；无可用特征时只剩常数模型。"""
    if not order:
        return [{'features': (), 'imputation': 'mean', 'model': 'constant'}]
    models = ([{'model': 'logistic', 'lambda': l2} for l2 in lambdas]
              + [{'model': 'knn', 'k': k} for k in knn_ks])
    return [{'features': tuple(order[:k]), 'imputation': strategy, **model}
            for k in range(1, len(order) + 1)
            for strategy in strategies for model in models]


def fit_pipeline(instances, config):
    """在可用实例上拟合 缺值策略 → 标准化 → 模型，全部变换随产物保存。"""
    train = [inst for inst in instances if inst['usable']]
    features = list(config['features'])
    labels = [inst['label'] for inst in train]
    imputer = fit_imputer([inst['features'] for inst in train], features, config['imputation'])
    filled = [apply_imputer(imputer, inst['features']) for inst in train]
    standardizer = fit_standardizer(filled, features)
    scaled = [apply_standardizer(standardizer, row) for row in filled]
    model, degraded = fit_model(config, scaled, features, labels)
    pipeline = {'features': features, 'imputation': imputer,
                'standardization': standardizer, 'model': model}
    return pipeline, degraded or not train


def predict_pipeline(pipeline, instances):
    out = []
    for inst in instances:
        filled = apply_imputer(pipeline['imputation'], inst['features'])
        scaled = apply_standardizer(pipeline['standardization'], filled)
        out.append(predict_proba(pipeline['model'], [scaled])[0])
    return out


def device_folds(instances, k, seed):
    """按 device 分组切内层折；同一 device 不会同时出现在两侧。"""
    devices = sorted({inst['device'] for inst in instances})
    random.Random(seed).shuffle(devices)
    k = max(1, min(k, len(devices)))
    if k == 1:
        return [(list(instances), list(instances))]
    buckets = [[] for _ in range(k)]
    for pos, device in enumerate(devices):
        buckets[pos % k].append(device)
    folds = []
    for bucket in buckets:
        held = set(bucket)
        folds.append(([inst for inst in instances if inst['device'] not in held],
                      [inst for inst in instances if inst['device'] in held]))
    return folds


def evaluate_config(instances, config, folds, miss_cost):
    """配置在内层折上的样本外成本；阈值只由这些内层分数决定。"""
    oof_scores, oof_labels = [], []
    degraded = 0
    for train_part, test_part in folds:
        pipeline, flag = fit_pipeline(train_part, config)
        degraded += 1 if flag else 0
        oof_scores += predict_pipeline(pipeline, test_part)
        oof_labels += [inst['label'] for inst in test_part]
    if not oof_scores:
        return None
    choice = select_threshold(oof_scores, oof_labels, miss_cost)
    return {'inner_cost': choice['cost'] / len(oof_scores), 'threshold': choice['threshold'],
            'degraded_inner_folds': degraded}


def inner_select(instances, whitelist, miss_cost, inner_folds, seed,
                 lambdas=LOGISTIC_LAMBDAS, knn_ks=KNN_KS):
    """内层选择：特征子集、缺值策略、模型参数与阈值，全部只看内层折。"""
    usable = [inst for inst in instances if inst['usable']]
    if not usable:
        return {'selected': {'features': (), 'imputation': 'mean', 'model': 'constant'},
                'threshold': 1.0, 'candidates': [], 'feature_order': []}
    labels = [inst['label'] for inst in usable]
    ranking_fill = fit_imputer([inst['features'] for inst in usable], whitelist, 'mean')
    filled = [apply_imputer(ranking_fill, inst['features']) for inst in usable]
    order = screen_features(filled, whitelist, labels)
    folds = device_folds(usable, inner_folds, seed)
    results = []
    for config in candidate_configs(order, lambdas, knn_ks):
        outcome = evaluate_config(usable, config, folds, miss_cost)
        if outcome is not None:
            results.append({'config': config, **outcome})
    if not results:
        return {'selected': {'features': (), 'imputation': 'mean', 'model': 'constant'},
                'threshold': 1.0, 'candidates': [], 'feature_order': order}

    def sort_key(item):
        config = item['config']
        return (item['inner_cost'], len(config['features']),
                MODEL_RANK[config['model']], repr(sorted(config.items())))

    results.sort(key=sort_key)
    best = results[0]
    candidates = [{'features': list(item['config']['features']),
                   'imputation': item['config']['imputation'],
                   'model': item['config']['model'],
                   'params': {k: v for k, v in item['config'].items()
                              if k not in ('features', 'imputation', 'model')},
                   'inner_cost': item['inner_cost'],
                   'degraded_inner_folds': item['degraded_inner_folds']}
                  for item in results]
    return {'selected': best['config'], 'threshold': best['threshold'],
            'candidates': candidates, 'feature_order': order}


def _confusion(labels, scores, threshold):
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        alert = score >= threshold
        tp += alert and label == 1
        fp += alert and label == 0
        tn += (not alert) and label == 0
        fn += (not alert) and label == 1
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def _calibration(labels, scores, n_bins=CALIBRATION_BINS):
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, s in enumerate(scores)
               if lo <= s < hi or (b == n_bins - 1 and s == hi)]
        bins.append({'lower': lo, 'upper': hi, 'count': len(idx),
                     'mean_predicted': (sum(scores[i] for i in idx) / len(idx)) if idx else None,
                     'observed_rate': (sum(labels[i] for i in idx) / len(idx)) if idx else None})
    return bins


def _minority_coverage(labels, scores, threshold, refused_labels):
    """少数类覆盖 = 被判对少数类的实例 / 少数类总数（含被拒绝实例）。

    被拒绝的少数类实例计入分母，拒绝行为会降低覆盖；测试折单类时返回 None。
    """
    counts = Counter(list(labels) + list(refused_labels))
    if len(counts) < 2:
        return None
    minority = min(counts, key=lambda key: (counts[key], key))
    caught = sum(1 for score, label in zip(scores, labels)
                 if label == minority and (score >= threshold) == (minority == 1))
    return {'label': minority, 'total': counts[minority], 'caught': caught,
            'coverage': caught / counts[minority]}


def fold_metrics(labels, scores, threshold, refused_labels):
    conf = _confusion(labels, scores, threshold)
    positives = conf['tp'] + conf['fn']
    single_class = len(set(list(labels) + list(refused_labels))) < 2
    return {'scored': len(labels), 'refused': len(refused_labels),
            'confusion': conf,
            'miss_rate': (conf['fn'] / positives) if positives else None,
            'minority': _minority_coverage(labels, scores, threshold, refused_labels),
            'brier': (sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(labels)) if labels else None,
            'auc': auc(scores, labels),
            'calibration': None if single_class else _calibration(labels, scores)}


def outer_evaluate(instances, whitelist, miss_cost=DEFAULT_MISS_COST,
                   inner_folds=DEFAULT_INNER_FOLDS, seed=DEFAULT_SEED,
                   lambdas=LOGISTIC_LAMBDAS, knn_ks=KNN_KS):
    """外层留一 batch：每折重新做内层选择，指标只在该折未见过的 batch 上计算。"""
    folds = []
    for held in sorted({inst['batch'] for inst in instances}):
        train = [inst for inst in instances if inst['batch'] != held]
        test = [inst for inst in instances if inst['batch'] == held]
        usable_test = [inst for inst in test if inst['usable']]
        refused = [inst for inst in test if not inst['usable']]
        status = 'ok'
        if len({inst['label'] for inst in train if inst['usable']}) < 2:
            status = 'degraded:train_single_class'
        selection = inner_select(train, whitelist, miss_cost, inner_folds, seed, lambdas, knn_ks)
        pipeline, _ = fit_pipeline(train, selection['selected'])
        scores = predict_pipeline(pipeline, usable_test)
        labels = [inst['label'] for inst in usable_test]
        refused_labels = [inst['label'] for inst in refused]
        if len(set(labels + refused_labels)) < 2:
            status = 'degraded:test_single_class' if status == 'ok' else status + ';test_single_class'
        folds.append({'batch': held, 'status': status,
                      'train_rows': sum(1 for inst in train if inst['usable']),
                      'train_devices': len({inst['device'] for inst in train}),
                      'test_rows': len(test), 'test_devices': sorted({inst['device'] for inst in test}),
                      'selected': {'features': list(selection['selected']['features']),
                                   'imputation': selection['selected']['imputation'],
                                   'model': selection['selected']['model'],
                                   'params': {k: v for k, v in selection['selected'].items()
                                              if k not in ('features', 'imputation', 'model')},
                                   'threshold': selection['threshold']},
                      'candidates': selection['candidates'],
                      'metrics': fold_metrics(labels, scores, selection['threshold'], refused_labels)})
    return folds


def _mean_or_none(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate(folds):
    """只在 ok 折上取均值；降级折保留在 degraded_folds 中，绝不静默跳过。"""
    ok = [fold for fold in folds if fold['status'] == 'ok']
    degraded = [{'batch': fold['batch'], 'status': fold['status']}
                for fold in folds if fold['status'] != 'ok']
    return {'folds_total': len(folds), 'folds_ok': len(ok), 'aggregated_over': len(ok),
            'degraded_folds': degraded,
            'brier_mean': _mean_or_none([f['metrics']['brier'] for f in ok]),
            'auc_mean': _mean_or_none([f['metrics']['auc'] for f in ok]),
            'miss_rate_mean': _mean_or_none([f['metrics']['miss_rate'] for f in ok]),
            'minority_coverage_mean': _mean_or_none(
                [f['metrics']['minority']['coverage'] for f in ok if f['metrics']['minority']])}


def feature_stability(folds):
    """所选特征的稳定性：各特征入选频率与折间平均 Jaccard。"""
    ok = [fold for fold in folds if fold['status'] == 'ok']
    sets = [set(fold['selected']['features']) for fold in ok]
    counts = Counter(name for subset in sets for name in subset)
    frequency = {name: count / len(ok) for name, count in sorted(counts.items())} if ok else {}
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1:]]
    jaccard = (_mean_or_none([len(a & b) / len(a | b) if a | b else 1.0 for a, b in pairs])
               if pairs else (1.0 if len(sets) == 1 else None))
    return {'selection_frequency': frequency, 'mean_jaccard': jaccard}


def train_report(records, instances, whitelist, extra_fields, miss_cost=DEFAULT_MISS_COST,
                 inner_folds=DEFAULT_INNER_FOLDS, seed=DEFAULT_SEED,
                 lambdas=LOGISTIC_LAMBDAS, knn_ks=KNN_KS):
    folds = outer_evaluate(instances, whitelist, miss_cost, inner_folds, seed, lambdas, knn_ks)
    return {'product': PRODUCT, 'report': 'train', 'schema_version': 1,
            'input_fingerprint': fingerprint(records),
            'config': {'miss_cost': miss_cost, 'inner_folds': inner_folds, 'seed': seed,
                       'field_whitelist': list(whitelist), 'extra_fields': list(extra_fields),
                       'logistic_lambdas': list(lambdas), 'knn_ks': list(knn_ks),
                       'imputation_strategies': list(IMPUTATION_STRATEGIES)},
            'time_rule': TIME_RULE,
            'rows': len(instances),
            'usable_rows': sum(1 for inst in instances if inst['usable']),
            'excluded_late_rows': sum(1 for inst in instances if not inst['usable']),
            'outer_folds': folds,
            'aggregate': aggregate(folds),
            'feature_stability': feature_stability(folds)}


def build_artifact(records, instances, whitelist, extra_fields, miss_cost=DEFAULT_MISS_COST,
                   inner_folds=DEFAULT_INNER_FOLDS, seed=DEFAULT_SEED,
                   lambdas=LOGISTIC_LAMBDAS, knn_ks=KNN_KS):
    """在全部数据上重跑同一内层选择流程，产出可加载、可审计的模型产物。"""
    selection = inner_select(instances, whitelist, miss_cost, inner_folds, seed, lambdas, knn_ks)
    pipeline, degraded = fit_pipeline(instances, selection['selected'])
    return {'artifact_type': 'cohort-feature-selection-model', 'artifact_version': 1,
            'product': PRODUCT,
            'input_fingerprint': fingerprint(records),
            'status': 'degraded:train_single_class' if degraded else 'ok',
            'label_field': LABEL, 'entity_field': ENTITY, 'group_field': GROUP,
            'field_whitelist': list(whitelist), 'extra_fields': list(extra_fields),
            'time_rule': TIME_RULE, 'miss_cost': miss_cost,
            'selected_features': pipeline['features'],
            'imputation': pipeline['imputation'],
            'standardization': pipeline['standardization'],
            'model': pipeline['model'],
            'threshold': selection['threshold'],
            'rows_used': sum(1 for inst in instances if inst['usable']),
            'rows_excluded_late': sum(1 for inst in instances if not inst['usable'])}
