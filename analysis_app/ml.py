"""训练流程的数值原语：拟合变换、模型、阈值与指标。

只使用标准库；所有拟合结果都是 JSON 基本类型，可直接进入模型产物。
"""

import math

MODEL_RANK = {'logistic': 0, 'knn': 1, 'constant': 2}


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def fit_imputer(rows, features, strategy):
    """按策略为每个字段拟合缺值填充值；整列缺失回退为 0.0。"""
    if strategy not in ('mean', 'median'):
        raise ValueError('未知缺值策略：' + str(strategy))
    values = {}
    for name in features:
        col = sorted(row[name] for row in rows if row[name] is not None)
        if not col:
            values[name] = 0.0
        elif strategy == 'mean':
            values[name] = _mean(col)
        else:
            mid = len(col) // 2
            values[name] = col[mid] if len(col) % 2 else (col[mid - 1] + col[mid]) / 2
    return {'strategy': strategy, 'values': values}


def apply_imputer(imputer, row):
    return {name: (row.get(name) if row.get(name) is not None else fill)
            for name, fill in imputer['values'].items()}


def fit_standardizer(rows, features):
    """拟合均值与标准差；零方差字段比例固定为 1.0（常数特征应在筛选阶段出局）。"""
    stats = {}
    for name in features:
        col = [row[name] for row in rows]
        mean = _mean(col)
        scale = math.sqrt(_mean((value - mean) ** 2 for value in col))
        stats[name] = {'mean': mean, 'scale': scale if scale > 0 else 1.0}
    return stats


def apply_standardizer(stats, row):
    return {name: (row[name] - stats[name]['mean']) / stats[name]['scale'] for name in stats}


def _fit_logistic(X, labels, l2, lr=0.5, iterations=150):
    """全批量梯度下降的逻辑回归，L2 正则强度为 l2（绝对量，不随样本数缩放）。"""
    width = len(X[0]) if X else 0
    weights = [0.0] * width
    intercept = 0.0
    n = len(X)
    for _ in range(iterations):
        grad_w = [0.0] * width
        grad_b = 0.0
        for xi, yi in zip(X, labels):
            z = max(-35.0, min(35.0, intercept + sum(w * x for w, x in zip(weights, xi))))
            err = 1.0 / (1.0 + math.exp(-z)) - yi
            grad_b += err
            for j in range(width):
                grad_w[j] += err * xi[j]
        for j in range(width):
            weights[j] -= lr * (grad_w[j] / n + l2 * weights[j])
        intercept -= lr * grad_b / n
    return weights, intercept


def fit_model(spec, rows, features, labels):
    """在已标准化的特征上拟合模型。返回 (model, degraded)。

    标签只剩单类时降级为常数先验模型，degraded=True，绝不静默跳过。
    """
    if not labels:
        return {'type': 'constant', 'probability': 0.0}, True
    if len(set(labels)) < 2:
        return {'type': 'constant', 'probability': _mean(labels)}, True
    kind = spec['model']
    if kind == 'logistic':
        X = [[row[name] for name in features] for row in rows]
        weights, intercept = _fit_logistic(X, labels, spec['lambda'])
        return {'type': 'logistic', 'lambda': spec['lambda'], 'features': list(features),
                'coefficients': weights, 'intercept': intercept}, False
    if kind == 'knn':
        X = [[row[name] for name in features] for row in rows]
        return {'type': 'knn', 'k': spec['k'], 'features': list(features),
                'train': [{'x': x, 'y': y} for x, y in zip(X, labels)]}, False
    if kind == 'constant':
        return {'type': 'constant', 'probability': _mean(labels)}, False
    raise ValueError('未知模型类型：' + str(kind))


def predict_proba(model, rows):
    """对标准化特征行给出正类概率。"""
    kind = model['type']
    if kind == 'constant':
        return [model['probability']] * len(rows)
    features = model['features']
    if kind == 'logistic':
        out = []
        for row in rows:
            z = model['intercept'] + sum(w * row[name] for w, name in zip(model['coefficients'], features))
            z = max(-35.0, min(35.0, z))
            out.append(1.0 / (1.0 + math.exp(-z)))
        return out
    if kind == 'knn':
        out = []
        train = [(item['x'], item['y']) for item in model['train']]
        for row in rows:
            vector = [row[name] for name in features]
            dists = sorted((sum((a - b) ** 2 for a, b in zip(vector, x)), i)
                           for i, (x, _) in enumerate(train))
            k = min(model['k'], len(dists))
            out.append(sum(train[i][1] for _, i in dists[:k]) / k)
        return out
    raise ValueError('未知模型类型：' + str(kind))


def select_threshold(scores, labels, miss_cost):
    """按预设漏报成本在内层样本外分数上选阈值，决策规则为 score >= threshold 报警。

    成本 = miss_cost * 漏报数 + 误报数；成本相同取更高阈值（更少误报）。
    只允许使用内层分数，禁止接触外层标签。
    """
    if not scores:
        return {'threshold': 1.0, 'cost': 0.0, 'fn': 0, 'fp': 0}
    unique = sorted(set(scores))
    candidates = [0.0]
    candidates += [(a + b) / 2 for a, b in zip(unique, unique[1:])]
    candidates.append(unique[-1] + 1e-12)
    best = None
    for threshold in candidates:
        fn = fp = 0
        for score, label in zip(scores, labels):
            if score >= threshold:
                fp += label == 0
            else:
                fn += label == 1
        cost = miss_cost * fn + fp
        key = (cost, -threshold)
        if best is None or key < best[0]:
            best = (key, threshold, fn, fp)
    return {'threshold': best[1], 'cost': best[0][0], 'fn': best[2], 'fp': best[3]}


def auc(scores, labels):
    """Mann-Whitney AUC，并列取平均秩；单类返回 None。"""
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for m in range(i, j + 1):
            ranks[order[m]] = (i + j) / 2 + 1
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    return (sum_pos - pos * (pos + 1) / 2) / (pos * neg)
