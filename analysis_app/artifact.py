"""模型产物的保存、加载与对新批次的预测。

产物自带字段白名单、时间可用性规则与全部拟合变换；
加载后对新批次给出概率，或对每条记录给出可验证的拒绝原因。
"""

import json
from pathlib import Path

from .ml import apply_imputer, apply_standardizer, predict_proba
from .schema import PRODUCT

ARTIFACT_TYPE = 'cohort-feature-selection-model'
REQUIRED_KEYS = ['artifact_version', 'field_whitelist', 'extra_fields', 'time_rule',
                 'selected_features', 'imputation', 'standardization', 'model',
                 'threshold', 'miss_cost', 'status', 'input_fingerprint']


def load_artifact(path):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        raise ValueError('模型文件不是合法 JSON') from None
    if not isinstance(data, dict) or data.get('artifact_type') != ARTIFACT_TYPE:
        raise ValueError('不是本产品的模型文件')
    if data.get('artifact_version') != 1:
        raise ValueError('模型版本不支持：' + str(data.get('artifact_version')))
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError('模型文件缺少字段：' + ','.join(missing))
    return data


def predict_instances(artifact, instances):
    """逐实例给出概率或拒绝原因；顺序按 observation_id，结果确定。"""
    pipeline = {'features': artifact['selected_features'], 'imputation': artifact['imputation'],
                'standardization': artifact['standardization'], 'model': artifact['model']}
    predictions = []
    for inst in instances:
        if not inst['usable']:
            predictions.append({'observation_id': inst['observation_id'], 'refused': True,
                                'reason': inst['refuse_reason']})
            continue
        filled = apply_imputer(pipeline['imputation'], inst['features'])
        scaled = apply_standardizer(pipeline['standardization'], filled)
        probability = predict_proba(pipeline['model'], [scaled])[0]
        predictions.append({'observation_id': inst['observation_id'],
                            'device': inst['device'], 'batch': inst['batch'],
                            'cycle': inst['cycle'],
                            'probability': probability,
                            'alert': probability >= artifact['threshold'],
                            'refused': False})
    return predictions


def predict_report(artifact, instances):
    predictions = predict_instances(artifact, instances)
    refused = sum(1 for item in predictions if item['refused'])
    return {'product': PRODUCT, 'report': 'predict', 'schema_version': 1,
            'model': {'input_fingerprint': artifact['input_fingerprint'],
                      'status': artifact['status'],
                      'selected_features': artifact['selected_features'],
                      'threshold': artifact['threshold'],
                      'time_rule': artifact['time_rule']},
            'rows': len(predictions), 'scored': len(predictions) - refused,
            'refused': refused, 'predictions': predictions}
