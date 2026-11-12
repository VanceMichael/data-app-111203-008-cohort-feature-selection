import argparse
import json
import sys
import tempfile
import os
from pathlib import Path

from .artifact import load_artifact, predict_report
from .io import load_records
from .summary import summarize
from .synthetic import leakage_demo
from .train import (DEFAULT_INNER_FOLDS, DEFAULT_MISS_COST, DEFAULT_SEED, build_artifact,
                    check_whitelist, load_instances, train_report)


def save_json(path, result):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.analysis-', dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _run_summary(args):
    result = summarize(load_records(args.input))
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


def _run_train(args):
    if args.miss_cost <= 0:
        raise ValueError('漏报成本必须为正数')
    whitelist = check_whitelist(args.extra_field)
    records = load_records(args.input)
    instances, _ = load_instances(args.input, whitelist, args.extra_field)
    report = train_report(records, instances, whitelist, args.extra_field,
                          miss_cost=args.miss_cost, inner_folds=args.inner_folds, seed=args.seed)
    artifact = build_artifact(records, instances, whitelist, args.extra_field,
                              miss_cost=args.miss_cost, inner_folds=args.inner_folds, seed=args.seed)
    save_json(args.model, artifact)
    if args.report:
        save_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


def _run_predict(args):
    artifact = load_artifact(args.model)
    extra = [name for name in artifact['selected_features'] if name in artifact['extra_fields']]
    instances, _ = load_instances(args.input, artifact['field_whitelist'], extra, strict=False)
    report = predict_report(artifact, instances)
    if args.output:
        save_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


def _run_demo(args):
    result = leakage_demo(miss_cost=args.miss_cost)
    if args.output:
        save_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description='本地观测数据分析')
    commands = parser.add_subparsers(dest='command', required=True)

    summary = commands.add_parser('summary', help='按业务维度核对观测口径')
    summary.add_argument('--input', type=Path, default=Path('data/observations.csv'))
    summary.add_argument('--output', type=Path)
    summary.set_defaults(run=_run_summary)

    train = commands.add_parser('train', help='嵌套分组评估特征选择并产出可加载模型')
    train.add_argument('--input', type=Path, default=Path('data/observations.csv'))
    train.add_argument('--model', type=Path, default=Path('output/model.json'))
    train.add_argument('--report', type=Path)
    train.add_argument('--miss-cost', type=float, default=DEFAULT_MISS_COST)
    train.add_argument('--inner-folds', type=int, default=DEFAULT_INNER_FOLDS)
    train.add_argument('--seed', type=int, default=DEFAULT_SEED)
    train.add_argument('--extra-field', action='append', default=[],
                       help='把额外列加入特征白名单（可重复）')
    train.set_defaults(run=_run_train)

    predict = commands.add_parser('predict', help='加载模型产物对新批次给概率或拒绝原因')
    predict.add_argument('--model', type=Path, required=True)
    predict.add_argument('--input', type=Path, default=Path('data/observations.csv'))
    predict.add_argument('--output', type=Path)
    predict.set_defaults(run=_run_predict)

    demo = commands.add_parser('demo', help='合成演示：重复设备如何欺骗随机拆分')
    demo.add_argument('--miss-cost', type=float, default=DEFAULT_MISS_COST)
    demo.add_argument('--output', type=Path)
    demo.set_defaults(run=_run_demo)

    args = parser.parse_args()
    try:
        return args.run(args)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
