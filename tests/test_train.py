import copy
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analysis_app.artifact import load_artifact, predict_instances, predict_report
from analysis_app.cli import save_json
from analysis_app.io import load_records
from analysis_app.ml import apply_imputer, fit_imputer, select_threshold
from analysis_app.synthetic import SYNTHETIC_EXTRAS, leakage_demo, make_leakage_rows
from analysis_app.train import (BASE_FEATURES, REFUSE_LATE, build_artifact, build_instances,
                                check_whitelist, load_instances, outer_evaluate,
                                screen_features, train_report)

SMALL = {'lambdas': (0.1,), 'knn_ks': (3,)}


def make_records(batch_labels, cycles=4):
    """按 {batch: [标签...]} 构造确定性的最小台账。"""
    records = []
    for b, labels in enumerate(batch_labels):
        for d, label in enumerate(labels):
            for c in range(cycles):
                records.append({'observation_id': f't-{b}-{d}-{c}', 'device': f'TD{b}-{d}',
                                'batch': f'TB{b}', 'cycle': c,
                                'vibration': 2.0 + 0.1 * label + 0.01 * c,
                                'temperature': 40.0 + label, 'load': 50.0 + c,
                                'failed': label, 'recorded': c})
    return records


class BoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_records('data/observations.csv')
        cls.instances = build_instances(cls.records, BASE_FEATURES)

    def test_late_rows_never_enter_training(self):
        late = [inst for inst in self.instances if not inst['usable']]
        self.assertEqual(len(late), 48)
        self.assertTrue(all(inst['refuse_reason'] == REFUSE_LATE for inst in late))
        usable = [inst for inst in self.instances if inst['usable']]
        self.assertTrue(all(inst['recorded'] <= inst['cycle'] for inst in usable))

    def test_outer_folds_grouped_by_batch_and_device(self):
        folds = outer_evaluate(self.instances, BASE_FEATURES, inner_folds=2, **SMALL)
        self.assertEqual(len(folds), 4)
        all_devices = {inst['device'] for inst in self.instances}
        for fold in folds:
            test_devices = set(fold['test_devices'])
            train_devices = all_devices - test_devices
            self.assertFalse(test_devices & train_devices)
            for inst in self.instances:
                if inst['device'] in test_devices:
                    self.assertEqual(inst['batch'], fold['batch'])
            self.assertEqual(fold['train_rows'] + fold['metrics']['scored']
                             + fold['metrics']['refused']
                             + sum(1 for i in self.instances if not i['usable']
                                   and i['batch'] != fold['batch']),
                             len(self.instances))

    def test_constant_feature_never_selected(self):
        rows = make_leakage_rows()
        whitelist = list(BASE_FEATURES) + SYNTHETIC_EXTRAS
        instances = build_instances(rows, whitelist)
        usable = [inst for inst in instances if inst['usable']]
        labels = [inst['label'] for inst in usable]
        imputer = fit_imputer([inst['features'] for inst in usable], whitelist, 'mean')
        filled = [apply_imputer(imputer, inst['features']) for inst in usable]
        self.assertNotIn('constant_channel', screen_features(filled, whitelist, labels))
        folds = outer_evaluate(instances, whitelist, inner_folds=2, **SMALL)
        self.assertTrue(folds)
        for fold in folds:
            self.assertNotIn('constant_channel', fold['selected']['features'])
        artifact = build_artifact(rows, instances, whitelist, SYNTHETIC_EXTRAS,
                                  inner_folds=2, **SMALL)
        self.assertNotIn('constant_channel', artifact['selected_features'])

    def test_single_class_fold_degraded_not_skipped(self):
        records = make_records([[0, 0], [0, 1], [1, 1]])
        instances = build_instances(records, BASE_FEATURES)
        report = train_report(records, instances, BASE_FEATURES, [], inner_folds=2, **SMALL)
        self.assertEqual(report['aggregate']['folds_total'], 3)
        statuses = {fold['batch']: fold['status'] for fold in report['outer_folds']}
        self.assertEqual(statuses['TB0'], 'degraded:test_single_class')
        self.assertEqual(statuses['TB2'], 'degraded:test_single_class')
        degraded = report['aggregate']['degraded_folds']
        self.assertEqual(len(degraded), 2)
        self.assertEqual(report['aggregate']['aggregated_over'], report['aggregate']['folds_ok'])
        self.assertEqual(report['aggregate']['folds_ok'], 1)

    def test_train_single_class_uses_constant_fallback(self):
        records = make_records([[0, 0], [1, 1]])
        instances = build_instances(records, BASE_FEATURES)
        folds = outer_evaluate(instances, BASE_FEATURES, inner_folds=2, **SMALL)
        statuses = {fold['batch']: fold['status'] for fold in folds}
        self.assertIn('train_single_class', statuses['TB0'])
        self.assertIn('train_single_class', statuses['TB1'])
        for fold in folds:
            self.assertIsNotNone(fold['metrics'])

    def test_random_split_fooled_but_grouped_not(self):
        result = leakage_demo()
        self.assertGreaterEqual(result['random_split_auc'] - result['device_grouped_auc'], 0.3)
        self.assertGreater(result['random_split_auc'], 0.9)
        checks = result['boundary_checks']
        self.assertTrue(checks['late_rows_all_excluded'])
        self.assertTrue(checks['constant_feature_never_selected'])
        self.assertFalse(checks['outer_device_overlap'])
        self.assertEqual(checks['late_rows'], 80)


class ThresholdTest(unittest.TestCase):
    def test_threshold_respects_miss_cost(self):
        scores = [0.95, 0.8, 0.3, 0.2]
        labels = [0, 1, 1, 0]
        high = select_threshold(scores, labels, miss_cost=5.0)
        self.assertEqual(high['threshold'], 0.25)
        self.assertEqual(high['fn'], 0)
        low = select_threshold(scores, labels, miss_cost=0.5)
        self.assertGreater(low['threshold'], 0.95)
        self.assertEqual(low['fp'], 0)
        self.assertEqual(low['fn'], 2)

    def test_threshold_tie_prefers_fewer_alerts(self):
        scores = [0.5, 0.5]
        labels = [1, 0]
        choice = select_threshold(scores, labels, miss_cost=1.0)
        self.assertGreater(choice['threshold'], 0.5)


class ArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_records('data/observations.csv')
        cls.instances = build_instances(cls.records, BASE_FEATURES)
        cls.artifact = build_artifact(cls.records, cls.instances, BASE_FEATURES, [],
                                      inner_folds=2, **SMALL)

    def test_artifact_carries_whitelist_rule_and_transforms(self):
        for key in ('field_whitelist', 'time_rule', 'imputation',
                    'standardization', 'model', 'threshold'):
            self.assertIn(key, self.artifact)
        self.assertEqual(self.artifact['time_rule']['rule'], 'recorded <= cycle')
        self.assertEqual(self.artifact['field_whitelist'], BASE_FEATURES)
        self.assertEqual(self.artifact['rows_excluded_late'], 48)

    def test_roundtrip_predictions_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.json'
            save_json(path, self.artifact)
            loaded = load_artifact(path)
        self.assertEqual(predict_instances(self.artifact, self.instances),
                         predict_instances(loaded, self.instances))

    def test_predict_refuses_late_rows_with_reason(self):
        report = predict_report(self.artifact, self.instances)
        self.assertEqual(report['refused'], 48)
        refused = [p for p in report['predictions'] if p['refused']]
        self.assertTrue(all('晚于' in p['reason'] for p in refused))
        scored = [p for p in report['predictions'] if not p['refused']]
        self.assertTrue(all(0.0 <= p['probability'] <= 1.0 for p in scored))

    def test_load_rejects_foreign_or_incomplete_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            save_json(path, {'artifact_type': 'other'})
            with self.assertRaisesRegex(ValueError, '模型文件'):
                load_artifact(path)
            broken = dict(self.artifact)
            del broken['threshold']
            save_json(path, broken)
            with self.assertRaisesRegex(ValueError, '缺少字段'):
                load_artifact(path)

    def test_missing_extra_field_at_predict_refused(self):
        artifact = dict(self.artifact)
        artifact['field_whitelist'] = ['vibration', 'extra_x']
        artifact['extra_fields'] = ['extra_x']
        artifact['selected_features'] = ['vibration', 'extra_x']
        artifact['imputation'] = {'strategy': 'mean', 'values': {'vibration': 2.0, 'extra_x': 1.0}}
        artifact['standardization'] = {'vibration': {'mean': 2.0, 'scale': 1.0},
                                       'extra_x': {'mean': 1.0, 'scale': 1.0}}
        artifact['model'] = {'type': 'logistic', 'lambda': 0.1,
                             'features': ['vibration', 'extra_x'],
                             'coefficients': [0.5, 0.5], 'intercept': -1.0}
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / 'model.json'
            save_json(model_path, artifact)
            loaded = load_artifact(model_path)
            instances, _ = load_instances('data/observations.csv',
                                          loaded['field_whitelist'], ['extra_x'], strict=False)
        report = predict_report(loaded, instances)
        self.assertEqual(report['scored'], 0)
        self.assertTrue(all('缺少白名单字段' in p['reason'] or '晚于' in p['reason']
                            for p in report['predictions'] if p['refused']))

    def test_forbidden_feature_rejected(self):
        with self.assertRaisesRegex(ValueError, '不能作为特征'):
            check_whitelist(['failed'])
        with self.assertRaisesRegex(ValueError, '不能作为特征'):
            check_whitelist(['recorded'])


class TrainReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_records('data/observations.csv')
        cls.instances = build_instances(cls.records, BASE_FEATURES)
        cls.report = train_report(cls.records, cls.instances, BASE_FEATURES, [],
                                  inner_folds=2, **SMALL)

    def test_selection_process_is_auditable(self):
        for fold in self.report['outer_folds']:
            self.assertIn('selected', fold)
            self.assertIn('threshold', fold['selected'])
            self.assertTrue(fold['candidates'])
            for candidate in fold['candidates']:
                self.assertIn('inner_cost', candidate)
                self.assertIn('features', candidate)
                self.assertIn('imputation', candidate)

    def test_report_covers_calibration_miss_minority_stability(self):
        fold = self.report['outer_folds'][0]
        self.assertEqual(len(fold['metrics']['calibration']), 5)
        self.assertIsNotNone(fold['metrics']['miss_rate'])
        self.assertIn('coverage', fold['metrics']['minority'])
        stability = self.report['feature_stability']
        self.assertIn('load', stability['selection_frequency'])
        self.assertIsNotNone(stability['mean_jaccard'])

    def test_train_is_deterministic(self):
        again = train_report(self.records, self.instances, BASE_FEATURES, [],
                             inner_folds=2, **SMALL)
        self.assertEqual(self.report, again)

    def test_cli_train_then_predict(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'model.json'
            report = Path(directory) / 'report.json'
            run = subprocess.run([sys.executable, '-m', 'analysis_app', 'train',
                                  '--model', str(model), '--report', str(report)],
                                 text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(report.read_text()), json.loads(run.stdout))
            prediction = Path(directory) / 'pred.json'
            run = subprocess.run([sys.executable, '-m', 'analysis_app', 'predict',
                                  '--model', str(model), '--output', str(prediction)],
                                 text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(prediction.read_text())
            self.assertEqual(result['rows'], 288)
            self.assertEqual(result['refused'], 48)

    def test_cli_rejects_bad_miss_cost(self):
        run = subprocess.run([sys.executable, '-m', 'analysis_app', 'train',
                              '--miss-cost', '0'], text=True, capture_output=True)
        self.assertEqual(run.returncode, 2)


if __name__ == '__main__':
    unittest.main()
