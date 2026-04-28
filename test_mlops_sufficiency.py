"""
tests/test_mlops_sufficiency.py
================================
Testes unitários e de integração para a biblioteca mlops_sufficiency.
Execute com: python -m pytest tests/ -v
"""

import sys
import os
import random
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlops_sufficiency import (
    TrainingSufficiencyEvaluator,
    InferenceMetricsCollector,
    DataDriftDetector,
    SufficiencyReport,
)
from mlops_sufficiency.metrics import DetectionResult
from mlops_sufficiency.evaluator import SufficiencyStatus, SufficiencyThresholds
from mlops_sufficiency.drift import DriftSeverity


def _make_collector(
    n: int,
    mean_conf: float = 0.80,
    std_conf: float = 0.07,
    tp: int = 8,
    fp: int = 1,
    fn: int = 2,
    seed: int = 42,
) -> InferenceMetricsCollector:
    random.seed(seed)
    collector = InferenceMetricsCollector()
    for i in range(n):
        scores = [max(0.0, min(1.0, random.gauss(mean_conf, std_conf))) for _ in range(5)]
        collector.add(DetectionResult(
            frame_id=f"f{i}",
            video_id=f"v{i // 30}",
            confidence_scores=scores,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            inference_time_ms=random.uniform(80, 200),
        ))
    return collector


class TestDetectionResult(unittest.TestCase):
    def setUp(self):
        self.r = DetectionResult(
            frame_id="f0",
            video_id="v0",
            confidence_scores=[0.9, 0.8, 0.7],
            true_positives=8,
            false_positives=2,
            false_negatives=1,
        )

    def test_precision(self):
        self.assertAlmostEqual(self.r.precision, 8 / 10)

    def test_recall(self):
        self.assertAlmostEqual(self.r.recall, 8 / 9)

    def test_f1(self):
        p, r = self.r.precision, self.r.recall
        expected = 2 * p * r / (p + r)
        self.assertAlmostEqual(self.r.f1_score, expected)

    def test_mean_confidence(self):
        self.assertAlmostEqual(self.r.mean_confidence, (0.9 + 0.8 + 0.7) / 3)

    def test_zero_division_precision(self):
        r = DetectionResult("f", "v", [], 0, 0, 0)
        self.assertEqual(r.precision, 0.0)
        self.assertEqual(r.recall, 0.0)
        self.assertEqual(r.f1_score, 0.0)


class TestInferenceMetricsCollector(unittest.TestCase):
    def setUp(self):
        self.collector = _make_collector(300)

    def test_total_samples(self):
        self.assertEqual(self.collector.total_samples, 300)

    def test_mean_confidence_range(self):
        conf = self.collector.mean_confidence()
        self.assertGreater(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_windowed_metrics_keys(self):
        w = self.collector.windowed_metrics(50)
        for key in ("window_size", "mean_confidence", "mean_precision", "mean_f1"):
            self.assertIn(key, w)

    def test_confidence_trend_length(self):
        trend = self.collector.confidence_trend(n_bins=5)
        self.assertGreater(len(trend), 0)

    def test_summary_dict_keys(self):
        s = self.collector.to_summary_dict()
        for key in ("total_samples", "mean_f1", "false_positive_rate"):
            self.assertIn(key, s)


class TestEvaluator(unittest.TestCase):
    def test_sufficient_model(self):
        collector = _make_collector(300, mean_conf=0.82, std_conf=0.06, tp=9, fp=1, fn=1)
        evaluator = TrainingSufficiencyEvaluator(
            SufficiencyThresholds(min_samples=200)
        )
        result = evaluator.evaluate(collector)
        self.assertEqual(result.status, SufficiencyStatus.SUFFICIENT)
        self.assertGreater(result.score, 0.7)

    def test_insufficient_data(self):
        collector = _make_collector(50)
        result = TrainingSufficiencyEvaluator.quick_check(collector)
        self.assertEqual(result.status, SufficiencyStatus.NEEDS_MORE_DATA)

    def test_bad_model(self):
        collector = _make_collector(
            300, mean_conf=0.45, std_conf=0.2, tp=4, fp=6, fn=8, seed=77
        )
        result = TrainingSufficiencyEvaluator.quick_check(collector)
        self.assertEqual(result.status, SufficiencyStatus.NEEDS_RETRAINING)

    def test_score_between_0_and_1(self):
        for seed in range(5):
            collector = _make_collector(250, seed=seed)
            result = TrainingSufficiencyEvaluator().evaluate(collector)
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)

    def test_sufficient_model_not_needs_retraining(self):
        collector = _make_collector(300, mean_conf=0.82, fp=1, fn=1)
        result = TrainingSufficiencyEvaluator(
            SufficiencyThresholds(min_samples=200)
        ).evaluate(collector)
        self.assertTrue(result.is_sufficient())
        self.assertFalse(result.needs_retraining())


class TestDriftDetector(unittest.TestCase):
    def setUp(self):
        self.baseline_collector = _make_collector(200, mean_conf=0.80, fp=1, fn=1)
        self.detector = DataDriftDetector()
        self.detector.set_baseline_from_collector(self.baseline_collector)

    def test_no_drift_when_stable(self):
        stable_collector = _make_collector(200, mean_conf=0.80, fp=1, fn=1)
        report = self.detector.evaluate(stable_collector)
        self.assertIn(report.severity, (DriftSeverity.NONE, DriftSeverity.LOW))

    def test_high_drift_when_degraded(self):
        degraded = _make_collector(200, mean_conf=0.50, std_conf=0.20, fp=5, fn=6, seed=99)
        report = self.detector.evaluate(degraded)
        self.assertIn(report.severity, (DriftSeverity.MEDIUM, DriftSeverity.HIGH))
        self.assertTrue(report.is_retraining_needed())

    def test_baseline_not_set_raises(self):
        fresh_detector = DataDriftDetector()
        with self.assertRaises(RuntimeError):
            fresh_detector.evaluate(self.baseline_collector)

    def test_manual_baseline(self):
        self.detector.set_baseline(
            mean_confidence=0.80,
            mean_f1=0.75,
            false_positive_rate=0.10,
            false_negative_rate=0.15,
        )
        report = self.detector.evaluate(_make_collector(200))
        self.assertIsNotNone(report)

    def test_ks_distance_identical_distributions(self):
        sample = [0.8] * 100
        ks = DataDriftDetector.kolmogorov_smirnov_distance(sample, sample)
        self.assertAlmostEqual(ks, 0.0)

    def test_ks_distance_different_distributions(self):
        random.seed(1)
        a = [random.gauss(0.8, 0.05) for _ in range(200)]
        b = [random.gauss(0.5, 0.15) for _ in range(200)]
        ks = DataDriftDetector.kolmogorov_smirnov_distance(a, b)
        self.assertGreater(ks, 0.1)


class TestReport(unittest.TestCase):
    def setUp(self):
        collector = _make_collector(300, mean_conf=0.82, fp=1, fn=1)
        self.result = TrainingSufficiencyEvaluator(
            SufficiencyThresholds(min_samples=200)
        ).evaluate(collector)

    def test_as_text_contains_model_name(self):
        report = SufficiencyReport(self.result, model_name="meu-modelo")
        text = report.as_text()
        self.assertIn("meu-modelo", text)

    def test_as_dict_has_required_keys(self):
        report = SufficiencyReport(self.result)
        d = report.as_dict()
        for key in ("status", "score", "is_sufficient", "needs_retraining", "metrics"):
            self.assertIn(key, d)

    def test_as_json_parseable(self):
        import json
        report = SufficiencyReport(self.result)
        parsed = json.loads(report.as_json())
        self.assertIsInstance(parsed, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
