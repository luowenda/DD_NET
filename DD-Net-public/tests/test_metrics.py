import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import numpy as np
    from ddnet_revision.metrics import RevisionEvaluator
except ImportError:
    np = None
    RevisionEvaluator = None


@unittest.skipIf(RevisionEvaluator is None, "Install numpy/scikit-image for numerical metric tests")
class MetricTests(unittest.TestCase):
    def test_pooled_categorical_and_sample_frame_continuous_aggregation(self):
        target = np.zeros((1, 2, 8, 8), dtype=np.float32)
        target[:, :, :4, :4] = 20.0
        prediction = target.copy()
        prediction[:, 0] += 1.0
        prediction[:, 1] = 0.0
        evaluator = RevisionEvaluator(output_frames=2, thresholds_dbz=(10.0,))
        evaluator.update(prediction, target)
        result = evaluator.result()
        threshold = result["overall"]["thresholds"]["10"]
        self.assertEqual((threshold["TP"], threshold["FP"], threshold["FN"]), (16, 0, 16))
        self.assertEqual((threshold["CSI"], threshold["FAR"], threshold["POD"]), (0.5, 0.0, 0.5))
        self.assertEqual(result["overall"]["MSE"], 50.5)
        self.assertEqual(result["overall"]["MAE"], 3.0)

    def test_zero_denominators_return_zero(self):
        empty = np.zeros((1, 1, 8, 8), dtype=np.float32)
        evaluator = RevisionEvaluator(output_frames=1, thresholds_dbz=(10.0,))
        evaluator.update(empty, empty)
        metrics = evaluator.result()["overall"]["thresholds"]["10"]
        self.assertEqual((metrics["CSI"], metrics["FAR"], metrics["POD"]), (0.0, 0.0, 0.0))

    def test_threshold_is_greater_equal_not_greater(self):
        value = np.full((1, 1, 8, 8), 10.0)
        evaluator = RevisionEvaluator(output_frames=1, thresholds_dbz=(10.0,))
        evaluator.update(value, value)
        self.assertEqual(evaluator.result()["overall"]["thresholds"]["10"]["TP"], 64)

    def test_overall_csi_is_not_arithmetic_mean_of_lead_scores(self):
        target = np.zeros((1, 2, 8, 8), dtype=np.float32)
        target[0, 0, 0, 0] = 20
        target[0, 1, 0, :3] = 20
        prediction = np.zeros_like(target)
        prediction[0, 0, 0, 0] = 20
        evaluator = RevisionEvaluator(output_frames=2, thresholds_dbz=(10.0,))
        evaluator.update(prediction, target)
        result = evaluator.result()
        self.assertEqual(result["overall"]["thresholds"]["10"]["CSI"], 0.25)
        self.assertEqual([lead["minutes"] for lead in result["by_lead_time"]], [6, 12])

    def test_nonfinite_inputs_are_rejected(self):
        value = np.zeros((1, 1, 8, 8))
        value[0, 0, 0, 0] = np.nan
        evaluator = RevisionEvaluator(output_frames=1)
        with self.assertRaises(ValueError):
            evaluator.update(value, np.zeros_like(value))
