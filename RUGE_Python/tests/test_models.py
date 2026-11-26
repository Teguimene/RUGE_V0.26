import math
import unittest

import torch

from ruge_python.models import ComplEx, RotatE
from ruge_python.trainer import _JavaLogisticLoss


class ModelScoreTests(unittest.TestCase):
    def test_complex_score_matches_java_formula(self):
        model = ComplEx(2, 1, 1, torch.device("cpu"), torch.float64)
        with torch.no_grad():
            model.entity_real.copy_(torch.tensor([[2.0], [3.0]], dtype=torch.float64))
            model.entity_imag.copy_(torch.tensor([[5.0], [7.0]], dtype=torch.float64))
            model.relation_real.fill_(11.0)
            model.relation_imag.fill_(13.0)

        score = model.score(torch.tensor([[0, 0, 1]], dtype=torch.long))
        expected = 2 * 11 * 3 + 5 * 11 * 7 + 2 * 13 * 7 - 5 * 13 * 3
        self.assertAlmostEqual(score.item(), expected)

    def test_rotate_score_is_zero_distance_at_quarter_turn(self):
        model = RotatE(2, 1, 1, margin=12.0, device=torch.device("cpu"))
        with torch.no_grad():
            model.entity_real.copy_(torch.tensor([[1.0], [0.0]], dtype=torch.float64))
            model.entity_imag.copy_(torch.tensor([[0.0], [1.0]], dtype=torch.float64))
            model.relation_phase.fill_(math.pi / 2.0)

        score = model.score(torch.tensor([[0, 0, 1]], dtype=torch.long))
        self.assertAlmostEqual(score.item(), 12.0, places=12)

    def test_rotate_phase_wrap_stays_in_interval(self):
        model = RotatE(1, 1, 2, device=torch.device("cpu"))
        with torch.no_grad():
            model.relation_phase.copy_(
                torch.tensor([[3.5, -4.0]], dtype=torch.float64)
            )
        model.wrap_phases_()
        self.assertTrue(torch.all(model.relation_phase >= -math.pi))
        self.assertTrue(torch.all(model.relation_phase < math.pi))

    def test_logistic_gradient_uses_java_sigmoid_cutoffs(self):
        scores = torch.tensor(
            [11.0, -11.0, 0.0], dtype=torch.float64, requires_grad=True
        )
        labels = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
        _JavaLogisticLoss.apply(scores, labels).sum().backward()
        self.assertEqual(scores.grad[0].item(), 0.0)
        self.assertEqual(scores.grad[1].item(), 0.0)
        self.assertAlmostEqual(scores.grad[2].item(), -0.5)


if __name__ == "__main__":
    unittest.main()
