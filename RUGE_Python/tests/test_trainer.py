import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from ruge_python.data import GroundRule
from ruge_python.models import ComplEx, RUGEEmbeddingModel
from ruge_python.trainer import RUGETrainer


class ScalarScoreModel(RUGEEmbeddingModel):
    def __init__(self):
        super().__init__(
            2,
            1,
            1,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        self.logit = nn.Parameter(torch.tensor(0.0, dtype=self.dtype))

    def score(self, triples: torch.Tensor) -> torch.Tensor:
        return self.logit.expand(triples.shape[0])

    def regularization(self, triples: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            triples.shape[0], dtype=self.dtype, device=self.device
        )

    def export_matrices(self) -> dict[str, torch.Tensor]:
        return {}


class TrainerTests(unittest.TestCase):
    def test_rule_polarity_moves_target_score_in_expected_direction(self):
        training = torch.tensor([[0, 0, 1]], dtype=torch.long)
        expected_gradients = {"positive": -0.04, "negative": 0.04}
        for polarity, expected_gradient in expected_gradients.items():
            with self.subTest(polarity=polarity):
                model = ScalarScoreModel()
                rule = GroundRule(
                    triples=((0, 0, 1), (1, 0, 0)),
                    unseen=2,
                    confidence=0.8,
                    polarity=polarity,
                )
                trainer = RUGETrainer(
                    model,
                    training,
                    training,
                    {(0, 0, 1)},
                    [rule],
                    num_negatives=2,
                    num_minibatches=1,
                    num_iterations=1,
                    rule_weight=0.1,
                )

                trainer._rule_loss(torch.tensor([0])).backward()

                self.assertAlmostEqual(
                    model.logit.grad.item(), expected_gradient, places=6
                )

    def test_rejects_negative_rule_target_already_known_positive(self):
        model = ScalarScoreModel()
        training = torch.tensor([[0, 0, 1]], dtype=torch.long)
        rule = GroundRule(
            triples=((0, 0, 1), (1, 0, 0)),
            unseen=2,
            confidence=0.8,
            polarity="negative",
        )

        with self.assertRaisesRegex(ValueError, "already known positive"):
            RUGETrainer(
                model,
                training,
                training,
                {(0, 0, 1), (1, 0, 0)},
                [rule],
                num_negatives=2,
                num_minibatches=1,
                num_iterations=1,
            )

    def test_one_cpu_epoch_writes_java_matrices_and_checkpoint(self):
        training = torch.tensor([[0, 0, 1]], dtype=torch.long)
        validation = torch.tensor([[1, 0, 0]], dtype=torch.long)
        known = {(0, 0, 1), (1, 0, 0)}

        with tempfile.TemporaryDirectory() as directory:
            model = ComplEx(
                2, 1, 2, device=torch.device("cpu"), dtype=torch.float64
            )
            trainer = RUGETrainer(
                model,
                training,
                validation,
                known,
                [],
                num_negatives=2,
                num_minibatches=1,
                num_iterations=1,
                output_iter_skip=1,
                validation_limit=1,
                output_dir=directory,
            )
            metrics = trainer.fit()

            self.assertGreaterEqual(metrics.mrr, 0.0)
            self.assertLessEqual(metrics.mrr, 1.0)
            self.assertTrue((Path(directory) / "ruge_python_best.pt").is_file())
            self.assertTrue((Path(directory) / "MatrixE.real.best").is_file())
            self.assertTrue((Path(directory) / "MatrixR.imag.best").is_file())


if __name__ == "__main__":
    unittest.main()
