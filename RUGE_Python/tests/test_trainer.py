import tempfile
import unittest
from pathlib import Path

import torch

from ruge_python.models import ComplEx
from ruge_python.trainer import RUGETrainer


class TrainerTests(unittest.TestCase):
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
