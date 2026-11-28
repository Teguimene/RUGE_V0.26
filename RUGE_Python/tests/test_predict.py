import unittest

import torch

from ruge_python.predict import rank_candidate, rank_candidates


class PredictionTests(unittest.TestCase):
    def test_ranks_candidates_and_filters_known_answers(self):
        ranked = rank_candidates(
            torch.tensor([0.2, 0.9, 0.7, 0.4]),
            blocked_ids={1},
            top_k=2,
        )

        self.assertEqual([entity_id for entity_id, _ in ranked], [2, 3])
        self.assertAlmostEqual(ranked[0][1], 0.7)
        self.assertAlmostEqual(ranked[1][1], 0.4)

    def test_rejects_nonpositive_top_k(self):
        with self.assertRaisesRegex(ValueError, "top_k must be positive"):
            rank_candidates(torch.tensor([0.2]), set(), 0)

    def test_ranks_a_specific_candidate_after_filtering_known_answers(self):
        rank, score = rank_candidate(
            torch.tensor([0.2, 0.9, 0.7, 0.4]),
            candidate_id=2,
            blocked_ids={1},
        )

        self.assertEqual(rank, 1)
        self.assertAlmostEqual(score, 0.7)


if __name__ == "__main__":
    unittest.main()
