import tempfile
import unittest
from pathlib import Path

from ruge_python.data import load_groundings
from ruge_python.preprocess import (
    analyze_inverse_exclusions,
    ground_inverse_exclusions,
)


class InverseExclusionMiningTests(unittest.TestCase):
    def test_scores_inverse_candidates_and_applies_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relation_map = root / "relationid.txt"
            relation_map.write_text(
                "0\t<asymmetric>\n1\t<self_loop>\n2\t<one_way>\n",
                encoding="utf-8",
            )
            training = root / "train.txt"
            training.write_text(
                "0\t0\t1\n"
                "1\t0\t0\n"
                "0\t0\t2\n"
                "0\t1\t1\n"
                "0\t2\t3\n"
                "4\t2\t5\n"
                "6\t2\t7\n"
                "8\t2\t9\n"
                "0\t2\t3\n",
                encoding="utf-8",
            )

            results = analyze_inverse_exclusions(
                relation_map,
                training,
                min_support=4,
                min_confidence=0.8,
            )

            by_relation = {result.relation_id: result for result in results}
            self.assertEqual(by_relation[0].support, 3)
            self.assertEqual(by_relation[0].violations, 2)
            self.assertAlmostEqual(by_relation[0].confidence_smoothed, 0.4)
            self.assertFalse(by_relation[0].retained)
            self.assertEqual(by_relation[1].support, 1)
            self.assertFalse(by_relation[1].retained)
            self.assertEqual(by_relation[2].support, 4)
            self.assertEqual(by_relation[2].violations, 0)
            self.assertAlmostEqual(
                by_relation[2].confidence_smoothed, 5 / 6
            )
            self.assertTrue(by_relation[2].retained)

    def test_writes_negative_groundings_and_skips_known_positive_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relation_map = root / "relationid.txt"
            relation_map.write_text(
                "0\t<low-confidence>\n1\t<one-way>\n",
                encoding="utf-8",
            )
            training = root / "train.txt"
            training.write_text(
                "0\t0\t1\n"
                "1\t0\t0\n"
                "0\t1\t2\n"
                "3\t1\t4\n"
                "5\t1\t6\n"
                "7\t1\t8\n",
                encoding="utf-8",
            )
            known_positive = root / "valid.txt"
            known_positive.write_text("2\t1\t0\n", encoding="utf-8")
            output = root / "inverse_groundings.txt"

            generated, skipped = ground_inverse_exclusions(
                relation_map,
                training,
                output,
                known_positive_paths=(known_positive,),
                min_support=4,
                min_confidence=0.8,
            )

            self.assertEqual((generated, skipped), (3, 1))
            rules = load_groundings(output, num_entities=9, num_relations=2)
            self.assertEqual(len(rules), 3)
            self.assertTrue(all(rule.polarity == "negative" for rule in rules))
            self.assertEqual(
                [rule.target for rule in rules],
                [(4, 1, 3), (6, 1, 5), (8, 1, 7)],
            )

    def test_can_limit_groundings_to_selected_relation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relation_map = root / "relationid.txt"
            relation_map.write_text(
                "0\t<hasChild>\n1\t<wasBornIn>\n",
                encoding="utf-8",
            )
            training = root / "train.txt"
            training.write_text("0\t0\t1\n2\t1\t3\n", encoding="utf-8")
            output = root / "has_child_groundings.txt"

            generated, skipped = ground_inverse_exclusions(
                relation_map,
                training,
                output,
                relation_names=("<hasChild>",),
                min_support=1,
                min_confidence=0.6,
            )

            self.assertEqual((generated, skipped), (1, 0))
            rules = load_groundings(output, num_entities=4, num_relations=2)
            self.assertEqual(len(rules), 1)
            self.assertEqual(rules[0].triples, ((0, 0, 1), (1, 0, 0)))
            self.assertEqual(rules[0].target, (1, 0, 0))


if __name__ == "__main__":
    unittest.main()
