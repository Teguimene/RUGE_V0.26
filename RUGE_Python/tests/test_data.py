import tempfile
import unittest
from pathlib import Path

from ruge_python.data import load_groundings, load_triples


class DataLoadingTests(unittest.TestCase):
    def test_loads_numeric_triples_and_validates_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.txt"
            path.write_text("0\t1\t2\n", encoding="utf-8")
            triples = load_triples(path, num_entities=3, num_relations=2)
            self.assertEqual(triples.tolist(), [[0, 1, 2]])

            path.write_text("0\t2\t2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_triples(path, num_entities=3, num_relations=2)

    def test_loads_two_and_three_triple_groundings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groundings.txt"
            path.write_text(
                "2\t(0\t1\t2)\t(2\t0\t1)\t0.8\n"
                "3\t(0\t1\t2)\t(2\t0\t1)\t(1\t1\t0)\t0.9\n",
                encoding="utf-8",
            )
            rules = load_groundings(path, num_entities=3, num_relations=2)
            self.assertEqual(len(rules), 2)
            self.assertEqual(rules[0].target, (2, 0, 1))
            self.assertAlmostEqual(rules[0].confidence, 0.8)
            self.assertEqual(rules[0].polarity, "positive")
            self.assertEqual(rules[1].target, (1, 1, 0))

    def test_loads_explicit_positive_and_negative_polarities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groundings.txt"
            path.write_text(
                "2\tpositive\t(0\t1\t2)\t(2\t0\t1)\t0.8\n"
                "2\tnegative\t(0\t1\t2)\t(2\t0\t1)\t0.9\n",
                encoding="utf-8",
            )

            rules = load_groundings(path, num_entities=3, num_relations=2)

            self.assertEqual(
                [rule.polarity for rule in rules], ["positive", "negative"]
            )
            self.assertEqual(rules[1].target, (2, 0, 1))


if __name__ == "__main__":
    unittest.main()
