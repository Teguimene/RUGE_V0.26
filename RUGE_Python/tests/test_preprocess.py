import tempfile
import unittest
from pathlib import Path

from ruge_python.preprocess import convert_data_form, ground_all_rules


class PreprocessingTests(unittest.TestCase):
    def test_conversion_reuses_existing_java_id_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entity_map = root / "entityid.txt"
            relation_map = root / "relationid.txt"
            entity_map.write_text("0\tAlice\n1\tBob\n", encoding="utf-8")
            relation_map.write_text("0\tknows\n", encoding="utf-8")
            raw_paths = [root / name for name in ("train.raw", "valid.raw", "test.raw")]
            for path in raw_paths:
                path.write_text("Alice\tknows\tBob\n", encoding="utf-8")
            outputs = [root / name for name in ("train.txt", "valid.txt", "test.txt")]

            counts = convert_data_form(
                entity_map,
                relation_map,
                *raw_paths,
                *outputs,
            )

            self.assertEqual(counts, (2, 1))
            self.assertTrue(all(path.read_text(encoding="utf-8") == "0\t0\t1\n"
                                for path in outputs))

    def test_grounds_one_and_two_premise_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relation_map = root / "relationid.txt"
            relation_map.write_text(
                "0\t<r1>\n1\t<r2>\n2\t<r3>\n", encoding="utf-8"
            )
            training = root / "train.txt"
            training.write_text("0\t0\t1\n1\t1\t2\n", encoding="utf-8")
            rules = root / "rules.txt"
            rules.write_text(
                "?a  <r1>  ?b => ?a  <r2>  ?b\t0.8\n"
                "?a  <r1>  ?b  ?b  <r2>  ?c => ?a  <r3>  ?c\t0.9\n",
                encoding="utf-8",
            )
            output = root / "groundings.txt"

            result = ground_all_rules(relation_map, rules, training, output)

            self.assertEqual(result, (2, 0))
            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                [
                    "2\t(0\t0\t1)\t(0\t1\t1)\t0.8",
                    "3\t(0\t0\t1)\t(1\t1\t2)\t(0\t2\t2)\t0.9",
                ],
            )


if __name__ == "__main__":
    unittest.main()
