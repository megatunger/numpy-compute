from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from numcompute.io import load_csv


class TestLoadCsv(unittest.TestCase):
    def test_load_csv_skips_header_and_reads_numbers(self):
        with TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "data.csv"
            csv_text = "x,y,z\n1,2,3\n4,5,6\n"
            file_path.write_text(csv_text, encoding="utf-8")

            loaded = load_csv(file_path)

        expected = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        np.testing.assert_array_equal(loaded, expected)

    def test_load_csv_handles_missing_values(self):
        with TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "missing.csv"
            csv_text = "x,y,z\n1,,3\n4,5,\n"
            file_path.write_text(csv_text, encoding="utf-8")

            loaded = load_csv(
                file_path,
                missing_values="",
                filling_values=-1.0,
            )

        expected = np.array([
            [1.0, -1.0, 3.0],
            [4.0, 5.0, -1.0],
        ])
        np.testing.assert_array_equal(loaded, expected)


if __name__ == "__main__":
    unittest.main()