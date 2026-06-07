import unittest

import numpy as np

from numcompute_stream.stats import RunningStats, StreamingHistogram, update_stats


class TestRunningStats(unittest.TestCase):
    def setUp(self):
        self.chunk1 = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        self.chunk2 = np.array([[4.0, 40.0], [5.0, 50.0]])

    def test_two_chunks_match_batch_mean_and_std(self):
        stats = RunningStats()
        stats.update(self.chunk1)
        stats.update(self.chunk2)

        mean, std = stats.result()
        batch = np.vstack([self.chunk1, self.chunk2])

        np.testing.assert_allclose(mean, batch.mean(axis=0))
        np.testing.assert_allclose(std, batch.std(axis=0))

    def test_single_chunk_matches_batch(self):
        stats = RunningStats()
        stats.update(self.chunk1)

        mean, std = stats.result()
        np.testing.assert_allclose(mean, self.chunk1.mean(axis=0))
        np.testing.assert_allclose(std, self.chunk1.std(axis=0))

    def test_1d_input_treated_as_single_feature(self):
        stats = RunningStats()
        stats.update(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        mean, std = stats.result()
        np.testing.assert_allclose(mean, [3.0])
        np.testing.assert_allclose(std, [np.std([1.0, 2.0, 3.0, 4.0, 5.0])])

    def test_empty_chunk_raises(self):
        stats = RunningStats()
        with self.assertRaises(ValueError):
            stats.update(np.array([]))

    def test_zero_variance_column(self):
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        stats = RunningStats()
        stats.update(X)

        mean, std = stats.result()
        np.testing.assert_allclose(mean, X.mean(axis=0))
        np.testing.assert_allclose(std, X.std(axis=0))

    def test_result_before_update_raises(self):
        stats = RunningStats()
        with self.assertRaises(ValueError):
            stats.result()

    def test_reset_clears_state(self):
        stats = RunningStats()
        stats.update(self.chunk1)
        stats.reset()

        self.assertEqual(stats.n, 0)
        self.assertIsNone(stats.mean_)
        self.assertIsNone(stats.var_)

    def test_update_returns_self(self):
        stats = RunningStats()
        out = stats.update(self.chunk1)
        self.assertIs(out, stats)

    def test_mismatched_feature_count_raises(self):
        stats = RunningStats()
        stats.update(self.chunk1)

        with self.assertRaises(ValueError):
            stats.update(np.array([[1.0, 2.0, 3.0]]))

    def test_ddof_matches_numpy(self):
        stats = RunningStats(ddof=1)
        stats.update(self.chunk1)

        mean, std = stats.result()
        np.testing.assert_allclose(mean, self.chunk1.mean(axis=0))
        np.testing.assert_allclose(std, self.chunk1.std(axis=0, ddof=1))


class TestStreamingHistogram(unittest.TestCase):
    def test_two_chunks_match_batch_histogram(self):
        chunk1 = np.array([1.0, 2.0, 3.0])
        chunk2 = np.array([7.0, 8.0])

        hist = StreamingHistogram(bins=2, range=(0.0, 10.0))
        hist.update(chunk1)
        hist.update(chunk2)

        counts, edges = hist.result()
        expected_counts, expected_edges = np.histogram(
            [1.0, 2.0, 3.0, 7.0, 8.0], bins=2, range=(0.0, 10.0)
        )

        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_array_equal(edges, expected_edges)

    def test_empty_chunk_raises(self):
        hist = StreamingHistogram(bins=2)
        with self.assertRaises(ValueError):
            hist.update(np.array([]))

    def test_result_before_update_raises(self):
        hist = StreamingHistogram(bins=2)
        with self.assertRaises(ValueError):
            hist.result()

    def test_reset_clears_state(self):
        hist = StreamingHistogram(bins=2, range=(0.0, 10.0))
        hist.update(np.array([1.0, 2.0]))
        hist.reset()

        self.assertIsNone(hist.counts_)
        self.assertIsNone(hist.bin_edges_)


class TestUpdateStats(unittest.TestCase):
    def test_delegates_to_stats_update(self):
        stats = RunningStats()
        chunk = np.array([[1.0, 2.0], [3.0, 4.0]])

        out = update_stats(stats, chunk)

        self.assertIs(out, stats)
        mean, std = stats.result()
        np.testing.assert_allclose(mean, chunk.mean(axis=0))
        np.testing.assert_allclose(std, chunk.std(axis=0))


if __name__ == "__main__":
    unittest.main()
