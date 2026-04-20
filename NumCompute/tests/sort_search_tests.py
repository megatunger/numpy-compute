import unittest
import numpy as np
from numcompute.sort_search import (
    stable_sort, 
    multi_key_sort, 
    topk, 
    binary_search, 
    quickselect
)

class StableSortTests(unittest.TestCase):

    def test_basic(self):
        a = np.array([3, 1, 2])
        self.assertTrue(np.array_equal(stable_sort(a), np.array([1, 2, 3])))

    def test_sorted(self):
        a = np.array([1, 2, 3])
        self.assertTrue(np.array_equal(stable_sort(a), a))

    def test_reverse_sorted(self):
        a = np.array([3, 2, 1])
        self.assertTrue(np.array_equal(stable_sort(a), np.array([1, 2, 3])))

    def test_repeats(self):
        a = np.array([3, 1, 2, 1])
        self.assertTrue(np.array_equal(stable_sort(a), np.array([1, 1, 2, 3])))

    def test_single(self):
        a = np.array([1])
        self.assertTrue(np.array_equal(stable_sort(a), a))

    def test_equal(self):
        a = np.array([1, 1, 1])
        self.assertTrue(np.array_equal(stable_sort(a), a))

    def test_negative(self):
        a = np.array([-3, -1, -2])
        self.assertTrue(np.array_equal(stable_sort(a), np.array([-3, -2, -1])))

    def test_shape(self):
        a = np.random.rand(3)
        self.assertEqual(stable_sort(a).shape, a.shape)

    def test_stable(self):
        # creating an array that stores value and order as key-value pairs to check later
        keys =   np.array([1, 2, 1, 2])
        orders = np.array([0, 1, 2, 3])
        a = np.array(list(zip(keys, orders)), dtype=[('key', int), ('order', int)])
        result = stable_sort(a)
        self.assertEqual(result[0]['order'], 0)  
        self.assertEqual(result[1]['order'], 2) 
        self.assertEqual(result[2]['order'], 1)
        self.assertEqual(result[3]['order'], 3)

class MultiKeySortTests(unittest.TestCase):
    def test_single_column(self):
        a = np.array([[3, 1],
                      [1, 2],
                      [2, 3]])
        expected = np.array([[1, 2],
                             [2, 3],
                             [3, 1]])
        self.assertTrue(np.array_equal(multi_key_sort(a, [0]), expected))

    def test_two_columns(self):
        a = np.array([[1, 3],
                      [2, 1],
                      [1, 1],
                      [2, 2]])
        expected = np.array([[1, 1],
                             [1, 3],
                             [2, 1],
                             [2, 2]])
        # sorting column 0 first, then 1
        self.assertTrue(np.array_equal(multi_key_sort(a, [0, 1]), expected))

    def test_two_columns_reversed(self):
        a = np.array([[1, 3],
                      [2, 1],
                      [1, 1],
                      [2, 2]])
        expected = np.array([[1, 1],
                             [2, 1],
                             [2, 2],
                             [1, 3]])
        # sorting column 1 first, then 0
        self.assertTrue(np.array_equal(multi_key_sort(a, [1, 0]), expected))

    def test_sorted(self):
        a = np.array([[1, 1],
                      [1, 2],
                      [2, 1]])
        self.assertTrue(np.array_equal(multi_key_sort(a, [0, 1]), a))

    def test_equal_primary_key(self):
        a = np.array([[1, 3],
                      [1, 1],
                      [1, 2]])
        expected = np.array([[1, 1],
                             [1, 2],
                             [1, 3]])
        np.testing.assert_array_equal(multi_key_sort(a, [0, 1]), expected)

    def test_stable(self):
        a = np.array([[1, 1],
                      [1, 1],
                      [2, 2]])
        self.assertTrue(np.array_equal(multi_key_sort(a, [0, 1]), a))

    def test_shape(self):
        a = np.random.rand(3, 2)
        self.assertEqual(multi_key_sort(a, [0, 1]).shape, a.shape)

class TopKTests(unittest.TestCase):
    def test_largest_true(self):
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        values, indices = topk(a, 3)
        self.assertTrue(np.array_equal(values, np.array([9, 6, 5])))
        self.assertTrue(np.array_equal(values, a[indices]))

    def test_largest_false(self):
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        values, indices = topk(a, 3, largest=False)
        self.assertTrue(np.array_equal(values, np.array([1, 1, 2])))
        self.assertTrue(np.array_equal(values, a[indices]))

    def test_return_indices_false(self):
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        values = topk(a, 3, return_indices=False)
        self.assertTrue(np.array_equal(values, np.array([9, 6, 5])))

    def test_k_equals_length(self):
        a = np.array([3, 1, 2])
        values, indices = topk(a, 3)
        self.assertTrue(np.array_equal(values, np.array([3, 2, 1])))
        self.assertTrue(np.array_equal(values, a[indices]))

    def test_k_equals_one(self):
        a = np.array([3, 1, 4, 1, 5])
        values, indices = topk(a, 1)
        self.assertTrue(np.array_equal(values, np.array([5])))
        self.assertTrue(np.array_equal(values, a[indices]))

    def test_negative(self):
        a = np.array([-3, -1, -4, -1, -5])
        values, indices = topk(a, 3, largest=True)
        self.assertTrue(np.array_equal(values, np.array([-1, -1, -3])))
        self.assertTrue(np.array_equal(values, a[indices]))

class BinarySearchTests(unittest.TestCase):
    def test_exist(self):
        a = np.array([1, 2, 3, 4, 5])
        index, exist = binary_search(a, 3)
        self.assertEqual(index, 2)
        self.assertTrue(exist)

    def test_not_exist(self):
        a = np.array([1, 2, 3, 4, 5])
        index, exist = binary_search(a, 6)
        self.assertEqual(index, len(a))
        self.assertTrue(not exist)

    def test_first(self):
        a = np.array([1, 2, 3, 4, 5])
        index, exist = binary_search(a, 1)
        self.assertEqual(index, 0)
        self.assertTrue(exist)

    def test_last(self):
        a = np.array([1, 2, 3, 4, 5])
        index, exist = binary_search(a, 5)
        self.assertEqual(index, 4)
        self.assertTrue(exist)

    def test_repeats(self):
        a = np.array([1, 2, 2, 2, 3])
        index, exist = binary_search(a, 2)
        self.assertEqual(index, 1)  
        self.assertTrue(exist)

    def test_single(self):
        a = np.array([1])
        index, exist = binary_search(a, 2)
        self.assertEqual(index, 1)  
        self.assertTrue(not exist)

    def test_floats(self):
        a = np.array([1.1, 2.2, 3.3, 4.4])
        index, exist = binary_search(a, 4.4)
        self.assertEqual(index, 3)  
        self.assertTrue(exist)

class QuickSelectTests(unittest.TestCase):

    def test_basic(self):
        a = np.array([3, 1, 4, 1, 5])
        self.assertEqual(quickselect(a, 2), 3)

    def test_smallest(self):
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertEqual(quickselect(a, 0), 1)

    def test_largest(self):
        a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertEqual(quickselect(a, len(a) - 1), 9)

    def test_single(self):
        a = np.array([1])
        self.assertEqual(quickselect(a, 0), 1)

    def test_sorted(self):
        a = np.array([1, 2, 3, 4, 5])
        self.assertEqual(quickselect(a, 2), 3)

    def test_repeats(self):
        a = np.array([1, 1, 1, 1, 1])
        self.assertEqual(quickselect(a, 2), 1)

    def test_negative(self):
        a = np.array([-3, -1, -4, -1, -5])
        self.assertEqual(quickselect(a, 4), -1)

if __name__ == '__main__':
    unittest.main()
