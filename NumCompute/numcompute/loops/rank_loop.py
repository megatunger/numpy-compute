import numpy as np
from typing import Union

def rank_loop(a: np.ndarray, method: str='average') -> np.ndarray:
    result = a.tolist()
    n = len(result)

    # pair each value with its original index, then sort by value
    indexed = [(v, i) for i, v in enumerate(result)]
    indexed.sort(key=lambda x: x[0])

    ranks = [0] * n

    if method == 'ordinal':
        # add 1 for ranks to be 1-indexed
        for rank, (value, orig_index) in enumerate(indexed):
            ranks[orig_index] = rank + 1

    elif method == 'dense':
        dense_rank = 0
        prev_value = None
        for rank, (value, orig_index) in enumerate(indexed):
            if value != prev_value:
                dense_rank += 1
                prev_value = value
            ranks[orig_index] = dense_rank

    elif method == 'average':
        # finding length of ties and then getting the average
        i = 0
        while i < n:
            j = i
            while j < n and indexed[j][0] == indexed[i][0]:
                j += 1
            avg = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[indexed[k][1]] = avg
            i = j

    else:
        raise ValueError("Unknown ranking method received! Valid methods are 'average', 'dense', or 'ordinal'.")

    return np.array(ranks)

def percentile_loop(a: np.ndarray, q: Union[float, list[float]], interpolation: str='linear') -> Union[np.ndarray, np.floating]:
    sorted_a = a.tolist()
    sorted_a.sort()
    n = len(sorted_a)

    scalar_input = isinstance(q, float) or isinstance(q, int)
    q_list = [q] if scalar_input else q

    results = []
    for percentile in q_list:
        if percentile < 0.0 or percentile > 100.0:
            raise ValueError(f"Percentile value {percentile} is out of range [0, 100].")

        # map percentile to a floating index in the sorted array
        index = (percentile / 100.0) * (n - 1)
        i = int(index)
        j = min(i + 1, n - 1)
        fraction = index - i

        if interpolation == 'linear':
            value = sorted_a[i] + (sorted_a[j] - sorted_a[i]) * fraction
        elif interpolation == 'lower':
            value = sorted_a[i]
        elif interpolation == 'higher':
            value = sorted_a[j]
        elif interpolation == 'midpoint':
            value = (sorted_a[i] + sorted_a[j]) / 2.0
        else:
            raise ValueError(f"Unknown interpolation '{interpolation}'. Supported: 'linear', 'lower', 'higher', 'midpoint'.")

        results.append(value)

    if scalar_input:
        return np.float64(results[0])

    return np.array(results)
