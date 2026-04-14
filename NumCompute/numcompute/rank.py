import numpy as np

def rank(a: np.ndarray, method: str='average') -> np.ndarray:

    ranks = np.empty(a.size)

    # ranks ties based on order of appearance
    if method == 'ordinal':
        order = np.argsort(a, kind='stable')
        print(order)
        return np.argsort(order, kind='stable') + 1

    a_sorted, ranks, counts = np.unique(a, return_inverse=True, return_counts=True)
    # print(a_sorted)
    # print(counts)

    if method == 'dense':
        return ranks + 1

    if method == 'average':
        # get the starting rank of each ties
        # the first tie always start at 1 so we add 1 at the start
        # the following ones = the prev + the next count
        start_ranks = np.cumsum(np.concatenate(([1], counts[:-1])))
        # average ranking = start rank + last rank / 2
        average_ranks = start_ranks + (counts - 1) / 2.0
        # apply the dense rank to get the correct order
        return average_ranks[ranks]

    raise ValueError("Unknown ranking method received! Valid methods are 'average', 'dense', or 'ordinal'.")

# a = np.array([4, 4, 1, 2, 2, 3])
# print(rank(a, 'ordinal'))
# print(rank(a, 'dense'))
# print("### Average")
# print(rank(a, 'average'))
# print(a)

def percentile(a: np.ndarray, q, interpolation: str='linear') -> np.ndarray | np.floating:
    
    methods = ['linear', 'lower', 'higher', 'midpoint']
    if interpolation not in methods:
        raise ValueError("Unknown interpolation method received! Valid methods are 'linear', 'lower', 'higher', or 'midpoint'.")

    return np.percentile(a, q, method=interpolation)

# a = np.array([[10, 7, 4], [3, 2, 1]])
# print(percentile(a, 50))
