# Author: DUY ANH
# - Mean, median, standard deviation, min, max
# - Histogram
# - Quantiles (with NaN handling)
# - Axis-wise stats with clear dimension/shape behaviour

from math import sqrt
    
def mean_loop(arr):
    return sum(arr) / len(arr)


def median_loop(arr):
    arr = sorted(arr)
    
    n = len(arr)
    
    if n % 2 == 0:
        return (arr[n // 2] + arr[n // 2 + 1]) / 2
    
    return arr[(n+1)//2]

def std_loop(arr):
    mean = mean_loop(arr)
    
    sum_sq = 0
    for a in arr:
        sum_sq += (a - mean) ** 2
        
    return sqrt(sum_sq / len(arr))


def min_loop(arr):
    return min(arr)

def max_loop(arr):
    return max(arr)


def histogram_loop(arr):
    return 

def quantile_loop(arr, q):
    return 
