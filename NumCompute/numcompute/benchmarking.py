import time
import numpy as np

from numcompute.loops.metrics_loop import (
    accuracy_score_loop,
    precision_score_loop,
    recall_score_loop,
    f1_score_loop,
    confusion_matrix_loop,
    mean_squared_error_loop
)
from numcompute.loops.stats_loop import (
    mean_loop,
    median_loop,
    std_loop,
    min_loop,
    max_loop
)
from numcompute.loops.sort_search_loop import stable_sort_loop
from numcompute import sort_search, metrics, stats


def benchmark(functions, params, repeats=5):
    """
    Benchmark multiple functions using the same parameters.

    functions: dictionary like {"loop": loop_func, "vectorized": vec_func}
    params: dictionary of arguments passed into each function
    repeats: number of times to run each function
    """
    results = {}

    for name, func in functions.items():
        start = time.perf_counter()

        for _ in range(repeats):
            func(**params)

        end = time.perf_counter()
        avg_time = (end - start) / repeats
        results[name] = avg_time

        print(f"{name}: {avg_time:.6f} seconds")

    if len(results) >= 2:
        names = list(results.keys())
        first = names[0]

        for name in names[1:]:
            speedup = results[first] / results[name]
            print(f"{name} is {speedup:.2f}x compared to {first}")

    return results


BENCHMARKS = {
    "sort_search": {
        "stable_sort": {
            "functions": {
                "loop": stable_sort_loop,
                "vectorized": sort_search.stable_sort,
            },
            "params": {
                "a": np.random.rand(10000),
            },
        },
    },
    "metrics": {
        "accuracy_score": {
            "functions": {
                "loop": accuracy_score_loop,
                "vectorized": metrics.accuracy_score
            },
            "params": {
                "y_true": np.random.randint(0,2,10000),
                "y_pred": np.random.randint(0,2,10000),
            },
        },
        "precision_score": {
            "functions": {
                "loop": precision_score_loop,
                "vectorized": metrics.precision_score
            },
            "params": {
                "y_true": np.random.randint(0,2,10000),
                "y_pred": np.random.randint(0,2,10000),
            },
        },
        "recall_score": {
            "functions": {
                "loop": recall_score_loop,
                "vectorized": metrics.recall_score
            },
            "params": {
                "y_true": np.random.randint(0,2,10000),
                "y_pred": np.random.randint(0,2,10000),
            },
        },
        "f1_score": {
            "functions": {
                "loop": f1_score_loop,
                "vectorized": metrics.f1_score
            },
            "params": {
                "y_true": np.random.randint(0,2,10000),
                "y_pred": np.random.randint(0,2,10000),
            },
        },
        "confusion_matrix": {
            "functions": {
                "loop": confusion_matrix_loop,
                "vectorized": metrics.confusion_matrix
            },
            "params": {
                "y_true": np.random.randint(0,2,10000),
                "y_pred": np.random.randint(0,2,10000),
            },
        },
        "mean_squared_error": {
            "functions": {
                "loop": mean_squared_error_loop,
                "vectorized": metrics.mean_squared_error
            },
            "params": {
                "y_true": np.random.rand(10000),
                "y_pred": np.random.rand(10000),
            },
        }
    },
    "stats": {
        "mean": {
            "functions": {
                "loop": mean_loop,
                "vectorized": stats.mean
            },
            "params": {
                "arr": np.random.rand(10000),
            },
        },
        "median": {
            "functions": {
                "loop": median_loop,
                "vectorized": stats.median
            },
            "params": {
                "arr": np.random.rand(10000),
            },
        },
        "std": {
            "functions": {
                "loop": std_loop,
                "vectorized": stats.std
            },
            "params": {
                "arr": np.random.rand(10000),
            },
        },
        "min": {
            "functions": {
                "loop": min_loop,
                "vectorized": stats.min
            },
            "params": {
                "arr": np.random.rand(10000),
            },
        },
        "max": {
            "functions": {
                "loop": max_loop,
                "vectorized": stats.max
            },
            "params": {
                "arr": np.random.rand(10000),
            },
        },
    },
}


def run_benchmark_group(group_name, repeats=5):
    group = BENCHMARKS[group_name]

    print(f"Start benchmark {group_name}.py")

    for test_name, test in group.items():
        print(f"\nBenchmark: {test_name}")
        benchmark(test["functions"], test["params"], repeats=repeats)


def main():
    for group_name in BENCHMARKS:
        run_benchmark_group(group_name, repeats=5)


if __name__ == "__main__":
    main()
