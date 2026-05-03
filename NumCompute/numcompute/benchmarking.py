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

from numcompute import optim
from numcompute import preprocessing
from numcompute.loops.optim_loop import grad_loop, jacobian_loop
from numcompute.loops.sort_search_loop import stable_sort_loop
from numcompute.loops.preprocessing_loop import (
    MinMaxScalerLoop,
    OneHotEncoderLoop,
    SimpleImputerLoop,
    StandardScalerLoop,
)
from numcompute.loops.sort_search_loop import (
    stable_sort_loop, 
    multi_key_sort_loop, 
    topk_loop, 
    binary_search_loop, 
    quickselect_loop,
)
from numcompute.sort_search import (
    stable_sort, 
    multi_key_sort, 
    topk, 
    binary_search, 
    quickselect
)
from numcompute.loops.rank_loop import rank_loop, percentile_loop
from numcompute.rank import rank, percentile


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
    "optim": {
        "grad": {
            "functions": {
                "loop": grad_loop,
                "vectorized": optim.grad,
            },
            "params": {
                "f": lambda X: np.sum(X ** 2, axis=1),
                "x": np.array([1.0, 2.0, 3.0, 4.0]),
            },
        },
        "jacobian": {
            "functions": {
                "loop": jacobian_loop,
                "vectorized": optim.jacobian,
            },
            "params": {
                "F": lambda X: X ** 2,
                "x": np.array([1.0, 2.0, 3.0, 4.0]),
            },
        },
    },
    "preprocessing": {
        "standard_scaler": {
            "functions": {
                "loop": StandardScalerLoop().fit_transform,
                "vectorized": preprocessing.StandardScaler().fit_transform,
            },
            "params": {
                "X": np.random.rand(1000, 10),
            },
        },
        "minmax_scaler": {
            "functions": {
                "loop": MinMaxScalerLoop().fit_transform,
                "vectorized": preprocessing.MinMaxScaler().fit_transform,
            },
            "params": {
                "X": np.random.rand(1000, 10),
            },
        },
        "one_hot_encoder": {
            "functions": {
                "loop": OneHotEncoderLoop().fit_transform,
                "vectorized": preprocessing.OneHotEncoder().fit_transform,
            },
            "params": {
                "X": np.random.choice(["red", "blue", "green"], size=(1000, 3)),
            },
        },
        "simple_imputer": {
            "functions": {
                "loop": SimpleImputerLoop().fit_transform,
                "vectorized": preprocessing.SimpleImputer().fit_transform,
            },
            "params": {
                "X": np.where(
                    np.random.rand(1000, 10) < 0.1,
                    np.nan,
                    np.random.rand(1000, 10),
                ),
            },
        },
    },
    "sort_search": {
        "stable_sort": {
            "functions": {
                "loop": stable_sort_loop,
                "vectorized": stable_sort,
            },
            "params": {
                "a": np.random.rand(10000),
            },
        },
        "multi_key_sort": {
            "functions": {
                "loop": multi_key_sort_loop,
                "vectorized": multi_key_sort,
            },
            "params": {
                "a": np.random.rand(1000, 3),
                "columns": [0, 1, 2],
            },
        },
        "topk": {
            "functions": {
                "loop": topk_loop,
                "vectorized": topk,
            },
            "params": {
                "a": np.random.rand(1000),
                "k": 10,
            },
        },
        "binary_search": {
            "functions": {
                "loop": binary_search_loop,
                "vectorized": binary_search,
            },
            "params": {
                "a": np.sort(np.random.rand(1000)),
                "x": 0.5,
            },
        },
        "quickselect": {
            "functions": {
                "loop": quickselect_loop,
                "vectorized": quickselect,
            },
            "params": {
                "a": np.random.rand(1000),
                "k": 100,
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
    "rank": {
        "rank": {
            "functions": {
                "loop": rank_loop,
                "vectorized": rank,
            },
            "params": {
                "a": np.random.rand(1000),
                "method": np.random.choice(['average', 'ordinal', 'dense'])
            },
        },
        "percentile": {
            "functions": {
                "loop": percentile_loop,
                "vectorized": percentile,
            },
            "params": {
                "a": np.random.rand(1000),
                "q": np.random.randint(1, 100),
                "interpolation": np.random.choice(['linear', 'lower', 'higher', 'midpoint'])
            },
        },
    },
}


def run_benchmark_group(group_name, repeats=5):
    group = BENCHMARKS[group_name]
    group_results = {}

    print(f"Start benchmark {group_name}.py")

    for test_name, test in group.items():
        print("-" * 50)
        print(f"\nBenchmark: {test_name}")
        result = benchmark(test["functions"], test["params"], repeats=repeats)

        if "loop" in result and "vectorized" in result:
            result["speedup"] = result["loop"] / result["vectorized"]

        group_results[test_name] = result

    return group_results


def main():
    all_results = {}

    for group_name in BENCHMARKS:
        all_results[group_name] = run_benchmark_group(group_name, repeats=5)

    return all_results


if __name__ == "__main__":
    main()
