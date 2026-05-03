import time

import numpy as np

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
from numcompute import sort_search


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
                "vectorized": sort_search.stable_sort,
            },
            "params": {
                "a": sort_search.np.random.rand(1000),
            },
        },
    },
}


def run_benchmark_group(group_name, repeats=5):
    group = BENCHMARKS[group_name]

    print(f"Start benchmark {group_name}.py")

    for test_name, test in group.items():
        print("-" * 50)
        print(f"\nBenchmark: {test_name}")
        benchmark(test["functions"], test["params"], repeats=repeats)


def main():
    for group_name in BENCHMARKS:
        run_benchmark_group(group_name, repeats=5)


if __name__ == "__main__":
    main()
