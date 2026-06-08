"""Compare single tree vs bagging when training chunk by chunk"""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcompute.io import load_csv
from numcompute_stream.ensemble import BaggingClassifier
from numcompute_stream.metrics import StreamingAccuracy
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.visualise import compare_models, plot_metric_over_time


def load_wine_chunks(chunk_size=50):
    """Load white wine data and split into streaming chunks"""
    wine = load_csv(
        ROOT / "assets" / "winequality-white.csv",
        dtype=np.float32,
        delimiter=";",
    )
    X = wine[:, :11]
    y = (wine[:, 11] >= 6).astype(int)
    chunks = [
        (X[i:i + chunk_size], y[i:i + chunk_size])
        for i in range(0, len(X), chunk_size)
    ]
    return chunks


def build_models():
    """Same pipelines as the demo notebook"""
    return {
        "Tree": Pipeline([
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=5, min_samples_split=2)),
        ]),
        "Bagging": Pipeline([
            ("scale", StandardScaler()),
            (
                "model",
                BaggingClassifier(n_estimators=5, max_depth=5, random_state=0),
            ),
        ]),
    }


def run_benchmark(chunks):
    """Train both models on every chunk and record time + accuracy"""
    models = build_models()
    results = {
        name: {
            "fit_time": [],
            "predict_time": [],
            "accuracy": [],
            "cumulative_accuracy": [],
        }
        for name in models
    }
    cumulative = {name: StreamingAccuracy() for name in models}

    for X_chunk, y_chunk in chunks:
        for name, pipe in models.items():
            # step 1: time partial_fit on this chunk
            t0 = time.perf_counter()
            pipe.partial_fit(X_chunk, y_chunk)
            results[name]["fit_time"].append(time.perf_counter() - t0)

            # step 2: time predict and score
            t0 = time.perf_counter()
            y_pred = pipe.predict(X_chunk)
            results[name]["predict_time"].append(time.perf_counter() - t0)

            chunk_acc = float(np.mean(y_pred == y_chunk))
            results[name]["accuracy"].append(chunk_acc)
            cumulative[name].update(y_chunk, y_pred)
            results[name]["cumulative_accuracy"].append(cumulative[name].result())

    return results


def print_summary(results):
    """Print a simple table for the report"""
    header = (
        f"{'Model':<10} {'avg fit (ms)':>14} {'avg predict (ms)':>18} "
        f"{'final chunk acc':>18} {'final cumulative':>18}"
    )
    print(header)
    print("-" * len(header))

    for name, stats in results.items():
        avg_fit = np.mean(stats["fit_time"]) * 1000
        avg_predict = np.mean(stats["predict_time"]) * 1000
        final_chunk = stats["accuracy"][-1]
        final_cumulative = stats["cumulative_accuracy"][-1]
        print(
            f"{name:<10} {avg_fit:>14.2f} {avg_predict:>18.2f} "
            f"{final_chunk:>18.3f} {final_cumulative:>18.3f}"
        )


def save_plots(results, output_dir):
    """Save accuracy plots next to this script"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_over_time(
        results["Tree"]["accuracy"],
        "Tree chunk accuracy",
        "Accuracy",
        save_path=output_dir / "tree_accuracy.png",
    )
    compare_models(
        results["Tree"]["accuracy"],
        results["Bagging"]["accuracy"],
        ["Tree", "Bagging"],
        save_path=output_dir / "tree_vs_bagging.png",
    )
    print(f"plots saved to {output_dir.resolve()}")


def main():
    chunk_size = 50
    chunks = load_wine_chunks(chunk_size=chunk_size)
    print(f"benchmark: {len(chunks)} chunks of up to {chunk_size} rows\n")

    results = run_benchmark(chunks)
    print_summary(results)

    plot_dir = Path(__file__).resolve().parent / "plots"
    save_plots(results, plot_dir)


if __name__ == "__main__":
    main()
