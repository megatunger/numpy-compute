# NumCompute Stream — Assignment 2.2

Streaming machine learning on top of NumCompute. Instead of loading the whole dataset and calling fit once, you feed data in chunks and call partial_fit each time — like data is still arriving.

Built with plain python, numpy, and matplotlib only.

## What we built

numcompute_stream is the package for this assignment. It sits next to the original numcompute batch code (we did not change numcompute for 2.2).

| Module        | What it does                                                  |
| ------------- | ------------------------------------------------------------- |
| stats         | Running mean/var (Welford), histograms, update_stats          |
| metrics       | StreamingAccuracy — accuracy that builds up chunk by chunk    |
| preprocessing | StandardScaler, SimpleImputer, OneHotEncoder with partial_fit |
| pipeline      | Chains preprocessors + model, partial_fit on each step        |
| tree          | DecisionTreeClassifier — grows splits as new chunks arrive    |
| ensemble      | BaggingClassifier — bootstrap sample per tree each chunk      |
| stream        | StreamTrainer — fit_chunk, score_chunk, per-chunk log         |
| visualise     | Plots for accuracy over time and model comparison             |

## Install

    cd NumCompute
    pip install -e .

You need Python 3.11+, numpy, and matplotlib.

## Quick example

    import numpy as np
    from numcompute.io import load_csv
    from numcompute_stream.pipeline import Pipeline
    from numcompute_stream.preprocessing import StandardScaler
    from numcompute_stream.tree import DecisionTreeClassifier
    from numcompute_stream.ensemble import BaggingClassifier
    from numcompute_stream.stream import StreamTrainer

    wine = load_csv("assets/winequality-white.csv", dtype=np.float32, delimiter=";")
    X = wine[:, :11]
    y = (wine[:, 11] >= 6).astype(int)

    chunk_size = 50
    chunks = [(X[i:i + chunk_size], y[i:i + chunk_size]) for i in range(0, len(X), chunk_size)]

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=5)),
    ])
    trainer = StreamTrainer(pipe)

    for X_chunk, y_chunk in chunks:
        trainer.fit_chunk(X_chunk, y_chunk)

    print("cumulative accuracy:", trainer.log_[-1]["cumulative_accuracy"])

Swap DecisionTreeClassifier for BaggingClassifier to try the ensemble.

## Demo notebook

demo/stream_demo.ipynb walks through the full flow on wine data:

- load csv and binarise quality (>= 6 is good)
- split into chunks of 50
- train tree vs bagging with StreamTrainer
- plot accuracy over chunks

Open it from the NumCompute folder or demo/ — the first cell fixes the import path.

## Benchmark

benchmark/streaming_models.py compares tree vs bagging on the same chunks and prints:

- average partial_fit time per chunk
- average predict time per chunk
- final chunk accuracy and cumulative accuracy

It also saves plots to benchmark/plots/

    python benchmark/streaming_models.py

## Tests

    pytest tests/numcompute_stream/

There are 70+ tests covering stats, metrics, preprocessing, pipeline, tree, ensemble, stream, and visualise.
