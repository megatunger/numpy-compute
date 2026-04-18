# CSV loading utilities for NumCompute.

from collections.abc import Iterator, Sequence
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["load_csv"]

# 10k feels like a decent chunk size for now.
# Might be worth making this configurable later.
_DEFAULT_CHUNK_SIZE: int = 10_000

_FilePath = str | PathLike[str]
_DTypeLike = type[Any] | np.dtype[Any] | str
_MissingValues = str | Sequence[str] | None
_FillingValues = float | int | str | None


def load_csv(
    filepath: _FilePath,
    delimiter: str = ",",
    dtype: _DTypeLike = float,
    missing_values: _MissingValues = "",
    filling_values: _FillingValues = np.nan,
) -> np.ndarray:
    # Load a homogeneous CSV file into a NumPy array.
    # Assumes the first row is a header, so it gets skipped.
    chunks: list[np.ndarray] = []

    for chunk in _iter_csv_chunks(
        filepath,
        delimiter,
        dtype,
        missing_values,
        filling_values,
    ):
        chunks.append(chunk)

    if len(chunks) == 0:
        return np.empty((0, 0), dtype=dtype)

    return np.vstack(chunks)


def _iter_csv_chunks(
    filepath: _FilePath,
    delimiter: str,
    dtype: _DTypeLike,
    missing_values: _MissingValues,
    filling_values: _FillingValues,
) -> Iterator[np.ndarray]:
    # Yield parsed CSV chunks after skipping the header row.
    lines: list[str] = []

    with Path(filepath).open("r", encoding="utf-8", newline="") as file:
        # Skip header row.
        next(file, None)

        for line in file:
            lines.append(line)

            if len(lines) >= _DEFAULT_CHUNK_SIZE:
                chunk = _parse_chunk(
                    lines,
                    delimiter,
                    dtype,
                    missing_values,
                    filling_values,
                )

                if chunk is not None:
                    yield chunk

                lines = []

    # Handle leftover lines at the end.
    last_chunk = _parse_chunk(
        lines,
        delimiter,
        dtype,
        missing_values,
        filling_values,
    )
    if last_chunk is not None:
        yield last_chunk


def _parse_chunk(
    lines: list[str],
    delimiter: str,
    dtype: _DTypeLike,
    missing_values: _MissingValues,
    filling_values: _FillingValues,
) -> np.ndarray | None:
    # Parse raw CSV lines into a 2D NumPy array.
    if not lines:
        return None

    raw_text = "".join(lines)

    array = np.genfromtxt(
        StringIO(raw_text),
        delimiter=delimiter,
        dtype=dtype,
        missing_values=missing_values,
        filling_values=filling_values,
    )

    if array.size == 0:
        return None

    # genfromtxt can flatten shapes depending on input size.
    if array.ndim == 0:
        return array.reshape(1, 1)

    if array.ndim == 1:
        if len(lines) == 1:
            return array.reshape(1, -1)
        return array.reshape(-1, 1)

    return array