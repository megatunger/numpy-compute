"""numcompute_stream package."""

from numcompute_stream.metrics import StreamingAccuracy
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler
from numcompute_stream.stats import RunningStats, StreamingHistogram, update_stats
from numcompute_stream.ensemble import BaggingClassifier
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier

from numcompute_stream.stream import StreamTrainer

__all__ = [
    "RunningStats",
    "StreamingHistogram",
    "StreamingAccuracy",
    "Pipeline",
    "StandardScaler",
    "SimpleImputer",
    "OneHotEncoder",
    "DecisionTreeClassifier",
    "BaggingClassifier",
    "StreamTrainer",
    "update_stats",
]

