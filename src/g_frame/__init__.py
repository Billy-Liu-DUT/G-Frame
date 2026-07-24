"""G-Frame v2 core workflow package.

The package preserves v1 directory-level entry points while providing typed,
testable implementations for the Team Game and decision-loop core.
"""

from .paths import REPO_ROOT
from .augmentation import AugmentedSource, SixStyleAugmenter
from .pipeline import PipelineStageHandlers, run_mock_pipeline
from .schemas import DecisionPlan, TrainingRecord

__version__ = "0.2.0"

__all__ = [
    "REPO_ROOT",
    "AugmentedSource",
    "DecisionPlan",
    "SixStyleAugmenter",
    "PipelineStageHandlers",
    "TrainingRecord",
    "run_mock_pipeline",
    "__version__",
]
