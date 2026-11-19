"""AI Dashboard — churn-model monitoring, live inference, animated charts."""

from dashboard.model import MLP
from dashboard.pipeline import build

__all__ = ["MLP", "build"]
__version__ = "1.0.0"
