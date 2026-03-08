"""
Shared package for feature engineering used by both modeling and Streamlit frontend.
Keep this file side-effect free (no file I/O, no heavy imports).
"""

from .feature_engineering import (
    FEATURES,
    PRIMARY_PREDICTORS,
    RISK_TYPE_MAP,
    add_tenure_quarters,
    build_feature_set,
)

__all__ = [
    "FEATURES",
    "PRIMARY_PREDICTORS",
    "RISK_TYPE_MAP",
    "add_tenure_quarters",
    "build_feature_set",
]
