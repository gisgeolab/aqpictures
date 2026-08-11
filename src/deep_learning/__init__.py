"""Utilities for the pretrained-CNN PM2.5 extension."""

from .embeddings import extract_embeddings, load_or_extract_embeddings
from .training import (
    DEFAULT_TRAINING_CONFIG,
    build_outer_splits,
    fit_and_evaluate_fusion_holdout,
    train_fusion_cross_validation,
    train_image_only_cross_validation,
)

__all__ = [
    "DEFAULT_TRAINING_CONFIG",
    "build_outer_splits",
    "extract_embeddings",
    "fit_and_evaluate_fusion_holdout",
    "load_or_extract_embeddings",
    "train_fusion_cross_validation",
    "train_image_only_cross_validation",
]
