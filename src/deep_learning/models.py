"""Regression heads for frozen image embeddings."""

import torch
import torch.nn as nn


class EmbeddingFusionRegressor(nn.Module):
    def __init__(
        self,
        image_dimension,
        tabular_dimension,
        dropout,
        image_hidden=128,
        tabular_hidden=64,
        tabular_output=32,
        fusion_hidden=64,
    ):
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Linear(image_dimension, image_hidden),
            nn.ReLU(),
            nn.LayerNorm(image_hidden),
            nn.Dropout(dropout),
        )
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dimension, tabular_hidden),
            nn.ReLU(),
            nn.LayerNorm(tabular_hidden),
            nn.Dropout(dropout),
            nn.Linear(tabular_hidden, tabular_output),
            nn.ReLU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(image_hidden + tabular_output, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, image_embedding, tabular):
        image_features = self.image_branch(image_embedding)
        tabular_features = self.tabular_branch(tabular)
        return self.fusion_head(
            torch.cat([image_features, tabular_features], dim=1)
        ).squeeze(1)


class ImageOnlyRegressor(nn.Module):
    def __init__(self, image_dimension, dropout):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dimension, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, image_embedding):
        return self.model(image_embedding).squeeze(1)
