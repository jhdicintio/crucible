"""Semantic similarity metric using sentence-transformers."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticSimilarityMetric:
    """Computes mean cosine similarity between prediction and reference embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)

    @property
    def name(self) -> str:
        return "semantic_similarity"

    def compute(
        self,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, float]:
        if not predictions:
            return {"semantic_similarity": 0.0}

        pred_emb = self._model.encode(predictions, convert_to_numpy=True)
        ref_emb = self._model.encode(references, convert_to_numpy=True)

        # Row-wise cosine similarity
        dot = np.sum(pred_emb * ref_emb, axis=1)
        pred_norm = np.linalg.norm(pred_emb, axis=1)
        ref_norm = np.linalg.norm(ref_emb, axis=1)
        similarities = dot / (pred_norm * ref_norm + 1e-8)

        return {"semantic_similarity": round(float(np.mean(similarities)), 6)}
