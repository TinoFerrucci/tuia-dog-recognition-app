"""Tests basicos para las funciones implementadas (Etapa 1 y 3) y metricas provistas.

No requieren dataset ni modelos entrenados: prueban logica pura.
Se ejecutan desde la raiz del repo con:  pytest tests/
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.environ.setdefault("USE_PGVECTOR", "false")


def test_predict_breed_from_neighbors_majority_vote():
    from lib.schemas import Neighbor
    from lib.services.similarity_service import SimilarityService

    svc = SimilarityService(
        store=_FakeStore([]),
        similarity_metric="cosine",
        similarity_threshold=0.3,
        top_k=3,
        image_size=224,
        model_name="baseline",
    )
    results = [
        Neighbor(path="a", breed="labrador", score=0.9),
        Neighbor(path="b", breed="labrador", score=0.8),
        Neighbor(path="c", breed="poodle", score=0.95),
    ]
    breed, score = svc.predict_breed_from_neighbors(results)
    assert breed == "labrador"
    assert score == pytest.approx(0.95)


def test_predict_breed_unknown_when_below_threshold():
    from lib.schemas import Neighbor
    from lib.services.similarity_service import SimilarityService

    svc = SimilarityService(
        store=_FakeStore([]),
        similarity_metric="cosine",
        similarity_threshold=0.9,
        top_k=3,
        image_size=224,
        model_name="baseline",
    )
    results = [Neighbor(path="a", breed="labrador", score=0.5)]
    breed, score = svc.predict_breed_from_neighbors(results)
    assert breed == "unknown"
    assert score == pytest.approx(0.5)


def test_predict_breed_empty_results():
    from lib.services.similarity_service import SimilarityService

    svc = SimilarityService(
        store=_FakeStore([]),
        similarity_metric="cosine",
        similarity_threshold=0.3,
        top_k=3,
        image_size=224,
        model_name="baseline",
    )
    breed, score = svc.predict_breed_from_neighbors([])
    assert breed == "unknown"
    assert score == 0.0


def test_search_similar_images_json_store_orders_by_score():
    from lib.schemas import EmbeddingRecord, Neighbor
    from lib.services.similarity_service import SimilarityService

    records = [
        EmbeddingRecord(id_imagen="1", embedding=[1.0, 0.0], path="a", breed="labrador", metadata={}),
        EmbeddingRecord(id_imagen="2", embedding=[0.0, 1.0], path="b", breed="poodle", metadata={}),
    ]
    store = _FakeStore(records)
    svc = SimilarityService(
        store=store,
        similarity_metric="cosine",
        similarity_threshold=0.3,
        top_k=2,
        image_size=224,
        model_name="baseline",
    )
    neighbors = svc.search_similar_images([1.0, 0.0], 2)
    assert len(neighbors) == 2
    assert neighbors[0].breed == "labrador"
    assert neighbors[0].score > neighbors[1].score


def test_similarity_cosine_and_l2():
    from lib.services.similarity_service import SimilarityService

    svc = SimilarityService(
        store=_FakeStore([]),
        similarity_metric="cosine",
        similarity_threshold=0.3,
        top_k=3,
        image_size=224,
        model_name="baseline",
    )
    assert svc.similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert svc.similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    svc_l2 = SimilarityService(
        store=_FakeStore([]),
        similarity_metric="l2",
        similarity_threshold=0.3,
        top_k=3,
        image_size=224,
        model_name="baseline",
    )
    assert svc_l2.similarity([0.0, 0.0], [0.0, 0.0]) == pytest.approx(1.0)


def test_clip_xyxy():
    _require_torch()
    from lib.services.detection_service import DetectionService

    assert DetectionService._clip_xyxy(10, 10, 20, 20, 100, 100) == (10, 10, 20, 20)
    assert DetectionService._clip_xyxy(-5, -5, 200, 200, 100, 100) == (0, 0, 100, 100)
    x1, y1, x2, y2 = DetectionService._clip_xyxy(50, 50, 50, 50, 100, 100)
    assert x2 > x1 and y2 > y1


def test_ndcg_at_k():
    from lib.evaluation.metrics import ndcg_at_k

    # ranking perfecto: todos relevantes primero
    assert ndcg_at_k([1.0, 1.0, 0.0], k=3) == pytest.approx(1.0, abs=1e-6)
    # ranking peor: relevante al final
    score = ndcg_at_k([0.0, 0.0, 1.0], k=3)
    assert 0.0 < score < 1.0


def test_precision_recall_f1_and_specificity():
    from lib.evaluation.metrics import precision_recall_f1, specificity

    p, r, f1 = precision_recall_f1(tp=8, fp=2, fn=2)
    assert p == pytest.approx(0.8)
    assert r == pytest.approx(0.8)
    assert f1 == pytest.approx(0.8)
    assert specificity(tn=90, fp=2) == pytest.approx(90 / 92)


# --- fixtures / helpers ---


class _FakeStore:
    def __init__(self, records):
        self._records = records

    def all(self):
        return self._records

    def append(self, record):
        self._records.append(record)


def _require_torch() -> None:
    """Salta tests que importan modulos que dependen de torch/onnxruntime."""
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch no disponible en este entorno")
