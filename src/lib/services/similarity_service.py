from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import cv2
import numpy as np

from lib.schemas import EmbeddingRecord, Neighbor, SearchResult
from lib.storage.base import EmbeddingStoreProtocol

logger = logging.getLogger(__name__)

# Cache lazy del modelo baseline (ResNet50 sin FC). Se instancia una sola vez
# por proceso y se comparte entre todas las llamadas a extract_embedding.
_BASELINE_MODEL = None
_BASELINE_DEVICE = None


class SimilarityService:
    """Etapa 1: buscador de imagenes por similitud.

    Funciones a implementar por el estudiante:
      - extract_embedding(image)
      - search_similar_images(embedding, top_k)
      - predict_breed_from_neighbors(results)

    La orquestacion (search, index_image, persistencia y metricas de similitud)
    ya esta provista y no debe modificarse sin justificarlo en el informe.
    """

    def __init__(
        self,
        store: EmbeddingStoreProtocol,
        similarity_metric: str,
        similarity_threshold: float,
        top_k: int,
        image_size: int,
        model_name: str,
        url_resolver: Optional[Callable[[Path], Optional[str]]] = None,
    ) -> None:
        self.store = store
        self.similarity_metric = similarity_metric
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.image_size = image_size
        self.model_name = model_name
        self.url_resolver = url_resolver

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV)
        return image

    # ------------------------------------------------------------------
    # Etapa 1: funciones a implementar
    # ------------------------------------------------------------------

    def extract_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando un modelo pre-entrenado en
        ImageNet (ResNet50) sin la capa de clasificacion final.

        Preprocesa la imagen (BGR OpenCV -> RGB, resize a self.image_size,
        normalizacion ImageNet) y devuelve el vector de la capa de pooling
        final (2048-d para ResNet50).
        """
        global _BASELINE_MODEL, _BASELINE_DEVICE
        if _BASELINE_MODEL is None:
            import torch
            from torchvision import models

            _BASELINE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            backbone = models.resnet50(weights=weights)
            backbone.fc = torch.nn.Identity()
            backbone.eval().to(_BASELINE_DEVICE)
            _BASELINE_MODEL = (backbone, weights.transforms())
            logger.info("Baseline ResNet50 loaded on %s (2048-d)", _BASELINE_DEVICE)

        import torch

        backbone, preprocess = _BASELINE_MODEL
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_like = _cv2_to_pil(rgb)
        tensor = preprocess(pil_like).unsqueeze(0).to(_BASELINE_DEVICE)
        with torch.no_grad():
            embedding = backbone(tensor).squeeze(0).cpu().numpy().astype(np.float32)
        return embedding.tolist()

    def search_similar_images(self, embedding: list[float], top_k: int) -> list[Neighbor]:
        """
        Recupera de la base vectorial las top_k imagenes mas similares.

        Con pgvector usa self.store.search (distancia coseno <=>) cuando la
        metrica es 'cosine'; en cualquier otro caso (JSON store o metrica l2)
        itera self.store.all() y calcula self.similarity() por pares.
        Retorna una lista de Neighbor (path, breed, score) ordenada por score
        descendente.
        """
        metric = self.similarity_metric.lower()
        store_search = getattr(self.store, "search", None)

        if store_search is not None and metric == "cosine":
            # pgvector requiere un numpy.ndarray (el dumper de register_vector lo
            # convierte a tipo `vector`); una list[float] se envia como
            # `double precision[]` y el operador `<=>` no resuelve (no hay cast
            # implicito para operadores, solo para INSERT).
            query_arr = np.asarray(embedding, dtype=np.float32)
            records = store_search(query_arr, top_k)
            neighbors: list[Neighbor] = []
            for rec in records:
                score = self._cosine(np.asarray(embedding, dtype=np.float32), np.asarray(rec.embedding, dtype=np.float32))
                neighbors.append(Neighbor(path=rec.path, breed=rec.breed, score=float(score)))
            neighbors.sort(key=lambda n: n.score, reverse=True)
            return neighbors

        records = self.store.all()
        scored: list[tuple[float, EmbeddingRecord]] = []
        for rec in records:
            score = self.similarity(embedding, rec.embedding)
            scored.append((score, rec))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [Neighbor(path=rec.path, breed=rec.breed, score=float(score)) for score, rec in scored[:top_k]]

    def predict_breed_from_neighbors(self, results: list[Neighbor]) -> tuple[str, float]:
        """
        Predice la raza a partir de los vecinos recuperados mediante voto
        mayoritario ponderado por score. Si no hay vecinos o el mejor score
        esta por debajo de self.similarity_threshold se considera "unknown".
        Retorna (raza, score).
        """
        if not results:
            return "unknown", 0.0
        best_score = max(n.score for n in results)
        if best_score < self.similarity_threshold:
            return "unknown", float(best_score)
        weights: dict[str, float] = {}
        for n in results:
            weights[n.breed] = weights.get(n.breed, 0.0) + float(n.score)
        breed = max(weights, key=weights.get)
        return breed, float(best_score)

    # ------------------------------------------------------------------
    # Helpers de similitud provistos
    # ------------------------------------------------------------------

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _l2_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dist = float(np.linalg.norm(a - b))
        return 1.0 / (1.0 + dist)

    def similarity(self, query: list[float], ref: list[float]) -> float:
        a = np.asarray(query, dtype=np.float32)
        b = np.asarray(ref, dtype=np.float32)
        if self.similarity_metric.lower() == "l2":
            return self._l2_similarity(a, b)
        return self._cosine(a, b)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def index_image(
        self, image_path: str, breed: str, metadata: dict[str, object] | None = None
    ) -> EmbeddingRecord:
        """Extrae el embedding de una imagen del dataset y lo persiste en la base vectorial."""
        image = self._load_image(image_path)
        embedding = self.extract_embedding(image)
        record = EmbeddingRecord(
            id_imagen=str(uuid4()),
            embedding=embedding,
            path=str(image_path),
            breed=breed,
            metadata=metadata or {},
        )
        self.store.append(record)
        return record

    def _with_url(self, neighbor: Neighbor) -> Neighbor:
        if self.url_resolver is not None and not neighbor.url:
            neighbor.url = self.url_resolver(Path(neighbor.path))
        return neighbor

    def search(
        self,
        source_path: str,
        output_path: Path,
        embedding_fn: Optional[Callable[[np.ndarray], list[float]]] = None,
        model_name: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """Pipeline completo de la Etapa 1: embedding -> vecinos -> raza predicha.

        `embedding_fn` permite seleccionar dinamicamente el extractor
        (baseline, resnet18_finetuned o cnn_custom, ver Etapa 2).
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        extractor = embedding_fn or self.extract_embedding
        embedding = extractor(image)

        k = int(top_k) if top_k else self.top_k
        neighbors = [self._with_url(n) for n in self.search_similar_images(embedding, k)]
        breed, score = self.predict_breed_from_neighbors(neighbors)
        logger.info("Predicted breed: %s (score=%.4f) for %s", breed, score, source_path)

        payload = SearchResult(
            source_path=source_path,
            model=model_name or self.model_name,
            predicted_breed=breed,
            score=round(float(score), 4),
            neighbors=neighbors,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)


def _cv2_to_pil(rgb: np.ndarray):
    """Convierte una imagen RGB (numpy uint8 HxWxC) a PIL.Image."""
    from PIL import Image

    return Image.fromarray(rgb.astype(np.uint8))
