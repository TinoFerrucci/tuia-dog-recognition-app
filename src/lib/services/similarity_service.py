from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from lib.schemas import EmbeddingRecord, Neighbor, SearchResult
from lib.storage.base import EmbeddingStoreProtocol

logger = logging.getLogger(__name__)


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
        Genera el embedding de una imagen usando ResNet18 pre-entrenado en
        ImageNet sin la capa de clasificacion final.
        Retorna una lista de 512 floats.
        """
        if not hasattr(self, "_model"):
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Identity()   #Remueve la capa de clasificacion final
            model.eval()  # Modo inferencia (desactiva dropout, batchnorm, etc)
            self._model = model
            self._transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    # Valores con los que fue entrenado ResNet18 en ImageNet
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self._transform(Image.fromarray(rgb)).unsqueeze(0)

        with torch.no_grad():
            embedding = self._model(tensor)

        return embedding.squeeze(0).tolist()

    def search_similar_images(self, embedding: list[float], top_k: int) -> list[Neighbor]:
        """
        Recupera de la base vectorial las top_k imagenes mas similares.
        Con pgvector usa search() para busqueda eficiente; con JSON itera
        todos los registros. El score se calcula siempre con self.similarity()
        para respetar SIMILARITY_METRIC (cosine | l2).
        Retorna una lista de Neighbor ordenada por score descendente.
        """
        if hasattr(self.store, "search"):
            records = self.store.search(embedding, top_k)
        else:
            all_records = self.store.all()
            records = sorted(
                all_records,
                key=lambda r: self.similarity(embedding, r.embedding),
                reverse=True,
            )[:top_k]

        neighbors = [
            Neighbor(path=r.path, breed=r.breed, score=self.similarity(embedding, r.embedding))
            for r in records
        ]
        return sorted(neighbors, key=lambda n: n.score, reverse=True)

    def predict_breed_from_neighbors(self, results: list[Neighbor]) -> tuple[str, float]:
        """
        Predice la raza usando votacion ponderada por score: cada vecino vota
        por su raza con peso igual a su similitud. Si el mejor score esta por
        debajo de self.similarity_threshold, retorna "unknown". Retorna (raza, score).
        """
        if not results:
            return "unknown", 0.0

        if results[0].score < self.similarity_threshold:
            return "unknown", results[0].score

        breed_scores: dict[str, float] = {}
        for neighbor in results:
            breed_scores[neighbor.breed] = breed_scores.get(neighbor.breed, 0.0) + neighbor.score

        predicted_breed = max(breed_scores, key=lambda b: breed_scores[b])
        confidence = breed_scores[predicted_breed] / sum(breed_scores.values())

        return predicted_breed, confidence

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
