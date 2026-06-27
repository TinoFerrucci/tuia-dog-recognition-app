from __future__ import annotations
from email.mime import image

#------------------------------------------------------------------
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
#------------------------------------------------------------------

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
    #                  Etapa 1: funciones a implementar 
    # ----------------                                  ----------------

    def extract_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding baseline (ResNet18 pre-entrenada en ImageNet, sin la
        capa fc -> 512-d) a partir de una imagen BGR de OpenCV.
        Retorna una lista de floats de dimension EMBEDDING_DIM.
        """
        # 1) Modelo baseline cacheado por instancia (lazy: se arma una sola vez).
        model = getattr(self, "_baseline_model", None)
        if model is None:
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = resnet18(weights=weights)
            model.fc = nn.Identity()  # cortamos la clasificacion -> features 512-d
            model.eval()
            self._baseline_model = model

        # 2) Preprocesamiento (DEBE ser identico al indexar y al consultar).
        #   los pesos pre-entrenados esperan RGB normalizado con estas medias/desvíos;
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV entrega BGR
        resized = cv2.resize(
            rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        arr = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std  # normalizacion ImageNet

        # 3) HWC -> CHW -> (1, 3, H, W) y forward sin gradientes.
        #   el modelo espera un batch de imagenes, por eso el unsqueeze(0) para agregar la dimension del batch.
        #   el contiguous() es necesario para evitar errores de memoria al hacer el forward con tensores que no estan almacenados de manera contigua en memoria.
        #   Resultado: un lote de 1 imagen, 3 canales, de 224×224
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
        with torch.no_grad():
            feats = model(tensor)

        embedding = feats.squeeze(0).cpu().numpy().astype(np.float32)
        return embedding.tolist()    

        
    def search_similar_images(self, embedding: list[float], top_k: int) -> list[Neighbor]:
        """
        Recupera de la base vectorial las top_k imagenes mas similares usando
        self.similarity (respeta SIMILARITY_METRIC). 
        Retorna una lista de Neighbor (path, breed, score) ordenada por score descendente.
        """
        neighbors = [
            Neighbor(
                path=record.path,
                breed=record.breed,
                score=self.similarity(embedding, record.embedding),
            )
            for record in self.store.all()
        ]
        neighbors.sort(key=lambda n: n.score, reverse=True)
        return neighbors[:top_k]

    
    def predict_breed_from_neighbors(self, results: list[Neighbor]) -> tuple[str, float]:
        """
        Predice la raza por voto ponderado por score sobre los vecinos.
        Si el mejor score < self.similarity_threshold -> ("unknown", best_score).
        Retorna (raza, score).
        """
        if not results:
            return "unknown", 0.0

        best_score = max(neighbor.score for neighbor in results)
        if best_score < self.similarity_threshold:
            return "unknown", best_score

        votes: dict[str, float] = {}
        for neighbor in results:
            votes[neighbor.breed] = votes.get(neighbor.breed, 0.0) + neighbor.score

        breed = max(votes, key=votes.get)
        score = max(n.score for n in results if n.breed == breed)
        return breed, score

    # ----------------                                  ----------------
    #                  Etapa 1: funciones a implementar 
    # ------------------------------------------------------------------



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
