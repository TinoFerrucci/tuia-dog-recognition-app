from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

import cv2
import numpy as np

import torch
import torchvision.models as models
import torchvision.transforms as transforms

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

        if self.model_name == "baseline":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.model.fc = torch.nn.Identity()  # Reemplazamos la última capa lineal de clasificación (.fc) por Identity
            self.model.to(self.device) 
            self.model.eval()

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
        ImageNet (ej: ResNet18, EfficientNet, ConvNeXt) sin la capa de
        clasificacion final.

        Sugerencias:
          - Preprocesar la imagen (resize a self.image_size, normalizacion ImageNet).
          - Usar torchvision.models o timm con pesos pre-entrenados.
          - Recordar que la imagen llega en BGR (OpenCV).
        Retorna una lista de floats de dimension EMBEDDING_DIM.
        """
        transform_etapa1 = transforms.Compose([
            transforms.ToPILImage(), # Agregamos esto solo para que acepte el array de OpenCV
            # Redimensionar la imagen en 224 x 224
            transforms.Resize((224, 224)),                         
            # Conversión a Tensor (convierte los píxeles de 0-255 a un rango de 0-1)
            transforms.ToTensor(),                                 
            # Normalización (media y desviación estándar)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                # Convertimos de BGR a RGB 
        tensor_image = transform_etapa1(image_rgb)                        # Aplicamos las transformaciones
        tensor_image = tensor_image.unsqueeze(0).to(self.device)          # agrega una dimensión extra de tamaño 1 en la posición 0

        # Extracción sin instanciar el modelo cada vez
        with torch.no_grad():
            output = self.model(tensor_image)
            embedding_np = output.squeeze().cpu().numpy()
            embedding_list = embedding_np.tolist()

        return embedding_list

    def search_similar_images(self, embedding: list[float], top_k: int) -> list[Neighbor]:
        """
        Recupera de la base vectorial las top_k imagenes mas similares.

        Sugerencias:
          - Con pgvector: self.store.search(embedding, top_k).
          - Con JSON: iterar self.store.all() y usar self.similarity(...).
          - Respetar SIMILARITY_METRIC (cosine | l2).
        Retorna una lista de Neighbor (path, breed, score) ordenada por score
        descendente.
        """
        # Si la base soporta búsqueda nativa (pgvector), usamos su método optimizado
        if hasattr(self.store, "search"): # O evaluar cómo se expone el store
            try:
                return self.store.search(embedding, top_k)
            except NotImplementedError:
                pass

        # Fallback para el JSON plano: iteramos sobre todos los registros y calculamos similitud manualmente
        raw_results = []
        for record in self.store.all():
            score = self.similarity(embedding, record.embedding)
            neighbor_record = Neighbor(
                path=str(record.path),
                breed=str(record.breed),
                score=float(score)
            )
            raw_results.append(neighbor_record)
            
        raw_results.sort(key=lambda x: x.score, reverse=True)
        return raw_results[:top_k]

    def predict_breed_from_neighbors(self, results: list[Neighbor]) -> tuple[str, float]:
        """
        Predice la raza a partir de los vecinos recuperados (ej: voto
        mayoritario, opcionalmente ponderado por score).

        Si el mejor score esta por debajo de self.similarity_threshold se
        considera "unknown". Retorna (raza, score).
        """
        if not results:
            return "unknown", 0.0
            
        # Como results ya viene ordenado de forma descendente, el primer índice tiene el mayor score
        mejor_score = results[0].score
        
        if mejor_score < self.similarity_threshold:
            return "unknown", 0.0

        # Implementación del algoritmo k-NN con Voto Ponderado
        votos_por_raza = {}
        
        for vecino in results:
            raza = vecino.breed
            score = vecino.score  # Notación de objeto para clases
            
            # Acumulamos la similitud como el "peso" del voto
            if raza in votos_por_raza:
                votos_por_raza[raza] += score
            else:
                votos_por_raza[raza] = score
                
        # Extraemos la clave con el valor de ponderación más alto
        raza_ganadora = max(votos_por_raza, key=votos_por_raza.get)
        
        return raza_ganadora, mejor_score

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
