from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from lib.schemas import ClassifyResult, DetectResult, DogDetection
from lib.services.classifier_service import ClassifierService

logger = logging.getLogger(__name__)


class DetectionService:
    """Etapa 3: pipeline de deteccion y clasificacion.

    Funciones a implementar por el estudiante:
      - detect_dogs(image)
      - classify_detected_dog(crop)

    La orquestacion (predict: deteccion -> recorte -> clasificacion -> JSON)
    ya esta provista.
    """

    def __init__(
        self,
        classifier: ClassifierService,
        yolo_model: str,
        conf_threshold: float,
        dog_class_id: int,
        imgsz: int = 1280,
    ) -> None:
        self.classifier = classifier
        self.yolo_model_name = yolo_model
        self.conf_threshold = conf_threshold
        self.dog_class_id = dog_class_id
        self.imgsz = imgsz

    @staticmethod
    def _clip_xyxy(
        x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))
        if x2 <= x1:
            x2 = min(x1 + 1, width)
        if y2 <= y1:
            y2 = min(y1 + 1, height)
        return x1, y1, x2, y2

    def _load_image(self, source_path: str) -> np.ndarray:
        image = cv2.imread(str(source_path))
        if image is None:
            raise ValueError(f"Could not read image: {source_path}")
        # BGR uint8 (convencion OpenCV / ultralytics)
        return image

    # ------------------------------------------------------------------
    # Etapa 3: funciones a implementar
    # ------------------------------------------------------------------

    def detect_dogs(self, image: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        """
        Detecta todos los perros presentes en la imagen usando un modelo YOLO
        pre-entrenado (YOLOv8n via ultralytics). No es necesario entrenar el detector.
        Retorna una lista de ((x1, y1, x2, y2), confidence) en pixeles.
        """
        # Carga perezosa: se instancia el modelo una sola vez y se cachea en la
        # instancia, para no releer los pesos en cada imagen.
        if getattr(self, "_yolo", None) is None:
            from ultralytics import YOLO

            self._yolo = YOLO(self.yolo_model_name)

        # ultralytics acepta directamente el array BGR de OpenCV.
        # Dejamos que YOLO filtre por clase 'dog' y por umbral de confianza.
        results = self._yolo.predict(
            source=image,
            conf=self.conf_threshold,
            classes=[self.dog_class_id],
            imgsz=self.imgsz,
            verbose=False,
        )

        detections: list[tuple[tuple[int, int, int, int], float]] = []
        for result in results:
            for xyxy, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
                x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
                detections.append(((x1, y1, x2, y2), float(conf)))

        logger.debug("detect_dogs: %d perro(s) detectado(s)", len(detections))
        return detections



    def classify_detected_dog(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Clasifica la raza del recorte de un perro detectado usando el modelo
        entrenado en la Etapa 2 (self.classifier.load_model()).

        El recorte llega en BGR (OpenCV). Retorna (raza, score).
        """
        import torch

        clf = self.classifier
        model = clf.load_model()              # objeto completo (.pth) con .classes adjunto
        device = clf.device
        model.to(device).eval()

        # Mismo preprocesamiento que Etapa 1/2 (BGR->RGB, resize, normalizacion ImageNet):
        # lo reusamos para no introducir train/serve skew.
        tensor = clf._preprocess_bgr(crop).to(device)
        with torch.no_grad():
            logits = model(tensor)            # [1, num_classes]
            probs = torch.softmax(logits, dim=1)
            score, idx = probs.max(dim=1)

        breed = model.classes[int(idx)]       # idx -> raza (mapping guardado con el modelo)
        return breed, float(score)

    # ------------------------------------------------------------------
    # Orquestacion provista
    # ------------------------------------------------------------------

    def classify_image(
        self, source_path: str, output_path: Path, model_name: str | None = None
    ) -> str:
        """Clasifica la imagen completa con el modelo entrenado (pestaña Etapa 2).

        Reutiliza classify_detected_dog tratando la imagen entera como recorte,
        por lo que requiere la Etapa 2 (modelo entrenado) y classify_detected_dog.
        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        if model_name:
            self.classifier.set_active_model(model_name)
        breed, score = self.classify_detected_dog(image)
        payload = ClassifyResult(
            source_path=source_path,
            model=model_name or self.classifier.active_model_name,
            breed=breed,
            score=round(float(score), 4),
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)

    def predict(self, source_path: str, output_path: Path) -> str:
        """Flujo completo: deteccion -> bounding boxes -> recortes -> clasificacion.

        Escribe el resultado como JSON en `output_path` y retorna su ruta.
        """
        image = self._load_image(source_path)
        height, width = image.shape[:2]

        detections: list[DogDetection] = []
        for (box, det_score) in self.detect_dogs(image):
            x1, y1, x2, y2 = self._clip_xyxy(*[int(v) for v in box], height, width)
            crop = image[y1:y2, x1:x2]
            breed, breed_score = self.classify_detected_dog(crop)
            detections.append(
                DogDetection(
                    bbox=[x1, y1, x2, y2],
                    det_score=round(float(det_score), 4),
                    breed=breed,
                    breed_score=round(float(breed_score), 4),
                )
            )

        detected_breeds = sorted({item.breed for item in detections if item.breed != "unknown"})
        payload = DetectResult(
            source_path=source_path,
            detections=detections,
            detected_breeds=detected_breeds,
        )
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"result-{uuid4()}.json"
        result_file.write_text(
            json.dumps(payload.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        return str(result_file)
