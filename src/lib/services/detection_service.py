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

# Cache lazy del modelo YOLO (se descarga automaticamente la primera vez).
_YOLO_MODELS: dict[str, object] = {}


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
    ) -> None:
        self.classifier = classifier
        self.yolo_model_name = yolo_model
        self.conf_threshold = conf_threshold
        self.dog_class_id = dog_class_id

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
        pre-entrenado (YOLOv8n via ultralytics). No es necesario entrenar
        el detector.

        Filtra por la clase 'dog' (self.dog_class_id = 16 en COCO) y por
        self.conf_threshold. Funciona con uno, multiples perros y escenas
        complejas con otros objetos.

        Retorna una lista de ((x1, y1, x2, y2), confidence) en pixeles.
        """
        from ultralytics import YOLO

        model = _YOLO_MODELS.get(self.yolo_model_name)
        if model is None:
            model = YOLO(self.yolo_model_name)
            _YOLO_MODELS[self.yolo_model_name] = model
            logger.info("YOLO loaded: %s", self.yolo_model_name)

        results = model.predict(image, conf=self.conf_threshold, verbose=False)
        detections: list[tuple[tuple[int, int, int, int], float]] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            for i, cls_id in enumerate(cls_ids):
                if int(cls_id) != int(self.dog_class_id):
                    continue
                x1, y1, x2, y2 = (int(v) for v in xyxy[i])
                detections.append(((x1, y1, x2, y2), float(confs[i])))
        logger.info("Detected %d dog(s) in image", len(detections))
        return detections

    def classify_detected_dog(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Clasifica la raza del recorte de un perro detectado usando el modelo
        entrenado en la Etapa 2 (self.classifier.load_model()).

        El recorte llega en BGR (OpenCV). Retorna (raza, score).
        """
        import torch
        from torchvision import transforms

        if crop is None or crop.size == 0:
            return "unknown", 0.0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, classes = self.classifier._load_classifier_model()
        model = model.to(device).eval()

        rgb = crop[..., ::-1].copy() if crop.ndim == 3 else crop
        from PIL import Image

        pil = Image.fromarray(rgb.astype(np.uint8))
        tf = transforms.Compose(
            [
                transforms.Resize((self.classifier.image_size, self.classifier.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        tensor = tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        breed = classes[int(idx.cpu().item())] if classes else "unknown"
        return breed, float(conf.cpu().item())

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
