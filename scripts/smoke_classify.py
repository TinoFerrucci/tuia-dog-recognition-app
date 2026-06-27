"""Valida classify_detected_dog sobre la imagen completa (camino de la pestaña Etapa 2)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

_here = Path(__file__).resolve()
for _candidate in (_here.parents[1] / "src", _here.parents[1]):
    if (_candidate / "lib").is_dir():
        sys.path.insert(0, str(_candidate))
        break

from lib.services.classifier_service import ClassifierService
from lib.services.detection_service import DetectionService


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Uso: python scripts/smoke_classify.py foto.jpg")
    img_path = sys.argv[1]

    model_path = Path(os.environ.get("MODEL_PATH", "models"))
    classifier = ClassifierService(
        checkpoints={
            "resnet18_finetuned": model_path / "resnet18_finetuned.pth",
            "cnn_custom": model_path / "cnn_custom.pth",
        },
        image_size=int(os.environ.get("IMAGE_SIZE", 224)),
        dataset_path=Path(os.environ.get("DATASET_PATH", "data/dataset")),
        output_path=Path(os.environ.get("OUTPUT_PATH", "output")),
        active_model="resnet18_finetuned",
    )
    service = DetectionService(
        classifier=classifier,
        yolo_model=os.environ.get("YOLO_MODEL", "models/yolov8n.pt"),
        conf_threshold=float(os.environ.get("YOLO_CONF_THRESHOLD", 0.25)),
        dog_class_id=int(os.environ.get("YOLO_DOG_CLASS_ID", 16)),
        imgsz=int(os.environ.get("YOLO_IMGSZ", 1280)),
    )

    image = cv2.imread(img_path)
    if image is None:
        raise SystemExit(f"No pude leer: {img_path}")
    breed, score = service.classify_detected_dog(image)
    print(f"Raza: {breed}  score={score:.4f}")


if __name__ == "__main__":
    main()
