"""Smoke test del pipeline completo de Etapa 3 (usa el predict() provisto)."""
from __future__ import annotations

import json
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
        raise SystemExit("Uso: python scripts/smoke_pipeline.py foto.jpg")
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

    out_dir = Path(os.environ.get("OUTPUT_PATH", "output"))
    result_file = service.predict(img_path, out_dir)
    data = json.loads(Path(result_file).read_text())

    dets = data["detections"]
    print(f"Perros detectados: {len(dets)}")
    image = cv2.imread(img_path)
    for i, d in enumerate(dets, start=1):
        x1, y1, x2, y2 = d["bbox"]
        print(f"  #{i}: {d['breed']}  breed_score={d['breed_score']:.3f}  "
              f"det_score={d['det_score']:.3f}  bbox=({x1},{y1},{x2},{y2})")
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{d['breed']} {d['breed_score']:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_img = out_dir / "smoke_pipeline_out.jpg"
    cv2.imwrite(str(out_img), image)
    print(f"Imagen anotada: {out_img.resolve()}")
    print(f"JSON: {result_file}")


if __name__ == "__main__":
    main()
