"""Smoke test de la Etapa 3: ejercita DetectionService.detect_dogs sobre una imagen."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2

# Ubica el paquete `lib` tanto en Docker (/app/lib) como local (src/lib).
_here = Path(__file__).resolve()
for _candidate in (_here.parents[1] / "src", _here.parents[1]):
    if (_candidate / "lib").is_dir():
        sys.path.insert(0, str(_candidate))
        break

from lib.services.detection_service import DetectionService


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Pasá la ruta de una imagen: python scripts/smoke_detect.py foto.jpg")

    img_path = sys.argv[1]
    yolo_model = os.environ.get("YOLO_MODEL", "models/yolov8n.pt")
    conf = float(os.environ.get("YOLO_CONF_THRESHOLD", "0.25"))
    dog_id = int(os.environ.get("YOLO_DOG_CLASS_ID", "16"))

    # detect_dogs no toca el classifier; para esta prueba lo dejamos en None.
    service = DetectionService(
        classifier=None,
        yolo_model=yolo_model,
        conf_threshold=conf,
        dog_class_id=dog_id,
    )

    image = cv2.imread(img_path)
    if image is None:
        raise SystemExit(f"No pude leer la imagen: {img_path}")

    dets = service.detect_dogs(image)
    print(f"Perros detectados: {len(dets)}")
    for i, (box, score) in enumerate(dets, start=1):
        print(f"  #{i}: bbox={box}  conf={score:.3f}")
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{score:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_dir = Path(os.environ.get("OUTPUT_PATH", "output"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "smoke_detect_out.jpg"
    cv2.imwrite(str(out), image)
    print(f"Imagen anotada guardada en: {out.resolve()}")


if __name__ == "__main__":
    main()
