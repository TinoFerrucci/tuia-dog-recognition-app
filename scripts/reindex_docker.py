"""Reindexa el dataset guardando paths Linux relativos (arregla Top K en docker).

Se ejecuta desde el HOST con el venv activado (GPU) contra el postgres del host.
Los paths se guardan en formato Linux (data/dataset/train/...) para que el
url_resolver del contenedor docker los resuelva correctamente.

No usa SimilarityService.index_image (que guarda el mismo path para abrir y
persistir); en su lugar extrae el embedding y crea el EmbeddingRecord con el
path relativo Linux, abriendo la imagen con su path absoluto Windows.

Uso (desde la raiz del repo, venv activado, postgres corriendo):
    python scripts/reindex_docker.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

os.chdir(SRC if (SRC / ".env").is_file() else ROOT)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    from lib.bootstrap import build_similarity, build_store
    from lib.config import settings
    from lib.schemas import EmbeddingRecord

    store = build_store(settings)
    similarity = build_similarity(settings, store)
    extractor = similarity.extract_embedding

    root = settings.dataset_path / "train"
    if not root.is_dir():
        sys.exit(f"No existe {root}. Descarga el dataset con scripts/download_dataset.py")

    data_root = settings.data_path.resolve()
    repo_root = data_root.parent  # padre de data/ (repo root en host, /app en docker)

    total = 0
    for breed_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        count = 0
        for image_path in sorted(breed_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            abs_path = image_path.resolve()
            # Path Linux relativo al repo root: data/dataset/train/Breed/xxx.jpg
            # (en docker cwd=/app -> /app/data/dataset/... -> url_resolver lo mapea a /files/data/...)
            rel_linux = abs_path.relative_to(repo_root).as_posix()

            image = similarity._load_image(str(abs_path))
            embedding = extractor(image)
            record = EmbeddingRecord(
                id_imagen=str(uuid4()),
                embedding=embedding,
                path=rel_linux,
                breed=breed_dir.name,
                metadata={"split": "train", "model": settings.embedding_model},
            )
            store.append(record)

            count += 1
            total += 1
            if total % 500 == 0:
                print(f"  ...{total} embeddings indexados", flush=True)
        print(f"{breed_dir.name}: {count} indexadas", flush=True)

    print(f"\nTotal: {total} embeddings almacenados (modelo: {settings.embedding_model})")
    print("Paths guardados en formato Linux relativo (data/dataset/train/...)")


if __name__ == "__main__":
    main()

