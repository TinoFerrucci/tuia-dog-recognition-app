"""Evalua la busqueda por similitud (Etapa 1) con NDCG@10 sobre un conjunto de prueba.

Para cada imagen del conjunto de prueba (por defecto el split `test/` del dataset, o
un conjunto independiente en `data/external/`):
  1. Extrae el embedding (modelo activo segun EMBEDDING_MODEL).
  2. Recupera Top-K vecinos desde la base vectorial.
  3. Calcula NDCG@10 con relevancia binaria (1.0 si el vecino es de la misma raza
     que la consulta, 0.0 si no).
Reporta el NDCG@10 medio.

Requiere:
  - Etapa 1 implementada (extract_embedding / extract_custom_embedding).
  - Base vectorial indexada (`python scripts/build_index.py --split train`).

Uso:
    python scripts/evaluate_search.py [--split test] [--limit 0] [--top-k 10]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

os.chdir(SRC if (SRC / ".env").is_file() else ROOT)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="test",
        help="Subcarpeta del dataset a usar como conjunto de prueba (test/valid/external).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max imagenes por raza (0 = todas).")
    parser.add_argument("--top-k", type=int, default=0, help="Top K (0 = usar TOP_K del .env).")
    args = parser.parse_args()

    from lib.bootstrap import build_classifier, build_similarity, build_store
    from lib.config import settings
    from lib.evaluation.metrics import ndcg_at_k

    store = build_store(settings)
    similarity = build_similarity(settings, store)

    if settings.embedding_model == "baseline":
        extractor = similarity.extract_embedding
    else:
        classifier = build_classifier(settings)
        classifier.set_active_model(settings.embedding_model)
        extractor = classifier.extract_custom_embedding
    similarity.extract_embedding = extractor  # type: ignore[method-assign]

    top_k = args.top_k or settings.top_k

    root = settings.dataset_path / args.split
    if not root.is_dir():
        # conjunto independiente en data/external
        alt = settings.data_path / args.split
        if alt.is_dir():
            root = alt
        else:
            sys.exit(f"No existe {root} ni {alt}. Descarga el dataset o crea el conjunto de prueba.")

    ndcgs: list[float] = []
    total = 0
    for breed_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        count = 0
        for image_path in sorted(breed_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if args.limit and count >= args.limit:
                break
            try:
                image = similarity._load_image(str(image_path))
                emb = extractor(image)
            except Exception as exc:
                print(f"skip {image_path}: {exc}")
                continue
            neighbors = similarity.search_similar_images(emb, top_k)
            relevances = [1.0 if n.breed == breed_dir.name else 0.0 for n in neighbors]
            score = ndcg_at_k(relevances, k=top_k)
            ndcgs.append(score)
            count += 1
            total += 1
        print(f"{breed_dir.name}: {count} imagenes evaluadas")

    if not ndcgs:
        sys.exit("No se evaluo ninguna imagen.")

    mean_ndcg = sum(ndcgs) / len(ndcgs)
    print(f"\nTotal: {total} imagenes evaluadas (modelo: {settings.embedding_model}, top_k={top_k})")
    print(f"NDCG@{top_k} medio: {mean_ndcg:.4f}")


if __name__ == "__main__":
    main()
