"""
Evalua la Etapa 1: NDCG@10 de la busqueda por similitud sobre un split independiente.

Requiere haber indexado 'train' con scripts/build_index.py. 
Usa 'test' como conjunto de consultas (independiente de train). 
Carga el indice una sola vez para evitar releer la base en cada consulta.

Uso:
    python scripts/evaluate_etapa1.py [--split test] [--k 10] [--limit 0]
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
    parser.add_argument("--split", default="test")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Maximo de consultas por raza (0 = todas).")
    args = parser.parse_args()

    from lib.bootstrap import build_similarity, build_store
    from lib.config import settings
    from lib.evaluation.metrics import ndcg_at_k

    store = build_store(settings)
    similarity = build_similarity(settings, store)

    root = settings.dataset_path / args.split
    if not root.is_dir():
        sys.exit(f"No existe {root}.")

    # Indice cargado UNA sola vez (no por consulta).
    records = store.all()
    print(f"Indice cargado: {len(records)} embeddings")
    ref_embeddings = [r.embedding for r in records]
    ref_breeds = [r.breed for r in records]

    ndcgs: list[float] = []
    per_breed: dict[str, list[float]] = {}

    breed_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for idx, breed_dir in enumerate(breed_dirs, 1):
        true_breed = breed_dir.name
        count = 0
        for image_path in sorted(breed_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if args.limit and count >= args.limit:
                break
            image = similarity._load_image(str(image_path))
            embedding = similarity.extract_embedding(image)
            # Mismo ranking que search_similar_images, pero sobre el indice ya cargado.
            scored = sorted(
                (
                    (similarity.similarity(embedding, ref), breed)
                    for ref, breed in zip(ref_embeddings, ref_breeds)
                ),
                key=lambda t: t[0],
                reverse=True,
            )[: args.k]
            relevances = [1.0 if breed == true_breed else 0.0 for _, breed in scored]
            ndcgs.append(ndcg_at_k(relevances, args.k))
            per_breed.setdefault(true_breed, []).append(ndcgs[-1])
            count += 1
        running = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        print(f"[{idx}/{len(breed_dirs)}] {true_breed}: {count} consultas (NDCG@{args.k} parcial={running:.4f})")

    mean_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    breed_means = {b: sum(v) / len(v) for b, v in per_breed.items()}
    ordered = sorted(breed_means.items(), key=lambda kv: kv[1])

    print(f"\nConsultas evaluadas: {len(ndcgs)} ({len(per_breed)} razas)")
    print(f"NDCG@{args.k} promedio: {mean_ndcg:.4f}\n")
    print("Peores 5 razas:")
    for b, m in ordered[:5]:
        print(f"  {b}: {m:.4f}")
    print("Mejores 5 razas:")
    for b, m in ordered[-5:]:
        print(f"  {b}: {m:.4f}")


if __name__ == "__main__":
    main()