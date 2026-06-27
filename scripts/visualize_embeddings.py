"""Genera visualizaciones PCA y t-SNE de los embeddings del modelo baseline.

Extrae embeddings de una muestra del split test, los reduce a 2D con PCA y t-SNE,
y guarda los graficos en training_results/embeddings/.

Uso (desde la raiz del repo, venv activado, postgres corriendo):
    python scripts/visualize_embeddings.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

os.chdir(SRC if (SRC / ".env").is_file() else ROOT)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_PER_BREED = 5  # muestra para que t-SNE no tarde demasiado


def main() -> None:
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lib.bootstrap import build_similarity, build_store
    from lib.config import settings

    store = build_store(settings)
    similarity = build_similarity(settings, store)

    test_root = settings.dataset_path / "test"
    if not test_root.is_dir():
        sys.exit(f"No existe {test_root}")

    embeddings: list[list[float]] = []
    labels: list[str] = []
    breed_names = sorted(p.name for p in test_root.iterdir() if p.is_dir())

    for breed_dir in sorted(test_root.iterdir()):
        if not breed_dir.is_dir():
            continue
        count = 0
        for img_path in sorted(breed_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if count >= MAX_PER_BREED:
                break
            try:
                img = similarity._load_image(str(img_path.resolve()))
                emb = similarity.extract_embedding(img)
                embeddings.append(emb)
                labels.append(breed_dir.name)
                count += 1
            except Exception as exc:
                print(f"skip {img_path}: {exc}")

    X = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(labels)
    print(f"Embeddings: {X.shape[0]} muestras, {X.shape[1]}-d, {len(set(y))} razas")

    out_dir = settings.dataset_path.parent.parent / "training_results" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    unique = sorted(set(y))
    cmap = plt.get_cmap("tab20", len(unique))
    color_map = {b: cmap(i) for i, b in enumerate(unique)}

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(14, 10))
    for b in unique:
        mask = y == b
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color_map[b]], s=30, alpha=0.7, label=b)
    ax.set_title(f"PCA de embeddings baseline (ResNet50, {X.shape[1]}-d) — {len(unique)} razas")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=5, ncol=4, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_embeddings.png", dpi=150)
    plt.close(fig)
    print(f"PCA guardado: {out_dir / 'pca_embeddings.png'}")
    print(f"  varianza explicada: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

    # t-SNE
    perplexity = min(30, X.shape[0] - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca", max_iter=1000)
    X_tsne = tsne.fit_transform(X)
    fig, ax = plt.subplots(figsize=(14, 10))
    for b in unique:
        mask = y == b
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[color_map[b]], s=30, alpha=0.7, label=b)
    ax.set_title(f"t-SNE de embeddings baseline (ResNet50, {X.shape[1]}-d) — {len(unique)} razas")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=5, ncol=4, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "tsne_embeddings.png", dpi=150)
    plt.close(fig)
    print(f"t-SNE guardado: {out_dir / 'tsne_embeddings.png'}")

    # Guardar resumen
    import json
    summary = {
        "n_samples": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]),
        "n_breeds": len(unique),
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_[:2]],
        "model": settings.embedding_model,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Resumen guardado: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
