import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader

sys.path.insert(0, "src")
from lib.services.classifier_service import ClassifierService  # noqa: E402

svc = ClassifierService(
    checkpoints={
        "resnet18_finetuned": Path("models/resnet18_finetuned.pth"),
        "cnn_custom": Path("models/cnn_custom.pth"),
    },
    image_size=224,
    dataset_path=Path("data/dataset"),
    output_path=Path("output"),
)

# 1) transforms: dummy RGB uint8 -> tensor CHW float32
dummy = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
t = svc._build_eval_transform()(image=dummy)["image"]
assert t.shape == (3, 224, 224) and t.dtype == torch.float32, t.shape
print("[ok] transforms ->", tuple(t.shape), t.dtype)

# 2) dataloaders + descubrimiento de clases
train_loader, valid_loader, test_loader, classes = svc._build_dataloaders(batch_size=8, num_workers=0)
assert len(classes) == 70, len(classes)
images, targets = next(iter(train_loader))
assert images.shape[1:] == (3, 224, 224), images.shape
print(f"[ok] dataset -> {len(classes)} clases | batch {tuple(images.shape)}")

# 3) modelo: forward (logits) y embed (512-d)
model = svc._build_model(num_classes=len(classes))
with torch.no_grad():
    logits = model(images)
    feats = model.embed(images)
assert logits.shape == (images.size(0), 70), logits.shape
assert feats.shape == (images.size(0), 512), feats.shape
print(f"[ok] modelo -> logits {tuple(logits.shape)} | embed {tuple(feats.shape)}")

# 4) un paso de entrenamiento sobre un subset chico (CPU)
small = DataLoader(Subset(train_loader.dataset, list(range(min(16, len(train_loader.dataset))))),
                   batch_size=8)
loss, acc = svc._run_epoch(model, small, torch.nn.CrossEntropyLoss(),
                           optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3))
print(f"[ok] _run_epoch -> loss={loss:.4f} acc={acc:.4f}")


# 5) preprocesamiento manual == eval transform (cero skew)
import cv2
bgr = (np.random.rand(257, 389, 3) * 255).astype(np.uint8)
manual = svc._preprocess_bgr(bgr).squeeze(0)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
alb = svc._build_eval_transform()(image=rgb)["image"]
maxdiff = (manual - alb).abs().max().item()
assert torch.allclose(manual, alb, atol=1e-5), maxdiff
print(f"[ok] manual == eval transform | maxdiff={maxdiff:.2e}")

# 6) path de embedding (sin checkpoint): embed sobre el tensor preprocesado
with torch.no_grad():
    emb = model.embed(svc._preprocess_bgr(bgr))
assert emb.shape == (1, 512), emb.shape
print(f"[ok] embed(preprocess) -> {tuple(emb.shape)}")

print("\nSMOKE TEST OK ✅")
