from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnxruntime

logger = logging.getLogger(__name__)

# Imagenet normalization (RGB)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Funciones a implementar por el estudiante:
      - train_classifier()
      - evaluate_classifier()
      - extract_custom_embedding(image)

    La carga de checkpoints (.pth / .onnx) y la seleccion del modelo activo
    ya estan provistas.
    """

    def __init__(
        self,
        checkpoints: dict[str, Path],
        image_size: int,
        dataset_path: Path,
        output_path: Path,
        active_model: str = "resnet18_finetuned",
    ) -> None:
        # checkpoints: nombre logico -> ruta del archivo (ej. resnet18_finetuned -> models/resnet18_finetuned.pth)
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Infraestructura provista
    # ------------------------------------------------------------------

    def set_active_model(self, name: str) -> None:
        """Define que checkpoint usan extract_custom_embedding y la clasificacion.

        Valores esperados: resnet18_finetuned | cnn_custom.
        """
        if name not in self.checkpoints:
            raise ValueError(f"Unknown model '{name}'. Expected one of: {sorted(self.checkpoints)}")
        self.active_model_name = name

    @property
    def active_checkpoint(self) -> Path:
        return self.checkpoints[self.active_model_name]

    def load_model(self, name: str | None = None) -> Any:
        """Carga (con cache) el checkpoint del modelo indicado o del activo.

        Soporta modelos PyTorch (.pth) y exportados a ONNX (.onnx).
        """
        key = name or self.active_model_name
        if key in self._loaded:
            return self._loaded[key]
        path = self.checkpoints[key]
        if not path.exists():
            raise ValueError(
                f"Checkpoint not found: {path}. Entrena el modelo (Etapa 2) y guardalo en esa ruta."
            )
        suf = path.suffix.lower()
        if suf == ".pth":
            model = torch.load(path, map_location="cpu", weights_only=False)
        elif suf == ".onnx":
            model = onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"Unsupported model format (expected .pth or .onnx): {path}")
        self._loaded[key] = model
        return model

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        """
        Entrena el clasificador de razas sobre el dataset (self.dataset_path).

        Modelo A (obligatorio): fine-tuning de ResNet18 pre-entrenado.
        Modelo B (obligatorio): CNN propia.

        Debe:
          - Usar los splits train/valid definidos en la notebook.
          - Aplicar el preprocesamiento y data augmentation justificados.
          - Guardar el checkpoint resultante en self.active_checkpoint
            (ej: models/resnet18_finetuned.pth).
        """
        import torch.nn as nn
        from torchvision import datasets, models, transforms

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Training %s on %s", self.active_model_name, device)

        train_dir = self.dataset_path / "train"
        valid_dir = self.dataset_path / "valid"
        if not train_dir.is_dir():
            raise FileNotFoundError(f"Train split not found: {train_dir}")

        size = self.image_size
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
        valid_tf = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )

        train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
        classes = train_ds.classes
        num_classes = len(classes)
        valid_ds = (
            datasets.ImageFolder(str(valid_dir), transform=valid_tf) if valid_dir.is_dir() else None
        )

        batch_size = 32
        num_workers = 2 if device.type == "cuda" else 0
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        valid_loader = (
            torch.utils.data.DataLoader(
                valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            if valid_ds is not None
            else None
        )

        if self.active_model_name == "cnn_custom":
            model = _build_cnn_custom(num_classes, size)
        else:
            model = self._build_resnet18(num_classes)

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

        epochs = 25
        patience = 5
        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            train_correct = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                train_correct += (out.argmax(1) == labels).sum().item()
            train_loss = running_loss / len(train_ds)
            train_acc = train_correct / len(train_ds)

            val_loss = train_loss
            val_acc = 0.0
            if valid_loader is not None:
                model.eval()
                vloss = 0.0
                vcorrect = 0
                with torch.no_grad():
                    for imgs, labels in valid_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        out = model(imgs)
                        vloss += criterion(out, labels).item() * imgs.size(0)
                        vcorrect += (out.argmax(1) == labels).sum().item()
                val_loss = vloss / len(valid_ds)
                val_acc = vcorrect / len(valid_ds)
                scheduler.step(val_loss)

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                }
            )
            logger.info(
                "epoch %d/%d  train_loss=%.4f val_loss=%.4f  train_acc=%.4f val_acc=%.4f",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                train_acc,
                val_acc,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch + 1, patience)
                    break

        if best_state is None:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        arch = "cnn_custom" if self.active_model_name == "cnn_custom" else "resnet18"
        embedding_dim = 512
        if isinstance(model, _ResNet18Wrapper):
            sd = {k: v.cpu().clone() for k, v in model.backbone.state_dict().items()}
        else:
            sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        checkpoint = {
            "state_dict": sd,
            "classes": classes,
            "arch": arch,
            "embedding_dim": embedding_dim,
            "num_classes": num_classes,
            "history": history,
        }
        self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, self.active_checkpoint)
        logger.info("Checkpoint saved: %s (%d classes)", self.active_checkpoint, num_classes)

        self._save_training_results(history, classes, arch)

    def evaluate_classifier(self) -> dict[str, float]:
        """
        Evalua el modelo activo sobre el conjunto de prueba.

        Debe reportar: accuracy, precision, recall (sensibilidad),
        specificity (especificidad) y F1-Score. La matriz de confusion y las
        curvas de entrenamiento se documentan en la notebook.

        Retorna un dict con las metricas, ej:
          {"accuracy": 0.91, "precision": 0.90, "recall": 0.89,
           "specificity": 0.99, "f1": 0.90}
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )
        from torchvision import datasets, transforms

        test_dir = self.dataset_path / "test"
        if not test_dir.is_dir():
            raise FileNotFoundError(f"Test split not found: {test_dir}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, classes = self._load_classifier_model()
        model = model.to(device).eval()
        num_classes = len(classes)

        tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
        test_ds = datasets.ImageFolder(str(test_dir), transform=tf)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                preds = out.argmax(dim=1).cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(labels.tolist())

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        fp = cm.sum(axis=0) - np.diag(cm)
        tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
        specificity_per_class = np.where((tn + fp) > 0, tn / (tn + fp), 0.0)
        specificity = float(np.mean(specificity_per_class))

        metrics = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": specificity,
            "f1": float(f1),
        }
        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando el modelo propio activo
        (penultima capa del ResNet18 fine-tuned o de la CNN custom).

        Se usa cuando EMBEDDING_MODEL != baseline para que la busqueda por
        similitud (Etapa 1) funcione con los modelos entrenados.
        La imagen llega en BGR (OpenCV). Retorna una lista de floats de
        dimension EMBEDDING_DIM.
        """
        import torch
        from torchvision import transforms

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = self._load_classifier_model()
        model = model.to(device).eval()
        feature_extractor = model.feature_extractor().to(device).eval()

        rgb = _bgr_to_rgb_pil(image)
        tf = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
        tensor = tf(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = feature_extractor(tensor).squeeze(0).cpu().numpy().astype(np.float32)
        return feat.tolist()

    # ------------------------------------------------------------------
    # Helpers aditivos (no modifican infraestructura provista)
    # ------------------------------------------------------------------

    def _build_resnet18(self, num_classes: int):
        """Construye ResNet18 pre-entrenado con cabeza reemplazada por num_classes."""
        from torchvision import models

        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)
        return _ResNet18Wrapper(backbone)

    def _save_training_results(self, history: list, classes: list[str], arch: str) -> None:
        """Guarda curvas, metricas y graficos en training_results/<modelo>/.

        Genera:
          - history.json  (curvas train/val loss y accuracy por epoch)
          - metrics.json  (accuracy, precision, recall, specificity, F1 sobre test)
          - training_curves.png  (loss + accuracy, train vs val)
          - confusion_matrix.png (sobre test)
        Requiere matplotlib (incluido en requirements.txt).
        """
        import json as _json

        results_dir = self.dataset_path.parent.parent / "training_results" / self.active_model_name
        results_dir.mkdir(parents=True, exist_ok=True)

        (results_dir / "history.json").write_text(
            _json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        try:
            metrics = self.evaluate_classifier()
            (results_dir / "metrics.json").write_text(
                _json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except FileNotFoundError:
            logger.warning("No test split found; metrics.json not generated")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ep = [h["epoch"] for h in history]
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
            axes[0].plot(ep, [h["val_loss"] for h in history], label="val")
            axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
            axes[0].set_title(f"{self.active_model_name} - Loss"); axes[0].legend(); axes[0].grid(True)
            axes[1].plot(ep, [h.get("train_acc", 0) for h in history], label="train")
            axes[1].plot(ep, [h.get("val_acc", 0) for h in history], label="val")
            axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
            axes[1].set_title(f"{self.active_model_name} - Accuracy"); axes[1].legend(); axes[1].grid(True)
            fig.tight_layout()
            fig.savefig(results_dir / "training_curves.png", dpi=120)
            plt.close(fig)

            test_dir = self.dataset_path / "test"
            if test_dir.is_dir():
                from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
                from torchvision import datasets, transforms

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model, cls = self._load_classifier_model()
                model = model.to(device).eval()
                tf = transforms.Compose(
                    [
                        transforms.Resize((self.image_size, self.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
                    ]
                )
                test_ds = datasets.ImageFolder(str(test_dir), transform=tf)
                loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)
                y_true, y_pred = [], []
                with torch.no_grad():
                    for imgs, labels in loader:
                        out = model(imgs.to(device))
                        y_pred.extend(out.argmax(1).cpu().tolist())
                        y_true.extend(labels.tolist())
                cm = confusion_matrix(y_true, y_pred, labels=list(range(len(cls))))
                fig, ax = plt.subplots(figsize=(16, 14))
                ConfusionMatrixDisplay(cm, display_labels=cls).plot(
                    xticks_rotation=90, ax=ax, colorbar=False, cmap="Blues"
                )
                ax.set_title(f"{self.active_model_name} - Confusion Matrix (test)")
                fig.tight_layout()
                fig.savefig(results_dir / "confusion_matrix.png", dpi=120)
                plt.close(fig)
        except Exception as exc:
            logger.warning("Could not generate plots: %s", exc)

        logger.info("Training results saved to %s", results_dir)

    def _load_classifier_model(self) -> tuple[Any, list[str]]:
        """Carga el checkpoint activo y reconstruye el nn.Module listo para usar.

        Soporta dos formatos de checkpoint:
          - dict con 'state_dict', 'classes', 'arch' (formato Etapa 2)
          - nn.Module crudo (fallback de checkpoints viejos)
        """
        ckpt = self.load_model()
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            classes = ckpt.get("classes", [])
            arch = ckpt.get("arch", "resnet18")
            num_classes = ckpt.get("num_classes", len(classes))
            if arch == "cnn_custom":
                model = _build_cnn_custom(num_classes, self.image_size)
            else:
                model = self._build_resnet18(num_classes)
            state = ckpt["state_dict"]
            if isinstance(model, _ResNet18Wrapper):
                model.backbone.load_state_dict(state)
            else:
                model.load_state_dict(state)
            model.classes = classes
            return model, classes
        # Fallback: ya es un nn.Module
        return ckpt, getattr(ckpt, "classes", [])


# ----------------------------------------------------------------------
# Modulos de modelo aditivos (no modifican infraestructura provista)
# ----------------------------------------------------------------------


class _ResNet18Wrapper(torch.nn.Module):
    """Envuelve ResNet18 para exponer feature_extractor() (todo menos fc).

    Permite que extract_custom_embedding reutilice la penultima capa de forma
    uniforme entre ResNet18 fine-tuned y la CNN custom.
    """

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = 512

    def feature_extractor(self) -> torch.nn.Module:
        class _Feat(torch.nn.Module):
            def __init__(self, bb: torch.nn.Module) -> None:
                super().__init__()
                self.bb = bb

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                bb = self.bb
                x = bb.conv1(x)
                x = bb.bn1(x)
                x = bb.relu(x)
                x = bb.maxpool(x)
                x = bb.layer1(x)
                x = bb.layer2(x)
                x = bb.layer3(x)
                x = bb.layer4(x)
                x = bb.avgpool(x)
                return torch.flatten(x, 1)

        return _Feat(self.backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class _CNNCustom(torch.nn.Module):
    """CNN propia: bloques conv + GAP + FC. Embedding de 512-d."""

    def __init__(self, num_classes: int, image_size: int = 224) -> None:
        super().__init__()
        self.image_size = image_size
        self.embedding_dim = 512
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Linear(512, num_classes)

    def feature_extractor(self) -> torch.nn.Module:
        return self.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        return self.classifier(feat)


def _build_cnn_custom(num_classes: int, image_size: int = 224) -> _CNNCustom:
    return _CNNCustom(num_classes=num_classes, image_size=image_size)


def _bgr_to_rgb_pil(image: np.ndarray):
    """Convierte BGR (OpenCV uint8) a PIL.Image RGB."""
    from PIL import Image

    rgb = image[..., ::-1].copy() if image.ndim == 3 else image
    return Image.fromarray(rgb.astype(np.uint8))
