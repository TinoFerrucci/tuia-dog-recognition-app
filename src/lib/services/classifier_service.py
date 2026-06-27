from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import onnxruntime
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

logger = logging.getLogger(__name__)


class _CustomCNN(nn.Module):
    """CNN propia para clasificacion de razas (Modelo B).

    Arquitectura estilo VGG: 5 bloques con doble conv por bloque,
    canales 3->64->128->256->512->512, AdaptiveAvgPool -> 512 dims.
    La capa de 512 dimensiones equivale a la penultima capa de ResNet18,
    lo que permite usar extract_custom_embedding con ambos modelos.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 64),
            self._block(64, 128),
            self._block(128, 256),
            self._block(256, 512),
            self._block(512, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(512, num_classes)

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)


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
        train_epochs: int = 20,
        train_batch_size: int = 32,
        train_lr: float = 0.001,
        train_lr_step: int = 7,
        train_lr_gamma: float = 0.1,
        train_num_workers: int = 2,
    ) -> None:
        self.checkpoints = checkpoints
        self.image_size = image_size
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.active_model_name = active_model
        self._loaded: dict[str, Any] = {}
        self.train_epochs = train_epochs
        self.train_batch_size = train_batch_size
        self.train_lr = train_lr
        self.train_lr_step = train_lr_step
        self.train_lr_gamma = train_lr_gamma
        self.train_num_workers = train_num_workers
        self.history: dict[str, list[float]] = {}

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
    # Helpers privados
    # ------------------------------------------------------------------

    def _get_transform(self, augment: bool) -> transforms.Compose:
        """Transformaciones de preprocesamiento.

        Train (augment=True): flip horizontal, rotacion, jitter de color y ruido.
        Val/Test (augment=False): solo resize y normalizacion.
        La normalizacion usa los valores de ImageNet porque ResNet18 fue pre-entrenado
        con ellos. Para la CNN propia se mantienen los mismos valores para coherencia.
        """
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        if augment:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                normalize,
            ])
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    def _build_model(self, num_classes: int) -> nn.Module:
        """Construye el modelo segun self.active_model_name."""
        if self.active_model_name == "resnet18_finetuned":
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(512, num_classes)
        else:
            model = _CustomCNN(num_classes)
        return model

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convierte imagen BGR de OpenCV a tensor normalizado [1, 3, H, W]."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self._get_transform(augment=False)(Image.fromarray(rgb))
        return tensor.unsqueeze(0)

    # ------------------------------------------------------------------
    # Etapa 2: funciones a implementar
    # ------------------------------------------------------------------

    def train_classifier(self) -> None:
        """
        Entrena el clasificador de razas sobre el dataset (self.dataset_path).

        Modelo A: fine-tuning de ResNet18 — se reemplaza la capa fc por una
        nueva de 512->num_classes y se entrena todo con Adam + StepLR.
        Modelo B: CNN propia (_CustomCNN) entrenada desde cero.

        Guarda el checkpoint en self.active_checkpoint con el modelo completo,
        los nombres de clases y el historial de entrenamiento.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Entrenando %s en %s", self.active_model_name, device)

        train_dir = self.dataset_path / "train"
        val_dir = self.dataset_path / "valid"

        train_dataset = ImageFolder(train_dir, transform=self._get_transform(augment=True))
        val_dataset = ImageFolder(val_dir, transform=self._get_transform(augment=False))

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.train_num_workers,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.train_num_workers,
        )

        classes = train_dataset.classes
        num_classes = len(classes)
        logger.info("Clases: %d", num_classes)

        model = self._build_model(num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.train_lr_step, gamma=self.train_lr_gamma
        )

        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
        }
        best_val_acc = 0.0

        for epoch in range(self.train_epochs):
            # --- fase de entrenamiento ---
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += images.size(0)

            train_loss = running_loss / total
            train_acc = correct / total

            # --- fase de validacion ---
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item() * images.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    val_total += images.size(0)

            val_loss /= val_total
            val_acc = val_correct / val_total
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            logger.info(
                "Epoch %d/%d — train_loss: %.4f train_acc: %.4f val_loss: %.4f val_acc: %.4f",
                epoch + 1, self.train_epochs, train_loss, train_acc, val_loss, val_acc,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model": model.cpu(), "classes": classes, "history": history},
                    self.active_checkpoint,
                )
                model.to(device)
                logger.info("Checkpoint guardado (val_acc=%.4f)", best_val_acc)

        self.history = history
        self._loaded.pop(self.active_model_name, None)
        logger.info("Entrenamiento finalizado. Mejor val_acc: %.4f", best_val_acc)

    def evaluate_classifier(self) -> dict[str, float]:
        """
        Evalua el modelo activo sobre el conjunto de prueba (test/).

        Carga el checkpoint, corre inferencia sobre todas las imagenes del
        split test y calcula accuracy, precision, recall, specificity y F1
        (promedio macro sobre las 70 clases).
        Retorna un dict con las metricas.
        """
        checkpoint = self.load_model()
        model = checkpoint["model"]
        classes = checkpoint["classes"]
        self.history = checkpoint.get("history", {})

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)

        test_dataset = ImageFolder(
            self.dataset_path / "test",
            transform=self._get_transform(augment=False),
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.train_num_workers,
        )

        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images.to(device))
                all_preds.extend(outputs.argmax(1).cpu().tolist())
                all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        # Especificidad = promedio de TN/(TN+FP) por clase (one-vs-rest)
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
        specificities = []
        for i in range(len(classes)):
            tn = cm.sum() - cm[i].sum() - cm[:, i].sum() + cm[i, i]
            fp = cm[:, i].sum() - cm[i, i]
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        specificity = float(np.mean(specificities))

        metrics = {
            "accuracy": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1": round(f1, 4),
        }
        logger.info("Metricas (%s): %s", self.active_model_name, metrics)
        return metrics

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando el modelo propio activo.

        Para ResNet18 fine-tuned: reemplaza fc con Identity para obtener 512 dims.
        Para CNN custom: usa get_embedding() que devuelve la capa de 512 dims
        antes del clasificador final.
        La imagen llega en BGR (OpenCV). Retorna una lista de 512 floats.
        """
        checkpoint = self.load_model()
        model = checkpoint["model"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval().to(device)

        tensor = self._image_to_tensor(image).to(device)

        with torch.no_grad():
            if isinstance(model, _CustomCNN):
                embedding = model.get_embedding(tensor)
            else:
                # ResNet18: reemplazar fc con Identity para extraer 512 dims
                original_fc = model.fc
                model.fc = nn.Identity()
                embedding = model(tensor)
                model.fc = original_fc

        return embedding.squeeze(0).cpu().tolist()
