from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as T
import onnxruntime
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class CNNCustom(nn.Module):
    def __init__(self, num_classes: int = 70) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

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

    def train_classifier(self) -> dict:
        """
        Entrena el clasificador de razas sobre el dataset (self.dataset_path).

        Modelo A (obligatorio): fine-tuning de ResNet18 pre-entrenado.
        Modelo B (opcional, recomendado): CNN propia.

        Debe:
          - Usar los splits train/valid definidos en la notebook.
          - Aplicar el preprocesamiento y data augmentation justificados.
          - Guardar el checkpoint resultante en self.active_checkpoint
            (ej: models/resnet18_finetuned.pth).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        NUM_CLASES = 70

        # Preprocesamiento
        tfm_train = T.Compose([
            T.Resize(256),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tfm_val = T.Compose([
            T.Resize(256),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_ds = datasets.ImageFolder(self.dataset_path / "train", transform=tfm_train)
        val_ds = datasets.ImageFolder(self.dataset_path / "valid", transform=tfm_val)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

        # Seleccion del modelo segun active_model_name
        if self.active_model_name == "cnn_custom":
            modelo = CNNCustom(num_classes=NUM_CLASES).to(device)
            optimizer = optim.Adam(modelo.parameters(), lr=1e-3)
            EPOCHS = 20
        else:
            # Modelo A: ResNet18 fine-tuned (solo head)
            modelo = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            for p in modelo.parameters():
                p.requires_grad = False
            modelo.fc = nn.Linear(modelo.fc.in_features, NUM_CLASES)
            modelo = modelo.to(device)
            optimizer = optim.Adam(modelo.fc.parameters(), lr=1e-3)
            EPOCHS = 10
        criterio = nn.CrossEntropyLoss()
        historia = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

        # Lo entrena
        for ep in range(1, EPOCHS + 1):
            modelo.train()
            ok, total, loss_sum = 0, 0, 0.0
            for imgs, y in train_loader:
                imgs, y = imgs.to(device), y.to(device)
                optimizer.zero_grad()
                out = modelo(imgs)
                loss = criterio(out, y)
                loss.backward()
                optimizer.step()
                ok += (out.argmax(1) == y).sum().item()
                total += y.size(0)
                loss_sum += loss.item() * y.size(0)

            modelo.eval()
            ok_v, total_v, loss_v_sum = 0, 0, 0.0
            with torch.no_grad():
                for imgs, y in val_loader:
                    imgs, y = imgs.to(device), y.to(device)
                    out_v = modelo(imgs)
                    loss_v = criterio(out_v, y)
                    ok_v += (out_v.argmax(1) == y).sum().item()
                    total_v += y.size(0)
                    loss_v_sum += loss_v.item() * y.size(0)

            historia["train_acc"].append(ok / total)
            historia["val_acc"].append(ok_v / total_v)
            historia["train_loss"].append(loss_sum / total)
            historia["val_loss"].append(loss_v_sum / total_v)

            logger.info(
                "Época %d/%d | train acc %.3f | val acc %.3f",
                ep, EPOCHS, ok/total, ok_v/total_v
            )

        # Guarda el checkpoint
        self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(modelo, self.active_checkpoint)
        logger.info("Checkpoint guardado en %s", self.active_checkpoint)
        self._loaded[self.active_model_name] = modelo

        return historia

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tfm_test = T.Compose([
            T.Resize(256),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_ds = datasets.ImageFolder(self.dataset_path / "test", transform=tfm_test)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

        modelo = self.load_model()
        modelo.eval()
        modelo.to(device)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, y in test_loader:
                imgs = imgs.to(device)
                preds = modelo(imgs).argmax(1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(y.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        cm = confusion_matrix(all_labels, all_preds)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)
        specificity = float(np.mean(tn / (tn + fp + 1e-8)))

        return {
            "accuracy": float((all_preds == all_labels).mean()),
            "precision": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
            "specificity": specificity,
            "f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        }

    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Genera el embedding de una imagen usando el modelo propio activo
        (penultima capa del ResNet18 fine-tuned o de la CNN custom).

        Se usa cuando EMBEDDING_MODEL != baseline para que la busqueda por
        similitud (Etapa 1) funcione con los modelos entrenados.
        La imagen llega en BGR (OpenCV). Retorna una lista de floats de
        dimension EMBEDDING_DIM.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tfm = T.Compose([
            T.Resize(256),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        modelo = self.load_model()
        modelo.eval()
        modelo.to(device)
        # extrae penúltima capa
        extractor = nn.Sequential(*list(modelo.children())[:-1])
        extractor.eval().to(device)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        tensor = tfm(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = extractor(tensor)
        return embedding.squeeze().cpu().numpy().tolist()
