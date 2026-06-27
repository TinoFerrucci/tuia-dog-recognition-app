from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnxruntime

import cv2

# Normalizacion ImageNet: igual que el baseline de Etapa 1 y lo que esperan los pesos pre-entrenados.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

logger = logging.getLogger(__name__)


from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Classifier(nn.Module):
    """ResNet18 pre-entrenada adaptada a num_classes, con acceso al embedding 512-d.

    forward(x) -> logits [B, num_classes]   (entrenamiento / evaluacion)
    embed(x)   -> features [B, 512]         (extract_custom_embedding)
    """

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features   # 512
        backbone.fc = nn.Identity()             # el backbone ahora produce features 512-d
        self.backbone = backbone
        self.head = nn.Linear(in_features, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)                 # [B, 512] (features crudos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(x))         # [B, num_classes]

# ------------------------------------------------------------------
#         Nuestro Modelo propio (CNN entrenada desde cero): 
# --------                                                 ---------
class CustomCNNClassifier(nn.Module):
    """CNN propia entrenada desde cero, con la misma interfaz que ResNet18Classifier.

    forward(x) -> logits [B, num_classes]   (entrenamiento / evaluacion)
    embed(x)   -> features [B, 512]         (extract_custom_embedding / Etapa 1)

    El 'backbone' (5 bloques conv + global average pooling) termina en 512 features,
    igual que ResNet18 -> intercambiable y compatible con pgvector (EMBEDDING_DIM=512).
    Usa los mismos nombres backbone/head que ResNet18Classifier para integrarse con _fit.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            # Convolucion (detecta patrones) + BatchNorm (estabiliza el entreno desde
            # cero) + ReLU (no linealidad) + MaxPool (achica el tamano a la mitad).
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        # backbone: imagen 3x224x224 -> 512 numeros.
        self.backbone = nn.Sequential(
            block(3, 32),              # 224 -> 112
            block(32, 64),             # 112 -> 56
            block(64, 128),            # 56  -> 28
            block(128, 256),           # 28  -> 14
            block(256, 512),           # 14  -> 7
            nn.AdaptiveAvgPool2d(1),   # promedia cada mapa 7x7 -> 512x1x1
            nn.Flatten(),              # -> [B, 512]  (este es el embedding)
        )
        # head: regularizacion + capa lineal a las 70 razas.
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)              # [B, 512]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))   # [B, num_classes]

# --------                                                 ---------
#         Nuestro Modelo propio (CNN entrenada desde cero): 
# ------------------------------------------------------------------

class AlbumentationsImageFolder(ImageFolder):
    """ImageFolder que lee con OpenCV (BGR) y aplica una transform de albumentations.

    Reutiliza el descubrimiento de clases de ImageFolder (classes, class_to_idx,
    samples) pero respeta la convencion BGR del proyecto y la misma conversion
    BGR->RGB que usa extract_embedding / extract_custom_embedding.
    """

    def __init__(self, root: str, transform: A.Compose) -> None:
        super().__init__(str(root))      # descubre clases y arma self.samples
        self.alb_transform = transform   # guardada aparte (no es una transform torchvision)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = cv2.imread(path)                        # BGR uint8 (convencion del proyecto)
        if image is None:
            raise ValueError(f"No se pudo leer la imagen: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # -> RGB uint8 (lo que esperan las transforms)
        tensor = self.alb_transform(image=image)["image"]
        return tensor, target

class ClassifierService:
    """Etapa 2: entrenamiento y comparacion de modelos de clasificacion.

    Métodos a implementar por el estudiante:
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


    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _run_epoch(self, model, loader, criterion, optimizer=None) -> tuple[float, float]:
        """Corre una epoca. Si optimizer is None -> modo evaluacion (sin gradientes)."""
        train_mode = optimizer is not None
        model.train(train_mode)
        device = self.device

        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train_mode):
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                if train_mode:
                    optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, targets)
                if train_mode:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * images.size(0)
                correct += (logits.argmax(1) == targets).sum().item()
                total += images.size(0)
        return total_loss / total, correct / total

    def _fit(self, model, train_loader, valid_loader, classes,
             epochs: int = 20, lr_backbone: float = 1e-4, lr_head: float = 1e-3,
             warmup_epochs: int = 2, weight_decay: float = 1e-4) -> dict:
        """Entrena `model` y guarda el mejor (por val accuracy) en self.active_checkpoint.

        Hiperparametros (a documentar en el informe): epochs, lr_backbone/lr_head
        (LR diferencial), warmup_epochs, weight_decay, optimizer AdamW, scheduler cosine.
        Estrategia: warmup con backbone congelado y luego fine-tuning completo.
        Retorna el historial para graficar las curvas.
        """
        device = self.device
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": lr_backbone},
                {"params": model.head.parameters(),     "lr": lr_head},
            ],
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            # Warmup: primeras epocas con backbone congelado (solo se entrena la cabeza).
            freeze_backbone = epoch <= warmup_epochs
            for p in model.backbone.parameters():
                p.requires_grad = not freeze_backbone

            train_loss, train_acc = self._run_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self._run_epoch(model, valid_loader, criterion, optimizer=None)
            scheduler.step()

            for k, v in zip(history, (train_loss, train_acc, val_loss, val_acc)):
                history[k].append(v)
            logger.info("Epoch %02d/%d | train acc=%.4f loss=%.4f | val acc=%.4f loss=%.4f%s",
                        epoch, epochs, train_acc, train_loss, val_acc, val_loss,
                        " [warmup]" if freeze_backbone else "")

            # Model selection: guardamos el mejor por val accuracy.
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.classes = classes  # el mapping idx->raza viaja con el pickle
                self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model, self.active_checkpoint)
                logger.info("  -> nuevo mejor (val_acc=%.4f) guardado en %s",
                            best_val_acc, self.active_checkpoint)

        return history

    # --------------------------------------------------------------------------------
    # Métodos de similitud (reutilizados de Etapa 1)
    #    Se establecen dos pipelines distintas:
    #    Train → con data augmentation (para regularizar y que el modelo no memorice).
    #    Valid / test → determinística: solo resize + normalización, sin augmentation.
    # --------------------------------------------------------------------------------

    def _build_train_transform(self) -> A.Compose:
        """Pipeline de ENTRENAMIENTO: resize + data augmentation + normalizacion.

        Espera una imagen RGB uint8 (HWC) y devuelve un tensor CHW float32 normalizado.
        La conversion BGR->RGB es responsabilidad de quien llama (el Dataset).
        """
        import albumentations as A # libreria de data augmentation y preprocesamiento de imagenes, compatible con OpenCV y PyTorch
        from albumentations.pytorch import ToTensorV2 # convierte imagenes a tensores PyTorch, normalizando y reordenando canales (HWC -> CHW)
        
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.HorizontalFlip(p=0.5),                                  # flip horizontal (no vertical)
            A.Rotate(limit=15, p=0.5),                               # rotacion leve +/-15 grados
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.5),   # brillo/contraste
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=5),
            ], p=0.3),                                               # blur (uno u otro)
            A.GaussNoise(std_range=(0.05, 0.15), p=0.3),            # ruido moderado (API nueva)
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def _build_eval_transform(self) -> A.Compose:
        """
        Pipeline para validacion / test / inferencia (sin augmentation).
        Debe ser identica a la que usara extract_custom_embedding, para evitar
        train/serve skew y mantener consistencia con la busqueda de Etapa 1.
        """
        import albumentations as A # libreria de data augmentation y preprocesamiento de imagenes, compatible con OpenCV y PyTorch
        from albumentations.pytorch import ToTensorV2 # convierte imagenes a tensores PyTorch, normalizando y reordenando canales (HWC -> CHW)
        
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    def _preprocess_bgr(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocesamiento manual de una imagen BGR -> tensor (1,3,H,W) normalizado.
        Equivalente a _build_eval_transform pero sin albumentations (inferencia liviana)
        y consistente con extract_embedding de Etapa 1.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_LINEAR)   # misma interpolacion que A.Resize
        arr = resized.astype(np.float32) / 255.0
        arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()

    # --------------------------------------------------------------------------------
    # Método constructor de loaders
    #   Lectura con cv2 (BGR→RGB): 
    #       Mantiene la convención de todo el proyecto y es idéntica a la que va a usar extract_custom_embedding
    #   Mapping de clases consistente: 
    #       ImageFolder ordena alfabéticamente. Los tres splits tienen las mismas carpetas
    #       idx-->raza coincide en train/valid/test. 
    # --------------------------------------------------------------------------------

    def _build_dataloaders(
        self, batch_size: int = 32, num_workers: int = 2
    ) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
        """Arma los DataLoaders de train/valid/test desde el split oficial del dataset.

        train -> transform con augmentation; valid/test -> transform determinista.
        Retorna (train_loader, valid_loader, test_loader, classes).
        """
        train_ds = AlbumentationsImageFolder(self.dataset_path / "train", self._build_train_transform())
        valid_ds = AlbumentationsImageFolder(self.dataset_path / "valid", self._build_eval_transform())
        test_ds  = AlbumentationsImageFolder(self.dataset_path / "test",  self._build_eval_transform())

        # ImageFolder ordena las clases alfabeticamente y los tres splits comparten
        # los mismos nombres de carpeta -> el mapping idx->raza es identico en los tres.
        classes = train_ds.classes
        logger.info("Dataset: %d clases | train=%d valid=%d test=%d",
                    len(classes), len(train_ds), len(valid_ds), len(test_ds))

        common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        train_loader = DataLoader(train_ds, shuffle=True,  **common)
        valid_loader = DataLoader(valid_ds, shuffle=False, **common)
        test_loader  = DataLoader(test_ds,  shuffle=False, **common)
        return train_loader, valid_loader, test_loader, classes

    # --------------------------------------------------------------------------------
    # Método dispatcher para construir el modelo activo (ResNet18 fine-tuned o CNN custom).
    # --------------------------------------------------------------------------------

    def _build_model(self, num_classes: int) -> nn.Module:
        """Construye el modelo segun el modelo activo."""
        if self.active_model_name == "resnet18_finetuned":
            return ResNet18Classifier(num_classes=num_classes, pretrained=True)
        if self.active_model_name == "cnn_custom":
            return CustomCNNClassifier(num_classes=num_classes)
        raise ValueError(f"Modelo desconocido: {self.active_model_name}")

    # ------------------------------------------------------------------
    #                   Etapa 2: Métodos a implementar
    # -----------------                                  ---------------

    def train_classifier(self) -> dict:
        """
        Entrena el modelo activo (resnet18_finetuned o cnn_custom) y guarda el mejor.

        Orquesta las herramientas ya definidas:
          _build_dataloaders -> _build_model -> _fit.
        Elige los hiperparametros segun el modelo (ver abajo) y se los pasa a _fit,
        que no necesita saber que modelo es.
        El mejor modelo (por val accuracy) lo guarda _fit en self.active_checkpoint.
        Deja el historial en self.history para graficar las curvas en la notebook.
        """
        train_loader, valid_loader, _test_loader, classes = self._build_dataloaders()
        model = self._build_model(num_classes=len(classes))

        logger.info("Entrenando '%s' | %d clases | device=%s",
                    self.active_model_name, len(classes), self.device)

        # Hiperparametros segun el tipo de modelo:
        #  - resnet18_finetuned: viene pre-entrenado -> LR diferencial (cuerpo lento
        #    para no arruinar ImageNet, cabeza rapida) + warmup con el cuerpo congelado.
        #  - cnn_custom: se entrena desde cero -> un unico LR normal para toda la red,
        #    sin congelar (no hay pesos previos que preservar) y mas epocas.
        if self.active_model_name == "cnn_custom":
            fit_kwargs = dict(epochs=30, lr_backbone=1e-3, lr_head=1e-3,
                              warmup_epochs=0, weight_decay=1e-4)
        else:
            fit_kwargs = dict(epochs=20, lr_backbone=1e-4, lr_head=1e-3,
                              warmup_epochs=2, weight_decay=1e-4)

        self.history = self._fit(model, train_loader, valid_loader, classes, **fit_kwargs)
        return self.history

    def evaluate_classifier(self) -> dict[str, float]:
        """Evalua el modelo activo sobre el test set.

        Carga el mejor checkpoint (load_model), corre inferencia sobre test y calcula
        accuracy, precision, recall, specificity y F1 (macro). Guarda y_true/y_pred/cm
        en self.last_eval para graficar la matriz de confusion en la notebook.
        """
        # Lazy-import: sklearn solo se necesita al evaluar (Colab), no para que la app importe el modulo.
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
        )

        device = self.device
        model = self.load_model()          # mejor checkpoint del modelo activo (objeto completo)
        model.to(device).eval()

        _train, _valid, test_loader, classes = self._build_dataloaders()

        y_true, y_pred = [], []
        with torch.no_grad():
            for images, targets in test_loader:
                logits = model(images.to(device))
                y_pred.extend(logits.argmax(1).cpu().tolist())
                y_true.extend(targets.tolist())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        n = len(classes)

        # Specificity no viene en sklearn para multiclase -> se calcula desde la matriz de confusion.
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
        TP = np.diag(cm).astype(float)
        FP = cm.sum(axis=0) - TP
        FN = cm.sum(axis=1) - TP
        TN = cm.sum() - (TP + FP + FN)
        with np.errstate(divide="ignore", invalid="ignore"):
            spec_per_class = np.where((TN + FP) > 0, TN / (TN + FP), np.nan)
        specificity = float(np.nanmean(spec_per_class))

        metrics = {
            "accuracy":    float(accuracy_score(y_true, y_pred)),
            "precision":   float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall":      float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "specificity": specificity,
            "f1":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

        self.last_eval = {"y_true": y_true, "y_pred": y_pred, "classes": classes, "cm": cm}
        logger.info("Eval '%s' | %s", self.active_model_name,
                    {k: round(v, 4) for k, v in metrics.items()})
        return metrics


    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
        """
        Embedding 512-d con el modelo propio activo (penultima capa via .embed()).
        Imagen en BGR (OpenCV). Se usa cuando EMBEDDING_MODEL != baseline para que la
        busqueda por similitud (Etapa 1) funcione con los modelos entrenados.
        """
        model = self.load_model()          # modelo activo: resnet18_finetuned | cnn_custom
        device = self.device
        model.to(device).eval()

        tensor = self._preprocess_bgr(image).to(device)
        with torch.no_grad():
            feats = model.embed(tensor)    # [1, 512]
        return feats.squeeze(0).cpu().numpy().astype(np.float32).tolist()

    # -----------------                                  ---------------
    #                   Etapa 2: Métodos a implementar
    # ------------------------------------------------------------------
