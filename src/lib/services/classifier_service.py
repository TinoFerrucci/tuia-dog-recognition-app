from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnxruntime
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

logger = logging.getLogger(__name__)


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
            #--------------------------------------------------------------------------------------------------------------------------------------------------------
            # model = torch.load(path, map_location="cpu", weights_only=False)         # Se reemplaza torch.load directo porque el modelo fue -
            # https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html   # guardado como state_dict y requiere reconstrucción explícita de la arquitectura
            
            model = models.resnet18(weights=None)                                      # Crea arquitectura vacía del modelo usado en entrenamiento
            model.fc = nn.Linear(model.fc.in_features, 70)                             # Ajusta la capa final a las 70 clases del dataset
            state_dict = torch.load(path, map_location="cpu")                          # Carga los pesos entrenados (state_dict) desde el checkpoint
            model.load_state_dict(state_dict)                                          # Asigna los pesos al modelo reconstruido
            model.eval()                                                               # Pone el modelo en modo inferencia 
            #--------------------------------------------------------------------------------------------------------------------------------------------------------


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
        Modelo B (opcional, recomendado): CNN propia.

        Debe:
          - Usar los splits train/valid definidos en la notebook.
          - Aplicar el preprocesamiento y data augmentation justificados.
          - Guardar el checkpoint resultante en self.active_checkpoint
            (ej: models/resnet18_finetuned.pth).
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.optim.lr_scheduler import StepLR
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from torchvision import datasets, models, transforms

        # =========================
        # CONFIG
        # =========================
        BATCH_SIZE = 32
        EPOCHS = 25
        LEARNING_RATE = 1e-4

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =========================
        # TRANSFORMS
        # =========================
        transform_etapa2 = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        transform_aug = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.GaussianBlur(kernel_size=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            #transforms.Lambda(lambda x: torch.clamp(x + 0.1 * torch.randn_like(x), 0, 1)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # =========================
        # DATASET
        # =========================
        train_dataset = datasets.ImageFolder(
            root=str(self.dataset_path / "train"),
            transform=transform_aug
        )

        valid_dataset = datasets.ImageFolder(
            root=str(self.dataset_path / "valid"),
            transform=transform_etapa2
        )

        # =========================
        # BALANCEO
        # =========================
        targets = train_dataset.targets
        class_counts = torch.bincount(torch.tensor(targets)).float()
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[targets]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            num_workers=2,
            drop_last=True
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )

        # =========================
        # MODEL
        # =========================
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 70)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = StepLR(optimizer, step_size=7, gamma=0.2)

        # =========================
        # EARLY STOPPING 
        # =========================
        patience = 7
        patience_counter = 0
        best_valid_loss = float('inf')

        # =========================
        # TRAIN LOOP
        # =========================
        for epoch in range(EPOCHS):

            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            # =========================
            # VALIDATION
            # =========================
            model.eval()
            running_valid_loss = 0.0
            correctas = 0
            total = 0

            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss_v = criterion(outputs, labels)

                    running_valid_loss += loss_v.item() * images.size(0)

                    _, predicciones = torch.max(outputs, 1)
                    total += labels.size(0)
                    correctas += (predicciones == labels).sum().item()

            epoch_valid_loss = running_valid_loss / len(valid_loader.dataset)
            epoch_acc = correctas / total

            print(
                f"Epoch [{epoch+1}/{EPOCHS}] | "
                f"Train Loss: {epoch_loss:.4f} | "
                f"Val Loss: {epoch_valid_loss:.4f} | "
                f"Acc: {epoch_acc:.4f}"
            )

            # =========================
            # SAVE + EARLY STOPPING
            # =========================
            if epoch_valid_loss < best_valid_loss:
                best_valid_loss = epoch_valid_loss
                patience_counter = 0

                torch.save(model.state_dict(), self.active_checkpoint)
                print(f"Modelo guardado en: {self.active_checkpoint}")

            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping")
                break

            scheduler.step()
        


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
        import torch
        import torch.nn as nn
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        import numpy as np

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # =========================
        # TRANSFORM 
        # =========================
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # =========================
        # DATASET TEST
        # =========================
        test_dataset = datasets.ImageFolder(
            root=str(self.dataset_path / "test"), 
            transform=transform
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False
        )

        # =========================
        # LOAD MODEL
        # =========================
        model = self.load_model()
        model = model.to(device)
        model.eval()

        # =========================
        # ACUMULACION DE PREDICCIONES Y LABELS
        # =========================
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)

                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # =========================
        # MATRIZ DE CONFUSION
        # =========================
        num_classes = len(test_dataset.classes)

        cm = np.zeros((num_classes, num_classes), dtype=int)

        for t, p in zip(all_labels, all_preds):
            cm[t][p] += 1

        # =========================
        # METRICS
        # =========================
        accuracy = np.trace(cm) / np.sum(cm)

        precision_list = []
        recall_list = []
        specificity_list = []

        for i in range(num_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - (tp + fp + fn)

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)

            precision_list.append(precision)
            recall_list.append(recall)
            specificity_list.append(specificity)

        precision = float(np.mean(precision_list))
        recall = float(np.mean(recall_list))
        specificity = float(np.mean(specificity_list))

        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return {
            "accuracy": float(accuracy),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": float(f1)
        }


#    def extract_custom_embedding(self, image: np.ndarray) -> list[float]:
#        """
#        Genera el embedding de una imagen usando el modelo propio activo
#        (penultima capa del ResNet18 fine-tuned o de la CNN custom).
#
#        Se usa cuando EMBEDDING_MODEL != baseline para que la busqueda por
#        similitud (Etapa 1) funcione con los modelos entrenados.
#        La imagen llega en BGR (OpenCV). Retorna una lista de floats de
#        dimension EMBEDDING_DIM.
#        """
#        raise NotImplementedError("Etapa 2: implementar extract_custom_embedding")
