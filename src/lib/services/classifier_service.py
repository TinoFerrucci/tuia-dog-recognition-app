from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import onnxruntime
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm.auto import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

logger = logging.getLogger(__name__)

class CustomCNN(nn.Module):
    """
    CNN propia de 4 bloques convolucionales con BatchNorm, ReLU y MaxPool.
    Cabeza clasificadora con Global Average Pooling + 2 capas densas y Dropout.
    """

    def __init__(self, num_classes: int, dropout_p: float = 0.4):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.features = nn.Sequential(
            conv_block(3,   64),   # 224 -> 112
            conv_block(64,  128),  # 112 -> 56
            conv_block(128, 256),  # 56  -> 28
            conv_block(256, 512),  # 28  -> 14
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # -> 512 x 1 x 1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)


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

    def train_classifier(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-4,
        patience: int = 5,
        weight_decay: float = 1e-4,
        dropout_p: float = 0.4,
        scheduler_patience: int = 2,
    ) -> dict:
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

        logger.info(f"Iniciando entrenamiento para el modelo: {self.active_model_name}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Rutas del dataset
        train_dir = self.dataset_path / "train"
        valid_dir = self.dataset_path / "valid"

        if not train_dir.exists() or not valid_dir.exists():
            raise FileNotFoundError(f"No se encontraron las carpetas train/valid en {self.dataset_path}")

        # Transformaciones (Preprocesamiento y Data Augmentation)
        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        valid_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Carga de Datasets
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transform)
        num_classes = len(train_dataset.classes)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Seleccion de Arquitectura 
        if "resnet18" in self.active_model_name:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.Linear(num_ftrs, num_classes),
            )
                
        elif "resnet50" in self.active_model_name:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.Linear(num_ftrs, num_classes),
            )
            
        elif self.active_model_name == "cnn_custom":
            model = CustomCNN(num_classes=num_classes, dropout_p=dropout_p)
            
        else:
            raise ValueError(f"Arquitectura no soportada: {self.active_model_name}")

        model = model.to(device)

        # Correción del desbalance de clases 
        y_train_labels = train_dataset.targets 
        class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
        class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float).to(device)
        logger.info("Pesos de clase calculados e integrados para balanceo.")

        # Optimizador y Función de Pérdida
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience)

        # Bucle de Entrenamiento
        best_acc = 0.0
        history = {'train_loss': [], 'val_acc': []}
        epochs_no_improve = 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

            # Validación
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = correct / total
            history['train_loss'].append(running_loss/len(train_loader))
            history['val_acc'].append(val_acc)
            logger.info(f"Epoch {epoch+1} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

            scheduler.step(val_acc)

            # Guardar el mejor modelo
            if val_acc > best_acc:
                best_acc = val_acc
                epochs_no_improve = 0
                
                self.active_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model, self.active_checkpoint)
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                logger.info(f"Nuevo mejor modelo encontrado con Acc: {best_acc:.4f}")
            else:
                epochs_no_improve += 1
                logger.info(f"Sin mejoras por {epochs_no_improve} época(s).")
                
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping activado. El modelo no mejoró en {patience} épocas.")
                    break 

        if best_state:
            model.load_state_dict(best_state)

        return history


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
        model = self.load_model()
        model = model.to(device)
        model.eval()

        test_dir = self.dataset_path / "test"
        test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Evaluando modelo"):
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        # Calcular métricas
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        specificities = []
        for i in range(len(cm)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            
        spec = np.mean(specificities)

        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "specificity": float(spec),
            "f1": float(f1),
            "confusion_matrix": cm.tolist()
        }
        
        return metrics
