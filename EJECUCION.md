# Guía de ejecución — TP2 Dog Breed Recognition

Pasos simples para poner a andar el sistema completo de detección y clasificación
de razas de perros. Seguilos en orden.

## Equipo
- Gianfranco Frattini
- Matías Prado

---

## Requisitos previos

- **Python 3.12** (para el entorno local / entrenamiento con GPU)
- **Docker Desktop** (incluye Docker Compose — para levantar la app)
- **Git**
- **Cuenta de Kaggle** con credenciales configuradas (para descargar el dataset)
- **GPU NVIDIA** recomendada para entrenar (RTX 3080 / Colab T4); si no, corre en CPU pero lento

---

## 1. Setup inicial (una sola vez)

### 1.1 Clonar el fork y crear el entorno virtual

```powershell
git clone https://github.com/GianfrancoFrattini/tuia-dog-recognition-app.git
cd tuia-dog-recognition-app
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Si `Activate.ps1` falla por política de ejecución, ejecutá primero:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 1.2 Instalar PyTorch con CUDA (para usar GPU)

El `requirements.txt` instala la versión CPU de torch por defecto. Para entrenar
con GPU hay que reinstalarla con CUDA:

```powershell
pip uninstall torch torchvision -y
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

Verificá que CUDA esté disponible:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

Debe imprimir `CUDA: True | NVIDIA GeForce RTX 3080` (o tu GPU).

### 1.3 Descargar el dataset

Necesitás credenciales de Kaggle en `C:\Users\<tu_usuario>\.kaggle\kaggle.json`
(se consiguen en https://www.kaggle.com/settings → API → Create New Token).

```powershell
python scripts/download_dataset.py
```

Esto descarga **70 Dog Breeds Image Dataset** en `data/dataset/` con los splits
`train/`, `valid/` y `test/` (una subcarpeta por raza).

> Alternativa manual: descargá el ZIP desde
> https://www.kaggle.com/datasets/gpiosenka/70-dog-breedsimage-data-set
> y descomprimí `train/`, `valid/` y `test/` dentro de `data/dataset/`.

### 1.4 Configurar variables de entorno

La configuración ya viene lista en `.env.docker.example` (para docker) y
`.env.local.example` (para local). Si querés ajustar algo (modelo, threshold,
top-k, paths), editá esos archivos.

Para uso local, copiá el ejemplo a `src/.env`:

```powershell
copy .env.local.example src\.env
```

---

## 2. Entrenar los modelos (Etapa 2)

El entrenamiento crea los checkpoints `.pth` que usa la app para clasificar.
Hay dos modelos obligatorios:

- **Modelo A**: ResNet18 fine-tuned (transfer learning de ImageNet)
- **Modelo B**: CNN propia (entrenada desde cero)

### 2.1 Entrenar con GPU local (recomendado)

```powershell
.venv\Scripts\Activate.ps1
python scripts/train_classifier.py --model resnet18_finetuned
python scripts/train_classifier.py --model cnn_custom
```

Cada entrenamiento:
- Usa **25 epochs** con **early stopping** (para si la validación no mejora en 5 epochs)
- Guarda el checkpoint en `models/<modelo>.pth`
- Genera gráficos en `training_results/<modelo>/`:
  - `training_curves.png` — loss y accuracy (train vs val) por epoch
  - `confusion_matrix.png` — matriz de confusión sobre test
  - `history.json` — curvas en formato JSON
  - `metrics.json` — accuracy, precision, recall, specificity, F1

> Con RTX 3080 cada modelo tarda unos minutos. En CPU puede tardar horas.

### 2.2 Alternativa: entrenar en Google Colab

Subí `etapa2_colab.ipynb` a Colab, activá la GPU
(Entorno de ejecución → Cambiar tipo → GPU T4) y ejecutá las celdas en orden.
Al final, descargá los checkpoints de `models/` a tu entorno local.

### 2.3 Evaluar los modelos

```powershell
python scripts/train_classifier.py --model resnet18_finetuned --skip-train
python scripts/train_classifier.py --model cnn_custom --skip-train
```

Imprime las métricas (accuracy, precision, recall, specificity, F1).

---

## 3. Indexar la base vectorial (Etapa 1)

La búsqueda por similitud necesita que las imágenes del dataset estén indexadas
como embeddings en la base vectorial (PostgreSQL + pgvector).

### 3.1 Levantar PostgreSQL

```powershell
docker compose up postgres -d
```

### 3.2 Indexar el dataset

```powershell
.venv\Scripts\Activate.ps1
python scripts/reindex_docker.py
```

Esto extrae embeddings de las ~7946 imágenes de `train/` con el modelo baseline
(ResNet50, 2048-d) y los guarda en pgvector con **paths en formato Linux**
( necesarios para que la app docker sirva las imágenes similares).

> **Importante:** esto se hace **una sola vez**. Los embeddings persisten en el
> volumen docker `pgdata` y sobreviven a cualquier `docker compose down/up`.
> Con GPU tarda unos minutos.

Verificá que haya quedado bien:

```powershell
docker exec tuia-dog-recognition-app-postgres-1 psql -U dogs_user -d dogs -c "SELECT COUNT(*), COUNT(DISTINCT breed) FROM embeddings;"
```

Debe mostrar ~7946 embeddings y 70 razas.

---

## 4. Levantar la aplicación con Docker

```powershell
docker compose build
docker compose up -d
```

Servicios:
- **Frontend (Gradio):** http://localhost:8080
- **Backend (FastAPI):** http://localhost:8000
- **PostgreSQL:** localhost:5432

Verificá que ande:

```powershell
curl http://localhost:8000/health
```

Debe responder `{"status":"ok", ...}` con los checkpoints en `true`.

> **No hace falta reindexar** cada vez que levantás docker: los embeddings
> quedan en el volumen `pgdata`. Solo reindexá si cambiás el modelo de embeddings.

---

## 5. Usar la aplicación

Abrí http://localhost:8080 en el navegador. Hay 3 pestañas:

### Etapa 1 — Búsqueda por similitud
1. Subí una imagen de un perro.
2. Elegí el modelo de embeddings (`baseline` / `resnet18_finetuned` / `cnn_custom`).
3. Ajustá Top K (default 10) y pulsá **Buscar similares**.
4. Consultá el resultado con el botón **Consultar resultado de este job**.
5. Verás: imagen consultada, top K imágenes similares, raza predicha y score.

### Etapa 2 — Clasificación supervisada
1. Subí una imagen de un perro.
2. Elegí el modelo entrenado (`resnet18_finetuned` / `cnn_custom`).
3. Pulsá **Clasificar** y consultá el resultado.
4. Verás: raza predicha y score de confianza.

### Etapa 3 — Detección y clasificación
1. Subí una imagen (con uno o varios perros).
2. Pulsá **Detectar y clasificar** y consultá el resultado.
3. Verás: bounding boxes, raza predicha y scores de confianza sobre cada perro.

---

## 6. Evaluar la Etapa 1 (NDCG@10)

Para medir la calidad de la búsqueda por similitud sobre el conjunto de test:

```powershell
.venv\Scripts\Activate.ps1
python scripts/evaluate_search.py --split test
```

Calcula NDCG@10 medio (relevancia binaria: 1.0 si el vecino es de la misma raza
que la consulta, 0.0 si no).

---

## 7. Cambiar el modelo de embeddings (opcional)

Por defecto la Etapa 1 usa `baseline` (ResNet50, 2048-d). Para usar un modelo
fine-tuned (512-d) hay que reindexar la base:

1. Editá `.env.docker.example` (y `src/.env` si usás local):
   ```
   EMBEDDING_MODEL=resnet18_finetuned
   EMBEDDING_DIM=512
   ```
2. Reindexá:
   ```powershell
   python scripts/reindex_docker.py
   ```
3. Reiniciá docker:
   ```powershell
   docker compose down
   docker compose up -d
   ```

> Mientras no cambies de modelo, **no hace falta reindexar** al reiniciar docker.

---

## 8. Tests

```powershell
.venv\Scripts\Activate.ps1
pytest tests/ -v
```

Tests de lógica pura (voto mayoritario, similitud coseno/L2, NDCG, métricas,
`_clip_xyxy`). No requieren dataset ni modelos entrenados.

---

## 9. Problemas comunes

### "CUDA: False" al entrenar
Tenés instalada la versión CPU de torch. Reinstalá con CUDA (ver paso 1.2).

### "Top K no muestra imágenes" en el frontend
Los paths en la base vectorial no están en formato Linux. Reindexá con
`scripts/reindex_docker.py` (ver paso 3.2).

### "operator does not exist: vector <=> double precision[]"
Error de tipo al buscar en pgvector. Ya está fixeado en el código; asegurate
de tener la última versión de `similarity_service.py`.

### Docker: "container name /backend already in use"
Hay un contenedor viejo con ese nombre. Eliminalo:
```powershell
docker rm -f backend frontend
docker compose up -d
```

### El contenedor backend se cae al indexar dentro de docker
Docker no tiene GPU por defecto. Indexá desde el host con GPU
(`scripts/reindex_docker.py`) en vez de dentro del contenedor.

---

## 10. Entrega

1. Asegurate de que `docker compose build` + `docker compose up` anden sin errores.
2. Commiteá los cambios al branch `DogRecognitionApp-Frattini-Prado`.
3. Subí los checkpoints a un link público de solo lectura (Google Drive, etc.)
   para que los docentes puedan descargarlos.
4. Abrí un Pull Request contra `upstream/main` (TinoFerrucci/tuia-dog-recognition-app).
5. Entregá el link del PR en el campus.
