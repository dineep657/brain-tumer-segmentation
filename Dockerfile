FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BT_DB_BACKEND=sqlite \
    BT_SQLITE_PATH=/app/webapp/data/brainscan.db

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU PyTorch first (smaller than full CUDA builds)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

COPY requirements-webapp.txt .
RUN pip install --no-cache-dir -r requirements-webapp.txt

COPY inference_2d.py classifier_2d.py gradcam.py image_processing.py \
     train_2d_unet.py train_2d_classifier.py ./
COPY models ./models
COPY webapp ./webapp
COPY schema.sql .

RUN mkdir -p webapp/uploads/doctor_docs webapp/data

EXPOSE 8000

CMD ["python", "webapp/start.py"]
