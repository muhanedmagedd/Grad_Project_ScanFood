# ── Stage 1: base ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps needed by OpenCV + PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download MiDaS (so the image is self-contained) ──────────────────────
RUN python - <<'EOF'
import torch
model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
_ = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
print("MiDaS cached ✅")
EOF

# ── Copy source ───────────────────────────────────────────────────────────────
COPY inference.py .
COPY main.py .
COPY best.pt /app/best.pt
# YOLO weights: mount at runtime  →  docker run -v /path/to/best.pt:/app/weights/best.pt
# OR copy them into the image (makes image bigger):
# COPY weights/ weights/

ENV YOLO_WEIGHTS=/app/weights/best.pt
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
