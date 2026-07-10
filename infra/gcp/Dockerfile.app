# Public showcase image: static playground + API + champions inference.
# Stage 1 builds the Next.js static export; stage 2 is the Python serve image
# with the `cloud` extra (MLflow read-layer + champion pulls at startup).
FROM node:20-slim AS ui
WORKDIR /ui
COPY playground/package.json playground/package-lock.json ./
RUN npm ci
COPY playground/ .
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY main.py .

# CPU torch wheels — this image only runs inference on the mini champions.
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.1,<2.3" "torchvision>=0.16,<0.18"
RUN uv pip install --system ".[cloud]"

# api.main mounts <repo>/playground/out at / when it exists.
COPY --from=ui /ui/out playground/out
COPY infra/gcp/entrypoint-app.sh /app/entrypoint-app.sh
RUN chmod +x /app/entrypoint-app.sh

ENTRYPOINT ["/app/entrypoint-app.sh"]
