FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (layer cache)
COPY pyproject.toml uv.lock ./

# Install dependencies (no dev extras)
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ src/
COPY main.py .

# Install the package itself (no deps — already installed above)
RUN uv pip install -e . --no-deps

# Persistent storage for run artifacts and data cache
VOLUME ["/app/runs", "/tmp/mini_networks_data"]

EXPOSE 8000

ENTRYPOINT ["uv", "run", "python", "main.py"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
