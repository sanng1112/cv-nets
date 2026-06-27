FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# Install package
RUN uv sync --frozen --no-dev

# Verify installation
RUN uv run python -c "import cvnets; print(f'cvnets {cvnets.__version__} ready')"

ENTRYPOINT ["uv", "run"]
CMD ["cvnets-train", "--help"]
