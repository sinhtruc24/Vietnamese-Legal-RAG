FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch keeps the image small; embeddings for one query and
# reranking 30 candidates are fast enough on CPU. The LLM runs in its own container.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[ml,serve]"

COPY app ./app

EXPOSE 8080
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
