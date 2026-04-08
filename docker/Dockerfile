# Reproducible CPU image for deep-ts-imputer.
# For GPU, swap the base image for tensorflow/tensorflow:2.15.0-gpu
# and add `--gpus all` to your `docker run` command.
FROM tensorflow/tensorflow:2.15.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install OS deps needed by matplotlib + git for editable installs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first so the Docker layer cache works.
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the source.
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
COPY README.md LICENSE ./

RUN pip install -e .

# Create mount points for data and outputs.
RUN mkdir -p /app/data /app/outputs

# Default: print available models. Override with docker-compose or `docker run`.
CMD ["python", "-c", "from deep_ts_imputer.models.factory import available_models; print(available_models())"]
