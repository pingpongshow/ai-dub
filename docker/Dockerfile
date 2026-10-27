# Ai Dub image: the app and its tools only. No model weights and no model
# environments are baked in; they are installed into /models on first use
# (Models page, or AIDUB_INSTALL_ON_START=true). The NVIDIA container runtime
# provides the driver; each model environment brings its own CUDA libraries.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    AIDUB_CONFIG=/config/config.yaml \
    AIDUB_WEB=/app/web \
    AIDUB_WORKERS=/app/workers \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_CACHE_DIR=/models/uv-cache \
    UV_LINK_MODE=hardlink \
    HF_HOME=/models/hf \
    HOME=/config/home \
    PUID=99 \
    PGID=100 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg mkvtoolnix rubberband-cli sox libsox-fmt-all libsndfile1 \
      git git-lfs ca-certificates curl build-essential tini gosu \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /usr/local/bin/

# The app itself runs in its own small environment.
COPY pyproject.toml README.md LICENSE /app/
COPY aidub /app/aidub
RUN uv python install 3.12 3.11 \
    && uv venv --python 3.12 /opt/app \
    && UV_CACHE_DIR=/tmp/uv-cache uv pip install --python /opt/app/bin/python --no-cache /app \
    && rm -rf /tmp/uv-cache \
    && chmod -R a+rX /opt/python /opt/app

COPY workers /app/workers
COPY web /app/web
COPY docker/entrypoint.sh /app/entrypoint.sh
COPY docker/config.example.yaml /app/docker/config.example.yaml
RUN chmod +x /app/entrypoint.sh

ENV PATH="/opt/app/bin:${PATH}"
EXPOSE 8790
VOLUME ["/config", "/models", "/work"]
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
CMD ["aidub", "serve"]
