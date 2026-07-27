# GPU worker for the offline HIT-MIRA video-processing pipeline.
# Build from the repository root:
#   docker build --platform linux/amd64 -f runpod_worker/Dockerfile \
#     -t hoangtechcs/hit-mira-runpod-worker:v0.1.3 .
# PyTorch installs its own CUDA/cuDNN runtime wheels below.  Using the plain
# CUDA runtime base avoids shipping a second cuDNN stack in the image.
# RTX PRO 6000 Blackwell workers use compute capability sm_120. PyTorch
# CUDA 12.9 wheels include support for that architecture.
FROM nvidia/cuda:12.9.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/API \
    CONFIG_PATH=API/Resources/dev.yaml \
    MODELS_PATH=API/Resources/model.yaml \
    PROMPTS_PATH=API/Resources/prompt.yaml \
    HF_HOME=/tmp/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 python3 python3-pip python3-venv \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY runpod_worker/requirements.txt /app/runpod-worker-requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu129 torch==2.11.0+cu129 torchvision==0.26.0+cu129
RUN python -m pip install -r /app/runpod-worker-requirements.txt

# PipelineService imports the MinIO helper even though the remote worker only
# exchanges presigned URLs, so keep this small dependency in a separate layer.
RUN python -m pip install minio
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); assert torch.version.cuda == '12.9'"

COPY API /app/API
COPY scripts/export_video_artifacts.py /app/scripts/export_video_artifacts.py
COPY runpod_worker /app/runpod_worker
RUN python -c "from src.rag_video_anh.pipeline.pipeline_service import PipelineService; print('pipeline_import_ok')"

CMD ["python", "-u", "/app/runpod_worker/handler.py"]
