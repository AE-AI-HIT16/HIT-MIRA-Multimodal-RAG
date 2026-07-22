# Colab MediaTaskWorker Deployment

This guide is for processing all videos already registered in MinIO/PostgreSQL
with the queue-style `MediaTaskWorker`.

## Architecture

The server remains the queue and storage source of truth:

```text
PostgreSQL processing_jobs
MinIO video/frame objects
```

Colab provides GPU workers for model-heavy tasks:

```text
ocr worker             uses EasyOCR on Colab
caption worker         uses the configured caption model
object_detection worker uses YOLO/Ultralytics
asr worker             uses faster-whisper
```

The recommended split is:

```text
server:
  enqueue existing videos
  run keyframe worker, or run it on Colab if preferred

colab:
  run ocr/caption/object_detection/asr workers
```

## Server: Enqueue All Existing Videos

From the repository root on the server:

```bash
cd /home/ubuntu/HIT-MIRA-Multimodal-RAG
PYTHONPATH=API python scripts/enqueue_existing_video_jobs.py
```

This creates one `keyframe` job for each `media.media_type = 'video'`.
It does not duplicate existing `PENDING`, `PROCESSING`, or `DONE`
keyframe jobs.

## Server: Start Keyframe Worker

Keyframe extraction is relatively light and can run on the server:

```bash
PYTHONPATH=API python -m src.rag_video_anh.pipeline.media_task_worker \
  --task-type keyframe \
  --loop
```

Each completed keyframe job creates downstream jobs:

```text
ocr              one job per frame
caption          one job per frame
object_detection one job per frame
asr              one job per video
```

## Colab: Environment

Clone the same branch in Colab, then install only the runtime dependencies
needed by workers:

```python
!git clone https://github.com/AE-AI-HIT16/HIT-MIRA-Multimodal-RAG.git
%cd HIT-MIRA-Multimodal-RAG
!git checkout Linhttm/feature/pipeline_media
!pip install sqlalchemy "psycopg[binary]" minio python-dotenv opencv-python-headless
!pip install easyocr ultralytics faster-whisper transformers accelerate torch torchvision
```

Set environment variables so Colab connects to the server services:

```python
%env CONFIG_PATH=./API/Resources/dev.yaml
%env MODELS_PATH=./API/Resources/model.yaml
%env PROMPTS_PATH=./API/Resources/prompt.yaml
%env DATABASE_URL=postgresql+psycopg://hit:<password>@<server-ip>:5432/hit_mira
%env MINIO_ENDPOINT=<server-ip>:9000
%env MINIO_ACCESS_KEY=<minio-access-key>
%env MINIO_SECRET_KEY=<minio-secret-key>
%env MINIO_BUCKET=hit-mira-media
%env MINIO_SECURE=false
%env OCR_BACKEND=easyocr
```

For OCR on Colab, use EasyOCR instead of PaddleOCR:

```python
%env OCR_BACKEND=easyocr
```

## Colab: Run Workers

Run one worker per notebook/session, or use background processes.

OCR:

```python
!PYTHONPATH=API OCR_BACKEND=easyocr python -m src.rag_video_anh.pipeline.media_task_worker --task-type ocr --loop
```

Caption:

```python
!PYTHONPATH=API OCR_BACKEND=easyocr python -m src.rag_video_anh.pipeline.media_task_worker --task-type caption --loop
```

Object detection:

```python
!PYTHONPATH=API OCR_BACKEND=easyocr python -m src.rag_video_anh.pipeline.media_task_worker --task-type object_detection --loop
```

ASR:

```python
!PYTHONPATH=API OCR_BACKEND=easyocr python -m src.rag_video_anh.pipeline.media_task_worker --task-type asr --loop
```

For a quick smoke test, use `--once`:

```python
!PYTHONPATH=API OCR_BACKEND=easyocr python -m src.rag_video_anh.pipeline.media_task_worker --task-type ocr --once
```

## Security Note

Direct Colab workers need network access to PostgreSQL and MinIO. Prefer a VPN,
SSH tunnel, or a temporary firewall rule restricted to the current Colab runtime
when possible. A safer alternative is the artifact flow:

```text
Colab processes video and writes outputs/<media_id>/
server downloads artifact folder
server runs scripts/import_colab_media_outputs.py
```

See `docs/colab-video-artifact-flow.md` for that import-based option.
