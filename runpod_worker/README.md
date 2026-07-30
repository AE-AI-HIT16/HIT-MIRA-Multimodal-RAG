# HIT-MIRA Runpod worker

This folder packages the existing video pipeline as a Runpod **queue-based**
worker. It accepts a presigned video download URL and writes a ZIP of the
standard artifact layout to a presigned upload URL. The GPU worker has no
database credentials and does not write directly to MinIO or PostgreSQL.

## Job input

```json
{
  "input": {
    "media_id": "video-media-uuid",
    "video_url": "https://storage.example/...signed-download...",
    "artifact_upload_url": "https://storage.example/...signed-upload...",
    "bucket_name": "hit-mira-media",
    "source_object_key": "events/post-id/media/video_01.mp4",
    "language": "vi"
  }
}
```

`artifact_upload_url` is optional for a local handler test, but required in
production. It must accept an HTTP `PUT` of `application/zip`.

## Build and publish

From the repository root:

```bash
docker build --platform linux/amd64 \
  -f runpod_worker/Dockerfile \
  -t hoangtechcs/hit-mira-runpod-worker:v0.1.3 .
docker push hoangtechcs/hit-mira-runpod-worker:v0.1.3
```

Deploy `hoangtechcs/hit-mira-runpod-worker:v0.1.3` as a Queue endpoint. Use
the asynchronous `/run` API, then import the uploaded ZIP on the trusted
backend with the existing artifact importer.
