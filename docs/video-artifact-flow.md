# Video Artifact Flow

Use this flow on the GPU server for offline inference. The server writes results to MinIO and PostgreSQL.

## Output Layout

Each pipeline run must produce one folder:

```text
outputs/<media_id>/
  manifest.json
  frames/
    frame_000123.jpg
  ocr.json
  captions.json
  detections.json
  transcript.json
  normalized_metadata.json
  run_log.txt
```

## Required Manifest Fields

```json
{
  "media_id": "video-media-uuid",
  "bucket_name": "hit-mira-media",
  "source_object_key": "events/post-id/media/video_01.mp4",
  "video_file": "video_01.mp4",
  "status": "done",
  "video": {
    "frame_count": 1000,
    "fps": 25.0,
    "duration_sec": 40.0,
    "width": 1280,
    "height": 720
  },
  "frames": [
    {
      "frame_id": "video-media-uuid_f000123",
      "frame_index": 123,
      "timestamp_sec": 4.92,
      "timestamp_ms": 4920,
      "file": "frames/frame_000123.jpg"
    }
  ],
  "outputs": {
    "ocr": "ocr.json",
    "captions": "captions.json",
    "detections": "detections.json",
    "transcript": "transcript.json",
    "normalized_metadata": "normalized_metadata.json"
  }
}
```

## Result JSON Shapes

`ocr.json`:

```json
{
  "media_id": "video-media-uuid",
  "status": "done",
  "model": "paddleocr",
  "results": [
    {
      "frame_id": "video-media-uuid_f000123",
      "status": "done",
      "full_text": "recognized text",
      "confidence": 0.93,
      "text_spans": [
        {
          "text": "recognized text",
          "confidence": 0.93,
          "bbox": [[0, 0], [100, 0], [100, 40], [0, 40]]
        }
      ]
    }
  ]
}
```

`captions.json`:

```json
{
  "media_id": "video-media-uuid",
  "status": "done",
  "model": "microsoft/Florence-2-base-ft",
  "results": [
    {
      "frame_id": "video-media-uuid_f000123",
      "status": "done",
      "caption_text": "a detailed frame caption"
    }
  ]
}
```

`detections.json`:

```json
{
  "media_id": "video-media-uuid",
  "status": "done",
  "model": "yolo11n.pt",
  "results": [
    {
      "frame_id": "video-media-uuid_f000123",
      "status": "done",
      "detections": [
        {
          "label": "person",
          "confidence": 0.88,
          "bbox": [10, 20, 120, 240]
        }
      ]
    }
  ]
}
```

`transcript.json`:

```json
{
  "media_id": "video-media-uuid",
  "status": "done",
  "language": "vi",
  "model": "large-v3",
  "segments": [
    {
      "start_sec": 0.0,
      "end_sec": 4.2,
      "text": "transcribed speech"
    }
  ]
}
```

## Server Import

After a pipeline run produces `outputs/<media_id>/` on the GPU server:

```bash
cd /home/ubuntu/HIT-MIRA-Multimodal-RAG
export PYTHONPATH=API
python scripts/import_media_outputs.py /path/to/outputs/<media_id>
```

The importer writes:

- keyframe JPEGs to MinIO under `frames/<media_id>/...`
- `videos`
- frame rows in `media` and `frames`
- `ocr_results` and `ocr_boxes`
- `caption_results`
- `object_results` and `detected_objects`
- `transcripts` and `transcript_segments`
