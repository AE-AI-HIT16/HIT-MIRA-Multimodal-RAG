from typing import Any, Optional

from pydantic import BaseModel

from src.config.config import config_models, config_object, config_prompts


def _config_value(root: Any, dotted_path: str, default: Any = None) -> Any:
    """Read optional nested config values from YAML-backed objects."""
    current = root
    for part in dotted_path.split("."):
        if current is None or not hasattr(current, part):
            return default
        current = getattr(current, part)
    return current


def _config_bool(root: Any, dotted_path: str, default: bool = False) -> bool:
    """Read optional boolean config values from YAML-backed objects."""
    value = _config_value(root, dotted_path, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


class QdrantConfig(BaseModel):
    """Qdrant connection settings."""
    url: Optional[str] = config_object.QDRANT.URL
    api_key: Optional[str] = config_object.QDRANT.API_KEY
    collection_name: Optional[str] = config_object.QDRANT.COLLECTION_NAME
    dense_vector_size: Optional[int] = config_object.QDRANT.DENSE_VECTOR_SIZE

class EmbeddingConfig(BaseModel):
    """Embedding model settings."""
    api_key: Optional[str] = config_object.EMBEDDING.API_KEY
    base_url: Optional[str] = config_object.EMBEDDING.EMBEDDING_BASE_URL
    dimensions: Optional[int] = config_object.EMBEDDING.EMBEDDING_DIMENSIONS
    model: Optional[str] = config_object.EMBEDDING.EMBEDDING_MODEL

class PipelineConfig(BaseModel):
    """Document pipeline settings."""
    chunk_size: Optional[int] = config_object.PIPELINE.CHUNK_SIZE
    chunk_overlap: Optional[int] = config_object.PIPELINE.CHUNK_OVERLAP
    minimum_chunk_size: Optional[int] = config_object.PIPELINE.MINIMUM_CHUNK_SIZE

class RetrievalConfig(BaseModel):
    """Retrieval pipeline settings."""
    top_k: Optional[int] = config_object.RETRIEVAL.TOP_K
    keyword_threshold: Optional[float] = config_object.RETRIEVAL.KEYWORD_THRESHOLD

class MinioConfig(BaseModel):
    """MinIO object storage connection settings."""
    endpoint: str = _config_value(config_object, "MINIO.ENDPOINT", "localhost:9000")
    access_key: str = _config_value(config_object, "MINIO.ACCESS_KEY", "minioadmin")
    secret_key: str = _config_value(config_object, "MINIO.SECRET_KEY", "minioadmin")
    bucket_name: str = _config_value(config_object, "MINIO.BUCKET", "hit-mira-media")
    secure: bool = _config_bool(config_object, "MINIO.SECURE", False)

class MediaPipelineConfig(BaseModel):
    """Media pipeline business and orchestration settings."""
    parallel_enrichment: bool = _config_value(config_object, "MEDIA_PIPELINE.PARALLEL_ENRICHMENT", True)
    stage_timeout: Optional[float] = _config_value(config_object, "MEDIA_PIPELINE.STAGE_TIMEOUT", None)
    retry_count: int = _config_value(config_object, "MEDIA_PIPELINE.RETRY_COUNT", 0)
    partial_success_policy: bool = _config_value(config_object, "MEDIA_PIPELINE.PARTIAL_SUCCESS_POLICY", True)
    supported_media_types: list[str] = _config_value(
        config_object,
        "MEDIA_PIPELINE.SUPPORTED_MEDIA_TYPES",
        [".mp4", ".mov", ".mkv", ".avi", ".webm"],
    )
    validation_strictness: str = _config_value(config_object, "MEDIA_PIPELINE.VALIDATION_STRICTNESS", "strict")
    minimum_duration: Optional[float] = _config_value(config_object, "MEDIA_PIPELINE.MINIMUM_DURATION", None)
    maximum_duration: Optional[float] = _config_value(config_object, "MEDIA_PIPELINE.MAXIMUM_DURATION", None)
    minimum_size_bytes: Optional[int] = _config_value(config_object, "MEDIA_PIPELINE.MINIMUM_SIZE_BYTES", None)
    maximum_size_bytes: Optional[int] = _config_value(config_object, "MEDIA_PIPELINE.MAXIMUM_SIZE_BYTES", None)
    audio_required: bool = _config_value(config_object, "MEDIA_PIPELINE.AUDIO_REQUIRED", False)
    enable_ocr: bool = _config_value(config_object, "MEDIA_PIPELINE.ENABLE_OCR", True)
    enable_caption: bool = _config_value(config_object, "MEDIA_PIPELINE.ENABLE_CAPTION", True)
    enable_detection: bool = _config_value(config_object, "MEDIA_PIPELINE.ENABLE_DETECTION", True)
    enable_asr: bool = _config_value(config_object, "MEDIA_PIPELINE.ENABLE_ASR", True)
    short_scene_threshold: int = _config_value(config_object, "MEDIA_PIPELINE.SHORT_SCENE_THRESHOLD", 10)
    keyframe_base_positions: list[float] = _config_value(config_object, "MEDIA_PIPELINE.KEYFRAME_BASE_POSITIONS", [0.0, 0.5, 1.0])
    keyframe_long_scene_interval_sec: float = _config_value(config_object, "MEDIA_PIPELINE.KEYFRAME_LONG_SCENE_INTERVAL_SEC", 2.5)
    keyframe_max_candidates_per_scene: int = _config_value(config_object, "MEDIA_PIPELINE.KEYFRAME_MAX_CANDIDATES_PER_SCENE", 8)
    fps_fallback: float = _config_value(config_object, "MEDIA_PIPELINE.FPS_FALLBACK", 25.0)
    min_blur_score: float = _config_value(config_object, "MEDIA_PIPELINE.MIN_BLUR_SCORE", 80.0)
    min_brightness: float = _config_value(config_object, "MEDIA_PIPELINE.MIN_BRIGHTNESS", 30.0)
    max_brightness: float = _config_value(config_object, "MEDIA_PIPELINE.MAX_BRIGHTNESS", 225.0)
    min_contrast: float = _config_value(config_object, "MEDIA_PIPELINE.MIN_CONTRAST", 10.0)
    brightness_midpoint: float = _config_value(config_object, "MEDIA_PIPELINE.BRIGHTNESS_MIDPOINT", 127.5)
    quality_blur_reference: float = _config_value(config_object, "MEDIA_PIPELINE.QUALITY_BLUR_REFERENCE", 400.0)
    quality_contrast_reference: float = _config_value(config_object, "MEDIA_PIPELINE.QUALITY_CONTRAST_REFERENCE", 64.0)
    quality_blur_weight: float = _config_value(config_object, "MEDIA_PIPELINE.QUALITY_BLUR_WEIGHT", 0.45)
    quality_brightness_weight: float = _config_value(config_object, "MEDIA_PIPELINE.QUALITY_BRIGHTNESS_WEIGHT", 0.25)
    quality_contrast_weight: float = _config_value(config_object, "MEDIA_PIPELINE.QUALITY_CONTRAST_WEIGHT", 0.30)
    quality_fallback_selection: str = _config_value(config_object, "MEDIA_PIPELINE.QUALITY_FALLBACK_SELECTION", "best_quality_candidate")
    frame_id_pattern: str = _config_value(config_object, "MEDIA_PIPELINE.FRAME_ID_PATTERN", "{media_id}_f{index:06d}")
    keyframe_output_sort_order: str = _config_value(config_object, "MEDIA_PIPELINE.KEYFRAME_OUTPUT_SORT_ORDER", "frame_index")
    jpeg_quality: int = _config_value(config_object, "MEDIA_PIPELINE.JPEG_QUALITY", 95)
    ocr_batch_size: int = _config_value(config_object, "MEDIA_PIPELINE.OCR_BATCH_SIZE", 1)
    ocr_min_confidence: float = _config_value(config_object, "MEDIA_PIPELINE.OCR_MIN_CONFIDENCE", 0.25)
    ocr_min_box_width: float = _config_value(config_object, "MEDIA_PIPELINE.OCR_MIN_BOX_WIDTH", 8.0)
    ocr_min_box_height: float = _config_value(config_object, "MEDIA_PIPELINE.OCR_MIN_BOX_HEIGHT", 12.0)
    ocr_min_box_area_ratio: float = _config_value(config_object, "MEDIA_PIPELINE.OCR_MIN_BOX_AREA_RATIO", 0.00003)
    ocr_min_text_chars: int = _config_value(config_object, "MEDIA_PIPELINE.OCR_MIN_TEXT_CHARS", 2)
    ocr_logo_max_area_ratio: float = _config_value(config_object, "MEDIA_PIPELINE.OCR_LOGO_MAX_AREA_RATIO", 0.015)
    ocr_logo_corner_margin_ratio: float = _config_value(config_object, "MEDIA_PIPELINE.OCR_LOGO_CORNER_MARGIN_RATIO", 0.18)
    ocr_logo_max_chars: int = _config_value(config_object, "MEDIA_PIPELINE.OCR_LOGO_MAX_CHARS", 8)
    ocr_other_min_text_chars: int = _config_value(config_object, "MEDIA_PIPELINE.OCR_OTHER_MIN_TEXT_CHARS", 4)
    ocr_other_min_width_ratio: float = _config_value(config_object, "MEDIA_PIPELINE.OCR_OTHER_MIN_WIDTH_RATIO", 0.03)
    ocr_other_min_area_ratio: float = _config_value(config_object, "MEDIA_PIPELINE.OCR_OTHER_MIN_AREA_RATIO", 0.0003)
    caption_batch_size: int = _config_value(config_object, "MEDIA_PIPELINE.CAPTION_BATCH_SIZE", 1)
    detection_batch_size: int = _config_value(config_object, "MEDIA_PIPELINE.DETECTION_BATCH_SIZE", 1)
    missing_dependency_policy: str = _config_value(config_object, "MEDIA_PIPELINE.MISSING_DEPENDENCY_POLICY", "skip")
    timestamp_match_strategy: str = _config_value(config_object, "MEDIA_PIPELINE.TIMESTAMP_MATCH_STRATEGY", "inclusive_interval_match")
    context_window: int = _config_value(config_object, "MEDIA_PIPELINE.CONTEXT_WINDOW", 1)
    context_text_joiner: str = _config_value(config_object, "MEDIA_PIPELINE.CONTEXT_TEXT_JOINER", " ")
    timestamp_tolerance: Optional[float] = _config_value(config_object, "MEDIA_PIPELINE.TIMESTAMP_TOLERANCE", None)
    canonical_schema_version: str = _config_value(config_object, "MEDIA_PIPELINE.CANONICAL_SCHEMA_VERSION", "v1")
    merge_strategy: str = _config_value(config_object, "MEDIA_PIPELINE.MERGE_STRATEGY", "merge_by_frame_id")
    conflict_policy: str = _config_value(config_object, "MEDIA_PIPELINE.CONFLICT_POLICY", "prefer_enrichment")
    language_normalization_policy: str = _config_value(config_object, "MEDIA_PIPELINE.LANGUAGE_NORMALIZATION_POLICY", "preserve")

class MediaModelConfig(BaseModel):
    """Media model and inference settings."""
    temporal_model_name: str = _config_value(config_models, "MEDIA_MODELS.TEMPORAL_MODEL_NAME", "TransNetV2")
    temporal_checkpoint_name: str = _config_value(config_models, "MEDIA_MODELS.TEMPORAL_CHECKPOINT_NAME", "transnetv2-pytorch-weights.pth")
    temporal_checkpoint_dir: str = _config_value(config_models, "MEDIA_MODELS.TEMPORAL_CHECKPOINT_DIR", "data/models")
    temporal_scene_detector_enabled: bool = _config_value(config_models, "MEDIA_MODELS.TEMPORAL_SCENE_DETECTOR_ENABLED", True)
    temporal_device_policy: str = _config_value(config_models, "MEDIA_MODELS.TEMPORAL_DEVICE_POLICY", "auto")
    frame_resize_width: int = _config_value(config_models, "MEDIA_MODELS.FRAME_RESIZE_WIDTH", 48)
    frame_resize_height: int = _config_value(config_models, "MEDIA_MODELS.FRAME_RESIZE_HEIGHT", 27)
    window_length: int = _config_value(config_models, "MEDIA_MODELS.WINDOW_LENGTH", 100)
    window_stride: int = _config_value(config_models, "MEDIA_MODELS.WINDOW_STRIDE", 50)
    leading_padding_frames: int = _config_value(config_models, "MEDIA_MODELS.LEADING_PADDING_FRAMES", 25)
    center_slice_start: int = _config_value(config_models, "MEDIA_MODELS.CENTER_SLICE_START", 25)
    center_slice_end: int = _config_value(config_models, "MEDIA_MODELS.CENTER_SLICE_END", 75)
    shot_boundary_threshold: float = _config_value(config_models, "MEDIA_MODELS.SHOT_BOUNDARY_THRESHOLD", 0.3)
    duplicate_threshold: float = _config_value(config_models, "MEDIA_MODELS.DUPLICATE_THRESHOLD", 0.90)
    clip_model_name: str = _config_value(config_models, "MEDIA_MODELS.CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    similarity_metric: str = _config_value(config_models, "MEDIA_MODELS.SIMILARITY_METRIC", "cosine")
    ocr_languages: list[str] = _config_value(config_models, "MEDIA_MODELS.OCR_LANGUAGES", ["vi", "en"])
    ocr_version: str = _config_value(config_models, "MEDIA_MODELS.OCR_VERSION", "PP-OCRv5")
    ocr_detection_model_name: str = _config_value(config_models, "MEDIA_MODELS.OCR_DETECTION_MODEL_NAME", "PP-OCRv5_server_det")
    ocr_recognition_model_name: str = _config_value(config_models, "MEDIA_MODELS.OCR_RECOGNITION_MODEL_NAME", "latin_PP-OCRv5_mobile_rec")
    ocr_use_doc_orientation_classify: bool = _config_value(config_models, "MEDIA_MODELS.OCR_USE_DOC_ORIENTATION_CLASSIFY", False)
    ocr_use_doc_unwarping: bool = _config_value(config_models, "MEDIA_MODELS.OCR_USE_DOC_UNWARPING", False)
    ocr_use_textline_orientation: bool = _config_value(config_models, "MEDIA_MODELS.OCR_USE_TEXTLINE_ORIENTATION", True)
    ocr_gpu: bool = _config_value(config_models, "MEDIA_MODELS.OCR_GPU", False)
    caption_model_name: str = _config_value(config_models, "MEDIA_MODELS.CAPTION_MODEL_NAME", "microsoft/Florence-2-base-ft")
    caption_model_version: str = _config_value(config_models, "MEDIA_MODELS.CAPTION_MODEL_VERSION", "base-ft")
    caption_trust_remote_code: bool = _config_value(config_models, "MEDIA_MODELS.CAPTION_TRUST_REMOTE_CODE", True)
    caption_max_new_tokens: int = _config_value(config_models, "MEDIA_MODELS.CAPTION_MAX_NEW_TOKENS", 100)
    caption_device: str = _config_value(config_models, "MEDIA_MODELS.CAPTION_DEVICE", "cpu")
    caption_dtype: str = _config_value(config_models, "MEDIA_MODELS.CAPTION_DTYPE", "auto")
    detection_model_name: str = _config_value(config_models, "MEDIA_MODELS.DETECTION_MODEL_NAME", "yolo11n")
    detection_weight_file: str = _config_value(config_models, "MEDIA_MODELS.DETECTION_WEIGHT_FILE", "yolo11n.pt")
    detection_confidence_threshold: float = _config_value(config_models, "MEDIA_MODELS.DETECTION_CONFIDENCE_THRESHOLD", 0.5)
    detection_device_policy: str = _config_value(config_models, "MEDIA_MODELS.DETECTION_DEVICE_POLICY", "runtime_default")
    detection_verbose: bool = _config_value(config_models, "MEDIA_MODELS.DETECTION_VERBOSE", False)
    whisper_model_size: str = _config_value(config_models, "MEDIA_MODELS.WHISPER_MODEL_SIZE", "large-v3")
    whisper_device: str = _config_value(config_models, "MEDIA_MODELS.WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = _config_value(config_models, "MEDIA_MODELS.WHISPER_COMPUTE_TYPE", "int8")
    whisper_beam_size: int = _config_value(config_models, "MEDIA_MODELS.WHISPER_BEAM_SIZE", 5)
    whisper_temperature: float | list[float] = _config_value(config_models, "MEDIA_MODELS.WHISPER_TEMPERATURE", 0.0)
    whisper_condition_on_previous_text: bool = _config_value(config_models, "MEDIA_MODELS.WHISPER_CONDITION_ON_PREVIOUS_TEXT", False)
    whisper_compression_ratio_threshold: Optional[float] = _config_value(config_models, "MEDIA_MODELS.WHISPER_COMPRESSION_RATIO_THRESHOLD", 2.4)
    whisper_log_prob_threshold: Optional[float] = _config_value(config_models, "MEDIA_MODELS.WHISPER_LOG_PROB_THRESHOLD", -1.0)
    whisper_no_speech_threshold: Optional[float] = _config_value(config_models, "MEDIA_MODELS.WHISPER_NO_SPEECH_THRESHOLD", 0.6)
    whisper_no_repeat_ngram_size: int = _config_value(config_models, "MEDIA_MODELS.WHISPER_NO_REPEAT_NGRAM_SIZE", 3)
    whisper_repetition_penalty: float = _config_value(config_models, "MEDIA_MODELS.WHISPER_REPETITION_PENALTY", 1.1)
    whisper_max_new_tokens: Optional[int] = _config_value(config_models, "MEDIA_MODELS.WHISPER_MAX_NEW_TOKENS", 96)
    whisper_vad_filter: bool = _config_value(config_models, "MEDIA_MODELS.WHISPER_VAD_FILTER", True)
    whisper_vad_threshold: float = _config_value(config_models, "MEDIA_MODELS.WHISPER_VAD_THRESHOLD", 0.5)
    whisper_vad_min_silence_duration_ms: int = _config_value(config_models, "MEDIA_MODELS.WHISPER_VAD_MIN_SILENCE_DURATION_MS", 700)
    whisper_vad_speech_pad_ms: int = _config_value(config_models, "MEDIA_MODELS.WHISPER_VAD_SPEECH_PAD_MS", 300)
    asr_language: Optional[str] = _config_value(config_models, "MEDIA_MODELS.ASR_LANGUAGE", None)

class MediaPromptConfig(BaseModel):
    """Media prompt settings."""
    caption_prompt: str = _config_value(config_prompts, "MEDIA_PROMPTS.CAPTION_PROMPT", "<MORE_DETAILED_CAPTION>")
    caption_prompt_version: str = _config_value(config_prompts, "MEDIA_PROMPTS.CAPTION_PROMPT_VERSION", "v1")

class LLMConfig(BaseModel):
    """The configurable fields for the model llm."""
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    model_name: Optional[str] = config_models.OPENAI_LLM_MODEL.MODEL_PATH
    timeout: Optional[float] = 30
    max_retries: Optional[int] = 3
    base_url: Optional[str] = None


class AppConfig(BaseModel):
    """Top-level application configuration."""
    qdrant: Optional[QdrantConfig] = QdrantConfig()
    embedding: Optional[EmbeddingConfig] = EmbeddingConfig()
    retrieval: Optional[RetrievalConfig] = RetrievalConfig()
    minio: Optional[MinioConfig] = MinioConfig()
    pipeline: Optional[PipelineConfig] = PipelineConfig()
    media_pipeline: Optional[MediaPipelineConfig] = MediaPipelineConfig()
    media_models: Optional[MediaModelConfig] = MediaModelConfig()
    media_prompts: Optional[MediaPromptConfig] = MediaPromptConfig()
    llm : Optional[LLMConfig] = LLMConfig()
