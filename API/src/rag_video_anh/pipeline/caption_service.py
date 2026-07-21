"""Image captioning component."""

from __future__ import annotations

from typing import Any

from src.configuration import AppConfig
from src.log.logger import logger
from src.rag_video_anh.schemas import CaptionRequest, CaptionResult, CaptionResultSet, StageStatus


class CaptionService:
    """Generate semantic captions for keyframes."""

    def __init__(
        self,
        config: AppConfig | None = None,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.pipeline_config = self.config.media_pipeline
        self.model_config = self.config.media_models
        self.prompt_config = self.config.media_prompts
        self.processor = processor
        self.model = model
        self._runtime_loaded = processor is not None and model is not None
        self._load_error: str | None = None

    def caption(self, request: CaptionRequest) -> CaptionResultSet:
        if not request.keyframes.frames:
            return CaptionResultSet(
                media_id=request.media_id,
                status=StageStatus.SKIPPED,
                reason="no keyframes available for captioning",
            )

        runtime = self._load_runtime()
        if runtime is None:
            return CaptionResultSet(
                media_id=request.media_id,
                status=StageStatus.SKIPPED,
                reason=self._load_error or "missing dependency 'transformers'",
            )

        results: list[CaptionResult] = []
        for frame in request.keyframes.frames:
            try:
                caption_text = self._caption_frame(frame.image_payload if frame.image_payload is not None else frame.image_path)
                results.append(
                    CaptionResult(
                        frame_id=frame.frame_id,
                        caption_text=caption_text,
                        generation_meta={
                            "model": self.model_config.caption_model_name,
                            "max_new_tokens": self.model_config.caption_max_new_tokens,
                        },
                        prompt_version=self.prompt_config.caption_prompt_version,
                        status=StageStatus.DONE if caption_text else StageStatus.NOT_FOUND,
                        reason=None if caption_text else "empty caption",
                    )
                )
            except Exception as exc:
                results.append(
                    CaptionResult(
                        frame_id=frame.frame_id,
                        prompt_version=self.prompt_config.caption_prompt_version,
                        status=StageStatus.ERROR,
                        reason=f"caption generation failed: {exc.__class__.__name__}",
                    )
                )

        status = StageStatus.DONE if all(result.status != StageStatus.ERROR for result in results) else StageStatus.ERROR
        logger.info(f"Generated caption results for {len(results)} frame(s) in media_id '{request.media_id}'")
        return CaptionResultSet(media_id=request.media_id, results=results, status=status)

    def _load_runtime(self) -> tuple[Any, Any] | None:
        if self._runtime_loaded:
            return self.processor, self.model
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            self._runtime_loaded = True
            self._load_error = "missing dependency 'transformers'"
            return None

        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_config.caption_model_name,
                trust_remote_code=self.model_config.caption_trust_remote_code,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_config.caption_model_name,
                trust_remote_code=self.model_config.caption_trust_remote_code,
            )
            if hasattr(self.model, "to"):
                self.model = self.model.to(self.model_config.caption_device)
            self._runtime_loaded = True
            return self.processor, self.model
        except Exception as exc:
            self._runtime_loaded = True
            self._load_error = f"caption model load failed: {exc.__class__.__name__}"
            return None

    def _caption_frame(self, image: Any) -> str:
        if image is None:
            raise ValueError("keyframe image is not available")
        if hasattr(self.model, "caption"):
            return str(self.model.caption(image)).strip()

        processor, model = self.processor, self.model
        prompt = self.prompt_config.caption_prompt
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model_config.caption_device)
        generated_ids = model.generate(
            input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            max_new_tokens=self.model_config.caption_max_new_tokens,
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        if hasattr(processor, "post_process_generation"):
            processed = processor.post_process_generation(generated_text, task=prompt)
            if isinstance(processed, dict):
                generated_text = next(iter(processed.values()), generated_text)
        return str(generated_text).strip()
