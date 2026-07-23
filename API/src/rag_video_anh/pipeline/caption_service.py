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
                            "device": self.model_config.caption_device,
                            "dtype": getattr(self.model_config, "caption_dtype", "auto"),
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
                        generation_meta={"model": self.model_config.caption_model_name},
                        prompt_version=self.prompt_config.caption_prompt_version,
                        status=StageStatus.ERROR,
                        reason=f"caption generation failed: {exc.__class__.__name__}: {exc}",
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
            try:
                from transformers import AutoModelForMultimodalLM
            except ImportError:
                AutoModelForMultimodalLM = None
        except ImportError:
            self._runtime_loaded = True
            self._load_error = "missing dependency 'transformers'"
            return None

        try:
            torch_dtype = self._torch_dtype()
            self.processor = AutoProcessor.from_pretrained(
                self.model_config.caption_model_name,
                trust_remote_code=self.model_config.caption_trust_remote_code,
            )
            model_kwargs = {"trust_remote_code": self.model_config.caption_trust_remote_code}
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            model_class = AutoModelForMultimodalLM or AutoModelForCausalLM
            self.model = model_class.from_pretrained(
                self.model_config.caption_model_name,
                **model_kwargs,
            )
            if hasattr(self.model, "eval"):
                self.model = self.model.eval()
            if hasattr(self.model, "to"):
                self.model = self.model.to(self.model_config.caption_device)
            self._runtime_loaded = True
            return self.processor, self.model
        except Exception as exc:
            self._runtime_loaded = True
            self._load_error = f"caption model load failed: {exc.__class__.__name__}: {exc}"
            return None

    def _caption_frame(self, image: Any) -> str:
        if image is None:
            raise ValueError("keyframe image is not available")
        image = self._normalize_image(image)
        if hasattr(self.model, "caption"):
            return str(self.model.caption(image)).strip()

        processor, model = self.processor, self.model
        prompt = self.prompt_config.caption_prompt
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = self._move_inputs_to_device(inputs)
        generated_ids = self._generate(model, inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        if hasattr(processor, "post_process_generation"):
            generated_text = self._post_process_generation(processor, generated_text, prompt, image)
        return str(generated_text).strip()

    def _move_inputs_to_device(self, inputs: Any) -> Any:
        if not hasattr(inputs, "to"):
            return inputs
        torch_dtype = self._torch_dtype()
        if torch_dtype is None:
            return inputs.to(self.model_config.caption_device)
        return inputs.to(self.model_config.caption_device, torch_dtype)

    def _generate(self, model: Any, inputs: Any) -> Any:
        generation_kwargs = {
            "input_ids": inputs.get("input_ids"),
            "pixel_values": inputs.get("pixel_values"),
            "max_new_tokens": self.model_config.caption_max_new_tokens,
        }
        try:
            import torch
        except ImportError:
            return model.generate(**generation_kwargs)
        with torch.no_grad():
            return model.generate(**generation_kwargs)

    @staticmethod
    def _post_process_generation(processor: Any, generated_text: str, prompt: str, image: Any) -> str:
        try:
            processed = processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(image.width, image.height),
            )
        except TypeError:
            processed = processor.post_process_generation(generated_text, task=prompt)
        if isinstance(processed, dict):
            return str(processed.get(prompt) or next(iter(processed.values()), generated_text)).strip()
        return str(processed).strip()

    def _torch_dtype(self) -> Any:
        dtype_name = str(getattr(self.model_config, "caption_dtype", "auto") or "auto").strip().lower()
        if dtype_name in {"", "auto", "none"}:
            return None
        try:
            import torch
        except ImportError:
            return None
        return {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype_name)

    @staticmethod
    def _normalize_image(image: Any) -> Any:
        if isinstance(image, str):
            try:
                from PIL import Image
            except ImportError:
                return image
            return Image.open(image).convert("RGB")

        try:
            import cv2
            import numpy as np
            from PIL import Image
        except ImportError:
            return image

        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return Image.fromarray(image).convert("RGB")
            return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image
