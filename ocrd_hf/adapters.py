from __future__ import annotations
from typing import List, Sequence, Dict, Optional
from PIL import Image
import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoModelForVision2Seq,
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

class BaseHFAdapter:
    """
    Abstract interface for HF OCR backends.
    """
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        fp16: bool = False,
        max_new_tokens: Optional[int] = None,
        quantize_kwargs: Optional[Dict] = None,
        gen_kwargs: Optional[Dict] = None
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.fp16 = fp16
        self.max_new_tokens = max_new_tokens
        self.quantize_kwargs = quantize_kwargs or {}
        self.gen_kwargs = gen_kwargs or {}

    def preprocess(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Convert PIL images to model input tensor."""
        raise NotImplementedError

    @torch.inference_mode()
    def generate(self, pixel_values: torch.Tensor) -> torch.LongTensor:
        """Generate model outputs (token ids)."""
        raise NotImplementedError

    def decode(self, outputs: torch.LongTensor) -> List[str]:
        """Convert model outputs to Unicode strings."""
        raise NotImplementedError


class TrOCRAdapter(BaseHFAdapter):
    """
    Adapter for TrOCR models (encoder-decoder) using TrOCRProcessor + VisionEncoderDecoderModel.
    """
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        fp16: bool = False,
        max_new_tokens: Optional[int] = None,
        quantize_kwargs: Optional[Dict] = None,
        gen_kwargs: Optional[Dict] = None
    ):
        super().__init__(model_id, device, fp16, max_new_tokens, quantize_kwargs, gen_kwargs)
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_id,
            **(quantize_kwargs or {})
        )
        self.model.to(self.device).eval()
        if self.fp16 and self.device.type == "cuda":
            self.model.half()

    def preprocess(self, images: Sequence[Image.Image]) -> torch.Tensor:
        images = [im.convert("RGB") for im in images]
        batch = self.processor(images=images, return_tensors="pt")
        return batch.pixel_values.to(self.device)

    @torch.inference_mode()
    def generate(self, pixel_values: torch.Tensor) -> torch.LongTensor:
        # Combine default gen_kwargs and max_new_tokens
        kwargs = dict(self.gen_kwargs)
        if self.max_new_tokens is not None:
            kwargs["max_new_tokens"] = self.max_new_tokens
        return self.model.generate(pixel_values, **kwargs)

    def decode(self, outputs: torch.LongTensor) -> List[str]:
        return self.processor.batch_decode(outputs, skip_special_tokens=True)


class AutoV2SAdapter(BaseHFAdapter):
    """
    Generic vision-to-sequence adapter using AutoProcessor + AutoModelForVision2Seq.
    """
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        fp16: bool = False,
        max_new_tokens: Optional[int] = None,
        quantize_kwargs: Optional[Dict] = None,
        gen_kwargs: Optional[Dict] = None
    ):
        super().__init__(model_id, device, fp16, max_new_tokens, quantize_kwargs, gen_kwargs)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            **(quantize_kwargs or {})
        )
        self.model.to(self.device).eval()
        if self.fp16 and self.device.type == "cuda":
            self.model.half()

    def preprocess(self, images: Sequence[Image.Image]) -> torch.Tensor:
        images = [im.convert("RGB") for im in images]
        batch = self.processor(images=images, return_tensors="pt")
        pixel_values = batch.get("pixel_values")
        if pixel_values is None:
            raise ValueError(f"Processor {self.processor} did not return 'pixel_values'")
        return pixel_values.to(self.device)

    @torch.inference_mode()
    def generate(self, pixel_values: torch.Tensor) -> torch.LongTensor:
        kwargs = dict(self.gen_kwargs)
        if self.max_new_tokens is not None:
            kwargs["max_new_tokens"] = self.max_new_tokens
        return self.model.generate(pixel_values, **kwargs)

    def decode(self, outputs: torch.LongTensor) -> List[str]:
        return self.processor.batch_decode(outputs, skip_special_tokens=True)


def build_adapter(
    model_id: str,
    device: str = "cpu",
    fp16: bool = False,
    max_new_tokens: Optional[int] = None,
    quantize_kwargs: Optional[Dict] = None,
    gen_kwargs: Optional[Dict] = None
) -> BaseHFAdapter:
    """
    Factory: pick adapter based on model config or name.

    - If config.model_type suggests vision-encoder-decoder / trocr, use TrOCRAdapter
    - Else fallback to AutoV2SAdapter
    """
    cfg = AutoConfig.from_pretrained(model_id)
    model_type = getattr(cfg, "model_type", "")
    # Some TrOCR models have model_type "vision-encoder-decoder" or "trocr"
    if model_type in {"trocr", "vision-encoder-decoder"}:
        try:
            return TrOCRAdapter(
                model_id, device=device, fp16=fp16,
                max_new_tokens=max_new_tokens,
                quantize_kwargs=quantize_kwargs,
                gen_kwargs=gen_kwargs
            )
        except Exception as e:
            # fallback with warning
            print(f"[Warning] TrOCRAdapter init failed for '{model_id}': {e}")
    # Fallback
    return AutoV2SAdapter(
        model_id, device=device, fp16=fp16,
        max_new_tokens=max_new_tokens,
        quantize_kwargs=quantize_kwargs,
        gen_kwargs=gen_kwargs
    )
