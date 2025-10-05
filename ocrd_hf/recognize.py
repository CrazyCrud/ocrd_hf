from __future__ import annotations
from typing import Optional, List
from PIL import Image
import torch

from ocrd import Processor, OcrdPage, OcrdPageResult
from ocrd_utils import VERSION as OCRD_VERSION
from ocrd_models.ocrd_page import TextEquivType

from .adapters import build_adapter


class HFRecognize(Processor):
    """
    OCR-D processor: recognize text on TextLine elements using HuggingFace models.
    """

    @property
    def executable(self) -> str:
        return "ocrd-hf-recognize"

    def show_version(self):
        import transformers
        print(
            f"Version {self.version}, "
            f"transformers {transformers.__version__}, "
            f"ocrd/core {OCRD_VERSION}"
        )

    def setup(self):
        """
        Initialize backend once per run. Called before processing.
        """
        model_id = self.parameter.get("model")
        if not model_id:
            raise ValueError("Missing required parameter: 'model'")
        
        try:
            model_path = self.resolve_resource(model_id)
        except Exception:
            # Fallback: treat it as a HF identifier
            model_path = model_id

        device = self.parameter.get("device", "cpu")
        fp16 = bool(self.parameter.get("fp16", False))
        max_new_tokens = self.parameter.get("max_new_tokens", None)
        self.batch_size = int(self.parameter.get("batch_size", 8))

        self.adapter = build_adapter(
            model_path,
            device=device,
            fp16=fp16,
            max_new_tokens=max_new_tokens
        )

        # Feature selection for image cropping: leave blank to allow full color/gray input
        self.features: str = ""

    def shutdown(self):
        """If adapter needs cleanup (optional)."""
        # Example: if adapter holds GPU memory or threads, free them here.
        pass

    def process_page_pcgts(
        self,
        *input_pcgts: Optional[OcrdPage],
        page_id: Optional[str] = None
    ) -> OcrdPageResult:
        """
        Recognize text for all TextLines on a page.
        """
        pcgts = input_pcgts[0]
        if pcgts is None:
            return OcrdPageResult(None)

        page = pcgts.get_Page()

        # Load page image & coordinates once
        page_image, page_coords, _ = self.workspace.image_from_page(
            page, page_id, feature_selector=self.features
        )

        # Find all text regions
        regions = page.get_AllRegions(classes=["Text"])
        total_lines = sum(len(r.get_TextLine() or []) for r in regions)
        if total_lines == 0:
            self.logger.warning("No TextLine elements on page '%s'", page_id)
            return OcrdPageResult(pcgts)

        # For each region: crop region-image, then crop lines, batch predict, write
        for region in regions:
            lines = region.get_TextLine() or []
            if not lines:
                continue

            # Crop region-level image and coords
            region_image, region_coords = self.workspace.image_from_segment(
                region, page_image, page_coords, feature_selector=self.features
            )

            batch_imgs: List[Image.Image] = []
            batch_line_objs: List = []

            for line in lines:
                try:
                    line_image, _ = self.workspace.image_from_segment(
                        line, region_image, region_coords, feature_selector=self.features
                    )
                except Exception as e:
                    self.logger.error(
                        "Could not crop line '%s' in region '%s': %s",
                        getattr(line, "id", "?"), getattr(region, "id", "?"), e
                    )
                    continue

                # Skip too tiny lines
                w, h = line_image.size
                if w < 2 or h < 2:
                    self.logger.warning(
                        "Skipping tiny/invalid line '%s' in region '%s' (%dx%d)",
                        getattr(line, "id", "?"), getattr(region, "id", "?"), w, h
                    )
                    continue

                batch_imgs.append(line_image.convert("RGB"))
                batch_line_objs.append(line)

                # Flush when enough
                if len(batch_imgs) >= self.batch_size:
                    self._predict_and_write(batch_imgs, batch_line_objs)
                    batch_imgs = []
                    batch_line_objs = []

            # Flush leftover
            if batch_imgs:
                self._predict_and_write(batch_imgs, batch_line_objs)

        return OcrdPageResult(pcgts)

    def _predict_and_write(
        self,
        images: List[Image.Image],
        lines: List
    ):
        """Run adapter and write back to TextLine elements."""
        try:
            pixel_values = self.adapter.preprocess(images)
            outputs = self.adapter.generate(pixel_values)
            texts = [t.strip() for t in self.adapter.decode(outputs)]
        except Exception as e:
            self.logger.error(
                "HF inference failed on batch size %d: %s", len(images), e
            )
            return

        for line, txt in zip(lines, texts):
            if line.get_TextEquiv():
                line.set_TextEquiv([])  # clear any existing
            line.add_TextEquiv(TextEquivType(Unicode=txt))
