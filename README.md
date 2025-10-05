# ocrd_hf

OCR-D processor for **Hugging Face** Transformer OCR models.

- Input: PAGE-XML with `TextRegion/TextLine`
- Output: same PAGE-XML with `TextEquiv` (line-level)
- Backends: `microsoft/trocr-*` and most `AutoModelForVision2Seq` models

## Install

```bash
pip install -e .
