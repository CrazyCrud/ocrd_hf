# ocrd_hf

OCR-D processor for **Hugging Face** Transformer OCR models.

- Input: PAGE-XML with `TextRegion/TextLine`
- Output: same PAGE-XML with `TextEquiv` (line-level)
- Backends: `microsoft/trocr-*` and most `AutoModelForVision2Seq` models

## Install

```commandline
pip install -e .
```
Or install via Dockerhub:
```commandline
docker compose build
docker-compose run ocrd-hf
```
For CPU only:
```commandline
docker compose build ocrd-hf-cpu
docker-compose run ocrd-hf-cpu
```   

## Usage
### Basic Usage
```
ocrd-hf-recognize -I OCR-D-TABLE-LINE -O OCR-D-TEXT-HF -P device cpu -P model "microsoft/trocr-base-handwritten"
```


