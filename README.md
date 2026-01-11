# ocrd_hf

OCR-D processor for **Hugging Face** Transformer OCR models.

- Input: PAGE-XML with `TextRegion/TextLine`
- Output: same PAGE-XML with `TextEquiv` (line-level)
- Backends: `microsoft/trocr-*` and most `AutoModelForVision2Seq` models

## Install

```commandline
pip install -e .
```

Or install via Docker:
```commandline
docker compose build
docker-compose run ocrd-hf
```
For CPU only:
```commandline
docker compose build ocrd-hf-cpu
docker-compose run ocrd-hf-cpu
```       
The *Docker* deployment is **still in testing**. 
## Usage
### Basic Usage
```
ocrd-hf-recognize -I OCR-D-TABLE-LINE -O OCR-D-TEXT-HF -P device cpu -P model "microsoft/trocr-base-handwritten"
```
## Remark
ChatGPT 4.5 as well as Claude Opus 4.5 have been used to generate this OCR-D extension.
At the beginning, the kraken extension was used as an example: [github.com/OCR-D/ocrd_kraken](https://github.com/OCR-D/ocrd_kraken)
Generated code has been manually tested and iteratively improved using the listed models.
