import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from .recognize import HFRecognize

@click.command()
@ocrd_cli_options
def ocrd_hf_recognize(*args, **kwargs):
    """
    Run Hugging Face (Transformer) OCR line recognition on PAGE-XML TextLines.
    """
    return ocrd_cli_wrap_processor(HFRecognize, *args, **kwargs)
