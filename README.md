# InvoiceConverterPoC-NLP-26.2

**English** · [Português (BR)](README.pt-BR.md)

OCR of invoices (PaddleOCR) → HTML, directly or via an LLM (Ollama or OpenRouter), with per-stage metrics.
The HTML comes out with markers (`data-field`, `data-section`, `data-item`, `data-index`) for the field contract.
The invoices are Brazilian (DANFE/NFC-e), so the generated HTML stays in Portuguese; code, comments and messages are in English.
Details of the flow are in [`docs/FLUXO.md`](docs/FLUXO.md) (in Portuguese).

## Notebook (main usage)

The pipeline runs in `notebooks/pipeline_openrouter.ipynb`, which has all the code inside it: no `pipeline.py` or zip needed.
It does OCR → HTML via LLM, HTML straight from the image (VLM), and compares the time of each path.
Old versions live in `notebooks/historico/` (`v1` to `v4`), kept only as a record.

### On Google Colab

With `ENGINE = "ollama"` (default) everything runs on Colab at no cost: `qwen2.5vl:7b` sits on the GPU.

1. Pick the T4 GPU runtime (Runtime → Change runtime type) and upload the notebook.
2. Run the cells and upload the invoices when the setup cell asks: several images at once or a `.zip` (subfolders allowed). The Ollama cell installs the server and downloads the model (~6 GB) every session. If Colab asks, restart the session after the install and run everything again.
3. For each invoice, the notebook generates the HTML through OCR + LLM and straight from the image (VLM). The time per invoice is saved in `runs/tempos_por_nota.csv`; OCR takes ~2 min per invoice on the CPU.
4. To use OpenRouter (paid), set `ENGINE = "openrouter"` and create the `OPENROUTER_API_KEY` secret (🔑).
5. The last cell downloads `output/` and `runs/` as `colab_resultados.zip`. A Colab session is temporary: whatever is not downloaded is lost.

Model reasoning is off (`REASONING = False`, OpenRouter only): when on, Qwen3.8 spends 5 to 16 thousand tokens thinking and may truncate the HTML.

### Locally

```bash
pip install -r requirements.txt
cp .env.example .env   # put your key here; only for the openrouter engine (paid)
```

Then open the notebook in Jupyter.

## Command line (`pipeline.py` and `main.py`)

`pipeline.py` is a copy of the notebook code, used by `main.py`, the tests and `scripts/avaliar.py`.
It has OCR, tables, HTML (direct, Ollama or OpenRouter), VLM (`ask_vlm`), markers and metrics;
`main.py` is only the command line (`python main.py ocr` and `python main.py html`).
**The code is duplicated**: when you change one (for example, the prompt), copy the change to the other.

Tests: `python -m unittest discover tests`.

Evaluation against the synthetic dataset ground truth: `python scripts/avaliar.py`
(report in `output/avaliacao/`). To first generate the 5 inputs with the same model, including
the image straight into the VLM: `python scripts/avaliar.py --gerar qwen2.5vl:7b --engine ollama`.
The report gives the total time of each path (OCR + LLM versus VLM); invoices whose OCR was run
before this version have no OCR time recorded.

## Documentation

These documents are kept in their original language (Portuguese):

- [`docs/FLUXO.md`](docs/FLUXO.md): pipeline flow
- [`docs/RESUMO_SEMANAL.md`](docs/RESUMO_SEMANAL.md): what was done each week
- [`docs/otimizacao.md`](docs/otimizacao.md): earlier changes and measurements
