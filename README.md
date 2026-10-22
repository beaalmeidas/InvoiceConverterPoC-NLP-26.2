# InvoiceConverterPoC-NLP-26.2

OCR de notas fiscais (PaddleOCR) → HTML, direto ou via LLM (OpenRouter), com métricas por etapa.
Detalhes do fluxo em [`docs/FLUXO.md`](docs/FLUXO.md).

## Rodar localmente

```bash
pip install -r requirements.txt
cp .env.example .env   # coloque sua chave; só para o engine openrouter
```

Depois abra `notebooks/pipeline_openrouter.ipynb` ou use `python main.py`.

## Rodar no Google Colab

O notebook tem todo o código da pipeline; não precisa do `pipeline.py` nem de zip.

1. No Colab, crie o secret `OPENROUTER_API_KEY` (🔑) e suba `notebooks/pipeline_openrouter.ipynb`.
2. Rode as células e envie a imagem da nota quando a célula de preparo pedir.
3. Se mudar o `pipeline.py` (por exemplo, o prompt), copie a mudança também para a célula correspondente do notebook.
4. A última célula baixa `output/` e `runs/` num `colab_resultados.zip`. A sessão do Colab é temporária: o que não for baixado se perde.

O raciocínio do modelo fica desligado (`REASONING = False`): ligado, o Qwen gasta de 5 a 16 mil tokens pensando e pode truncar o HTML.
