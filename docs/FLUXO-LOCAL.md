# Imagem → OCR → HTML

Na pasta do projeto, ative o ambiente e processe uma nota:

```bash
source .venv/bin/activate
python main.py ocr --limit 1
python main.py html
```

Abra o arquivo terminado em `_html.html` dentro de `output/html/output-sintetico/`.
Os JSONs, recortes e logs ficam em `output/intermediarios/output-sintetico/`.
Para processar todas as notas, troque `--limit 1` por `--limit 0`.

O OCR usa português, prioriza capturas de 300 dpi e reconstrói células pelas
bordas da nota. A geração padrão de HTML dispensa Ollama.

## Usar suas próprias notas

Coloque os arquivos em `input/minhas-notas/`:

```bash
python main.py ocr --input input/minhas-notas --output output/minhas-notas --limit 0
python main.py html --output output/minhas-notas
```

Os HTMLs ficam em `output/html/minhas-notas/`; os intermediários, em
`output/intermediarios/minhas-notas/`. Confira os valores no documento gerado.

## Conferir a amostra

```bash
python tests/validate_dataset.py
```

Os relatórios ficam em `output/intermediarios/output-sintetico/`.

Os resultados anteriores estão guardados em `output/outputs antigos/`.
As próximas execuções usam normalmente `output/html/` e `output/intermediarios/`.

[HTML da nota já testada](<../output/outputs antigos/html/ocr-melhoria/final/300/01-1-cinza-300/p01_png/p01_page_001_html.html>)

## HTML com LLM (OpenRouter)

Com a chave em `OPENROUTER_API_KEY` ou no `.env`:

```bash
python main.py html --engine openrouter --model qwen/qwen3.8-27b:free
```

O raciocínio do modelo fica desligado: ligado (`--reasoning`), o Qwen3.8 gastou
5–16 mil tokens pensando e truncou o HTML. Respostas truncadas ou vazias são rejeitadas.

## Métricas

Cada execução imprime, por etapa, tempo, CPU, pico de RAM, pico de VRAM (só com
Paddle em GPU), tokens e custo, e salva em `runs/metrics_<timestamp>.json`.

## Notebook e Colab

`notebooks/pipeline_openrouter.ipynb` roda OCR → HTML via OpenRouter com as métricas.
No Colab: crie o secret `OPENROUTER_API_KEY`, suba o notebook e envie o `colab.zip`
(`pipeline.py` + uma imagem) quando pedido. Se o Colab pedir, reinicie a sessão após a
instalação e rode tudo de novo. A última célula baixa `output/` e `runs/` num zip.

## Código

`pipeline.py` tem OCR, tabelas, HTML (direto, Ollama ou OpenRouter) e métricas;
`main.py` é só a linha de comando. O registro de OCR guarda apenas status e recorte.
Testes: `python -m unittest discover tests`.

[Alterações e medições anteriores](otimizacao.md)
