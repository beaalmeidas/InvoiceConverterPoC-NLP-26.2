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
Um zip também serve: `python main.py ocr --input notas.zip --limit 0` extrai o zip em
`output/entradas/notas/` e processa as notas (subpastas incluídas).

O OCR usa português, prioriza capturas de 300 dpi e reconstrói células pelas
bordas da nota. A orientação de linha fica desligada (`--textline-orientation` liga):
ligada, ela girava linhas das notas em paisagem e o OCR lia lixo. A geração padrão
de HTML (`--engine direct`) não usa LLM.

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
python scripts/validate_dataset.py
```

Os relatórios ficam em `output/intermediarios/output-sintetico/`.

Os resultados anteriores estão guardados em `output/outputs antigos/`.
As próximas execuções usam normalmente `output/html/` e `output/intermediarios/`.

[HTML da nota já testada](<../output/outputs antigos/html/ocr-melhoria/final/300/01-1-cinza-300/p01_png/p01_page_001_html.html>)

## HTML com LLM (Ollama ou OpenRouter)

**Ollama** (sem custo): com o servidor rodando (`OLLAMA_CONTEXT_LENGTH=32768 ollama serve`)
e o modelo baixado (`ollama pull qwen2.5vl:7b`):

```bash
python main.py html --engine ollama
```

O contexto padrão do Ollama é pequeno e corta o JSON da nota e a imagem sem avisar.

**OpenRouter** (pago): com a chave em `OPENROUTER_API_KEY` ou no `.env`:

```bash
python main.py html --engine openrouter --model qwen/qwen3.8-27b
```

O Qwen3.8 é pago (US$ 0,20 por milhão de tokens de entrada e US$ 2,50 de saída); a versão
`:free` saiu da OpenRouter. O Qwen2.5-VL lá só existe em 72B, também pago. O custo de cada
chamada aparece nas métricas.

O raciocínio do modelo fica desligado: ligado (`--reasoning`, só OpenRouter), o Qwen3.8 gastou
5–16 mil tokens pensando e truncou o HTML. Respostas truncadas ou vazias são rejeitadas.

## Métricas

Cada execução imprime, por etapa, tempo, CPU, pico de RAM, pico de VRAM (só com
Paddle em GPU), tokens e custo, e salva em `runs/metrics_<timestamp>.json`.

## Notebook e Colab

`notebooks/pipeline_openrouter.ipynb` roda OCR → HTML via LLM, HTML direto da imagem (VLM)
e compara o tempo de cada caminho. O notebook tem o código da pipeline inteiro. No Colab, com
`ENGINE = "ollama"` (padrão), use o ambiente T4: o notebook instala o Ollama e baixa o
`qwen2.5vl:7b`, sem chave nem custo. Com `ENGINE = "openrouter"`, crie o secret `OPENROUTER_API_KEY`.
Envie as notas quando pedido: várias imagens de uma vez ou um `.zip`. O tempo de cada nota
fica em `runs/tempos_por_nota.csv`. Se o Colab pedir, reinicie a sessão após a instalação e
rode tudo de novo. A última célula baixa `output/` e `runs/` num zip.

## Código

`pipeline.py` tem OCR, tabelas, HTML (direto, Ollama ou OpenRouter), VLM (`ask_vlm`) e métricas;
`main.py` é só a linha de comando. O registro de OCR (`ocr_run.json`) guarda status e tempo.
Testes: `python -m unittest discover tests`.

Avaliação contra os gabaritos do dataset sintético: `python scripts/avaliar.py`
(relatório em `output/avaliacao/`). Para gerar antes as 5 entradas com um mesmo modelo, inclusive
a imagem direto no VLM: `python scripts/avaliar.py --gerar qwen2.5vl:7b --engine ollama`.
O relatório traz o tempo total de cada caminho (OCR + LLM contra VLM); notas com OCR feito
antes desta versão não têm o tempo do OCR registrado.

[Alterações e medições anteriores](otimizacao.md)
