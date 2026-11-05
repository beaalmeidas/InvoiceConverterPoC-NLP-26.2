# InvoiceConverterPoC-NLP-26.2

OCR de notas fiscais (PaddleOCR) → HTML, direto ou via LLM (Ollama ou OpenRouter), com métricas por etapa.
O HTML sai com marcadores (`data-field`, `data-section`, `data-item`, `data-index`) para o contrato de campos.
Detalhes do fluxo em [`docs/FLUXO.md`](docs/FLUXO.md).

## Notebook (uso principal)

A pipeline roda em `notebooks/pipeline_openrouter.ipynb`, que tem todo o código dentro dele: não precisa do `pipeline.py` nem de zip.
Ele faz OCR → HTML via LLM, HTML direto da imagem (VLM) e compara o tempo de cada caminho.
Versões antigas ficam em `notebooks/historico/` (`v1` a `v4`), só como registro.

### No Google Colab

Com `ENGINE = "ollama"` (padrão) roda 100% no Colab, sem custo: o `qwen2.5vl:7b` fica na GPU.

1. Escolha o ambiente com GPU T4 (Ambiente de execução → Alterar o tipo) e suba o notebook.
2. Rode as células e envie as notas quando a célula de preparo pedir: várias imagens de uma vez ou um `.zip` (pode ter subpastas). A célula do Ollama instala o servidor e baixa o modelo (~6 GB) a cada sessão. Se o Colab pedir, reinicie a sessão após a instalação e rode tudo de novo.
3. Para cada nota, o notebook gera o HTML pelo OCR + LLM e direto da imagem (VLM). O tempo de cada nota fica em `runs/tempos_por_nota.csv`; o OCR leva ~2 min por nota na CPU.
4. Para usar a OpenRouter (pago), troque `ENGINE = "openrouter"` e crie o secret `OPENROUTER_API_KEY` (🔑).
5. A última célula baixa `output/` e `runs/` num `colab_resultados.zip`. A sessão do Colab é temporária: o que não for baixado se perde.

O raciocínio do modelo fica desligado (`REASONING = False`, só OpenRouter): ligado, o Qwen3.8 gasta de 5 a 16 mil tokens pensando e pode truncar o HTML.

### Localmente

```bash
pip install -r requirements.txt
cp .env.example .env   # coloque sua chave; só para o engine openrouter (pago)
```

Depois abra o notebook no Jupyter.

## Linha de comando (`pipeline.py` e `main.py`)

`pipeline.py` é a cópia do código do notebook, usada por `main.py`, pelos testes e por `scripts/avaliar.py`.
Ela tem OCR, tabelas, HTML (direto, Ollama ou OpenRouter), VLM (`ask_vlm`), marcadores e métricas;
`main.py` é só a linha de comando (`python main.py ocr` e `python main.py html`).
**O código está duplicado**: ao mudar um (por exemplo, o prompt), copie a mudança para o outro.

Testes: `python -m unittest discover tests`.

Avaliação contra os gabaritos do dataset sintético: `python scripts/avaliar.py`
(relatório em `output/avaliacao/`). Para gerar antes as 5 entradas com um mesmo modelo, inclusive
a imagem direto no VLM: `python scripts/avaliar.py --gerar qwen2.5vl:7b --engine ollama`.
O relatório traz o tempo total de cada caminho (OCR + LLM contra VLM); notas com OCR feito
antes desta versão não têm o tempo do OCR registrado.

## Documentação

- [`docs/FLUXO.md`](docs/FLUXO.md): fluxo da pipeline
- [`docs/RESUMO_SEMANAL.md`](docs/RESUMO_SEMANAL.md): o que foi feito a cada semana
- [`docs/otimizacao.md`](docs/otimizacao.md): alterações e medições anteriores
