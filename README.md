# InvoiceConverterPoC-NLP-26.2

OCR de notas fiscais (PaddleOCR) → HTML, direto ou via LLM (Ollama ou OpenRouter), com métricas por etapa.
Detalhes do fluxo em [`docs/FLUXO.md`](docs/FLUXO.md).

## Rodar localmente

```bash
pip install -r requirements.txt
cp .env.example .env   # coloque sua chave; só para o engine openrouter (pago)
```

Depois abra `notebooks/pipeline_openrouter.ipynb` ou use `python main.py`.

## Rodar no Google Colab

O notebook tem todo o código da pipeline; não precisa do `pipeline.py` nem de zip. Com
`ENGINE = "ollama"` (padrão) roda 100% no Colab, sem custo: o `qwen2.5vl:7b` fica na GPU.

1. No Colab, escolha o ambiente com GPU T4 (Ambiente de execução → Alterar o tipo) e suba `notebooks/pipeline_openrouter.ipynb`.
2. Rode as células e envie as notas quando a célula de preparo pedir: várias imagens de uma vez ou um `.zip` (pode ter subpastas). A célula do Ollama instala o servidor e baixa o modelo (~6 GB) a cada sessão.
3. Para cada nota, o notebook gera o HTML pelo OCR + LLM e direto da imagem (VLM), e compara o tempo dos caminhos (`runs/tempos_por_nota.csv`). O OCR leva ~2 min por nota na CPU.
4. Para usar a OpenRouter (pago), troque `ENGINE = "openrouter"` e crie o secret `OPENROUTER_API_KEY` (🔑).
5. Se mudar o `pipeline.py` (por exemplo, o prompt), copie a mudança também para a célula correspondente do notebook.
6. A última célula baixa `output/` e `runs/` num `colab_resultados.zip`. A sessão do Colab é temporária: o que não for baixado se perde.

O raciocínio do modelo fica desligado (`REASONING = False`, só OpenRouter): ligado, o Qwen3.8 gasta de 5 a 16 mil tokens pensando e pode truncar o HTML.
