# InvoiceConverterPoC-NLP-26.2

OCR de notas fiscais (PaddleOCR) → HTML, direto ou via LLM (Ollama / OpenRouter), com métricas por etapa.
Detalhes do fluxo em [`docs/FLUXO.md`](docs/FLUXO.md).

## Rodar localmente

```bash
pip install -r requirements.txt
echo "OPENROUTER_API_KEY=sua_chave" > .env   # só para o engine openrouter
```

Depois abra `notebooks/pipeline_openrouter.ipynb` ou use `python main.py`.

## Rodar no Google Colab

1. Gere o `colab.zip` na raiz do projeto (ele não vai para o GitHub):

   ```bash
   zip colab.zip pipeline.py input/amostra_diversa/captura/300/01-1-cinza-300/p01.png
   ```

   Gere de novo sempre que alterar o `pipeline.py` (por exemplo, o prompt), senão o Colab roda a versão antiga.

2. No Colab, crie o secret `OPENROUTER_API_KEY` (🔑) e suba `notebooks/pipeline_openrouter.ipynb`.
3. Rode a primeira célula de código e envie o `colab.zip` quando ela pedir.
4. A última célula baixa `output/` e `runs/` num `colab_resultados.zip`.
