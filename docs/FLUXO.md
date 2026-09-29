# Fluxo atual

```text
input/img_1.jpg
    → main.py (PaddleOCR local)
    → output/img_1_res.json
    → notebooks/comparacao_preprocessamento.ipynb (Qwen2.5-VL:7b no Colab)
    → com_preprocessamento.html + manifest.json
```

1. Instale as dependências conforme o [README](../README.md).
2. Execute `python main.py` na raiz do projeto.
3. Abra o notebook no Colab com GPU e execute as células em ordem.
4. Envie a imagem original e o `*_res.json` correspondente.
5. Mantenha `MODEL = "qwen2.5vl:7b"`, `PREPROCESS = "com"`,
   `INPUT_FORMAT = "json"` e `OUTPUT_FORMAT = "html"`.
6. Baixe o ZIP e salve em `output/comparacoes/` para versionamento.
7. Confira o HTML contra a imagem original e consulte o manifesto para falhas.

O PaddleOCR não roda no Colab. O notebook inclui seu próprio código de validação,
compactação e geração, sem depender de `invoice_formats.py` na raiz.
O JSON compacto conserva textos e linhas de tabelas, mas não os atributos
originais de mesclagem de células (`rowspan`/`colspan`).

O ZIP padrão contém o HTML quando a geração termina com sucesso, `manifest.json`
e `ocr/nota_res.json`. O manifesto registra parâmetros, status e tempos de geração.

O `main.py` também exporta Markdown e HTML de tabelas como artefatos auxiliares.
Esses resultados e todos os outputs anteriores permanecem no projeto para
versionamento. O HTML de tabelas exportado pelo Paddle não substitui a geração
da nota completa pelo Qwen.
