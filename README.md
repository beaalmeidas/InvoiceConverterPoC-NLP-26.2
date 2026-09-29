# Invoice Converter

PoC de conversão de notas fiscais: **imagem → PaddleOCR → JSON → Qwen2.5-VL:7b → HTML**.
O OCR roda localmente e a geração com Qwen via Ollama roda no Google Colab.

## Instalação e extração local

Use Linux e Python 3.12. Execute na raiz do projeto:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

O `main.py` lê `input/img_1.jpg` e grava `output/img_1_res.json`.
Também exporta Markdown, HTML de tabelas, imagens auxiliares e layout simplificado.
Para trocar a entrada, edite o caminho de `pipeline.predict` em `main.py`;
o script não possui argumentos de linha de comando.
Uma nova extração pode sobrescrever saídas de mesmo nome.

## Geração de HTML

Abra [o notebook Qwen](notebooks/comparacao_preprocessamento.ipynb) no Colab com GPU.
Execute as células em ordem e envie a imagem e o JSON completo da mesma nota.
A configuração padrão é:

```python
MODEL = "qwen2.5vl:7b"
PREPROCESS = "com"
INPUT_FORMAT = "json"
OUTPUT_FORMAT = "html"
```

O notebook inclui os módulos necessários e instala suas dependências no Colab.
Ele compacta o JSON em seções e linhas de tabelas e gera HTML com o modelo.
Essa compactação não preserva explicitamente rowspan/colspan.
Baixe o ZIP com o HTML, o manifesto e o JSON de origem.
Confira os dados contra a imagem: OCR e geração podem alterar ou omitir campos.

## Organização e versionamento

- `main.py`: extração local.
- `notebooks/comparacao_preprocessamento.ipynb`: geração com Qwen.
- `notebooks/NLP_26_2_openrouter_test.ipynb`: notebook preservado para continuidade dos experimentos.
- `input/`: entradas preservadas para reprodução.
- `output/`: todos os resultados, incluindo experimentos anteriores e ZIPs.
- `docs/FLUXO.md`: passo a passo.
- `docs/RESUMO_EXPERIMENTOS.md`: registro histórico dos experimentos.

Entradas e outputs são destinados ao versionamento. Nenhum output foi removido
na limpeza do projeto. Saídas de Gemma, Qwen3 e outras abordagens permanecem como
evidências históricas; o fluxo atual usa Qwen2.5-VL:7b.
Ambientes virtuais, caches e configuração local continuam ignorados.
O guia antigo em `output/README.md` foi preservado com os artefatos; suas menções
a resultados ignorados e ao gerador de DANFEs não descrevem o fluxo atual.
