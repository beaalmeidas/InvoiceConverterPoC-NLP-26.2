# Notebooks

`comparacao_preprocessamento.ipynb` é o notebook principal, com Qwen2.5-VL:7b
via Ollama no Colab. Recebe o JSON completo gerado pelo PaddleOCR local,
compacta o conteúdo documental e solicita HTML.

Padrões: `PREPROCESS = "com"`, `INPUT_FORMAT = "json"`, `OUTPUT_FORMAT = "html"`.
As opções anteriores de comparação ainda estão disponíveis na configuração.
O notebook contém os módulos necessários e instala as dependências no Colab.

`NLP_26_2_openrouter_test.ipynb` permanece disponível para continuidade dos
experimentos. O notebook Gemma e as cópias da pasta `historico/` foram removidos;
todos os outputs foram preservados.
Veja o [fluxo atual](../docs/FLUXO.md).
