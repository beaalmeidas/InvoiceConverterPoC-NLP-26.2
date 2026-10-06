### Fluxo Google Colab

Atualmente, a pipeline é realizada inteiramente através do google colab, uma vez que, naquele ambiente, os arquivos são simplificados, a execução é sequencial e, com a GPU T4 do colab, a velocidade de execução é bem mais veloz do que se fosse realizada localmente.

O notebook usado é o `notebooks/pipeline_openrouter.ipynb`, que chama as funções anteriormente presentes no `pipeline.py`.

O pipeline funciona da seguinte maneira:

-> Antes de rodar, a chave `OPENROUTER_API_KEY` é guardada em Secrets do colab
-> O projeto é empacotado em um `colab.zip` e enviado quando o notebook pedir
-> Na primeira célula, são definidos modelo, imagem de entrada, tipo de entrada do LLM (`compact`, `markdown` ou `full_json`) e a pasta de saída
-> A segunda célula extrai o zip em `/content/projeto`, lê a chave e instala as dependências (PaddleOCR, PaddlePaddle, PaddleX, openai, etc.)
-> OCR: o modelo do PaddleOCR é carregado e a imagem é processada, gerando JSON, Markdown e um `_layout.json` compacto
-> HTML: o `_layout.json` é enviado para o LLM na OpenRouter (`qwen/qwen3.8-27b:free`, com o raciocínio desligado), que devolve o HTML da nota
-> O HTML gerado é exibido dentro do próprio notebook
-> Resumo: é mostrada a tabela com o tempo de cada etapa
-> Por fim, `output/` e `runs/` são zipados em `colab_resultados.zip` e baixados para a máquina local

OBS:

- O raciocínio do qwen fica desligado porque, ligado, ele gasta de 5 a 16 mil tokens pensando e pode truncar o HTML.
- A sessão do colab é temporária, então o que não for baixado no último passo se perde quando ela fechar.
