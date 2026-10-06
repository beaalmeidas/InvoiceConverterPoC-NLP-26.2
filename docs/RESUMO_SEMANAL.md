# Resumo semanal

## Semana 1 — 14 a 20/09

Foi feito:

- Criada a base do projeto.
- Adicionadas imagens e um PDF de DANFE para experimentação.
- Criada a primeira extração com PaddleOCR e scripts iniciais para gerar HTML.

## Semana 2 — 21 a 27/09

Foi feito:

- Criado JSON simplificado com o conteúdo e a posição dos blocos do OCR.
- Mantido o HTML das tabelas extraídas e reduzido o uso de CPU.
- Criado notebook para experimentar a geração de HTML a partir de Markdown e JSON.
- Guardados exemplos de OCR e de HTML gerado pelo Qwen para comparação.
- Integrada a branch de refatoração.

## Semana 3 — 28/09 a 01/10

Foi feito:

- Organizados os resultados dos experimentos e adicionada uma segunda imagem de DANFE.
- Criado notebook para comparar diferentes fluxos de processamento.
- Documentados o projeto, os notebooks e o fluxo de OCR para HTML.
- Ajustados parâmetros do PaddleOCR para testar orientação e correção da imagem.
- Adaptado o prompt para gerar HTML a partir do JSON compacto.
- Adição de imagens sintéticas criadas por Guilherme e Humberto no treinamento, para trazer maior variedade nos dados.
- Leitura de várias imagens e PDFs e geração de um JSON e um HTML por página.
- Testando Qwen2.5vl:7b localmente e com geração múltipla de HTMLs comparativos com 3 diferentes intermediários: JSON compacto, JSON completo e Markdown
- Arquivo `validate_dataset.py` para comparar o HTML gerado com o "gabarito" presente no dataset sintético a fim de apresentar diferenças no output.
    - *_html.html: JSON compacto
    - *from_full_json_html.html: JSON completo do PaddleOCR.
    - *_from_md_html.html: markdown do PaddleOCR.
- Configurado OCR em português, com prioridade para imagens de 300 dpi e ampliação das imagens pequenas.
- Limitada a resolução da detecção de texto para reduzir o consumo de memória na CPU.
- Reconstruídas células e mesclagens pelas bordas visíveis das tabelas e pela posição dos textos reconhecidos.
- Separados os HTMLs em `output/html/` e os JSONs, logs e relatórios em `output/intermediarios/`.
- Guardados os resultados anteriores em `output/outputs antigos/` para comparação com novas execuções.
- Processadas 5 imagens de 300 dpi e geradas 15 versões de HTML. Código, quantidade, preço e total dos produtos passaram na conferência em 4 das 5 notas.
- Adicionados testes para conferir a reconstrução das células, a preservação dos valores e a separação das saídas.

## Semana 4 — 02/10 em diante

Foi feito (notebook `pipeline_openrouter.ipynb`):

- Criado notebook que roda toda a pipeline (OCR com PaddleOCR → HTML via OpenRouter) no Google Colab.
- A ideia era deixar a pipeline inteira só no notebook, por causa do tempo e dos recursos da máquina local (o OCR na CPU leva ~90 s por nota).
- HTML gerado pela OpenRouter com `qwen/qwen3.8-27b:free`; raciocínio do modelo desligado por padrão, porque ligado gastava muitos tokens e truncava o HTML.
- Parâmetros no topo do notebook: modelo, imagem, tipo de entrada do LLM (compact, markdown ou full_json), pasta de saída e threads de CPU.
- No Colab, a chave vem dos Secrets e o código do projeto é enviado em `colab.zip`.
- Métricas de tempo, CPU, RAM e tokens por etapa, com resumo no final.
- Com 1 nota no Colab: `predict` 131 s, HTML pela OpenRouter 30 s, custo $0.
- Última célula empacota `output/` e `runs/` num zip para baixar os resultados.
