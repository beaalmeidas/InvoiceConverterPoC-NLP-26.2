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

## Semana 3 — 28/09 em diante

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
