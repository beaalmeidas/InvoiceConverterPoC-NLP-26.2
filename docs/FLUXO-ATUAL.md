Resumo: o projeto transforma imagens de notas fiscais em HTML. Primeiro roda OCR, depois gera o HTML. Li docs/FLUXO.md, pipeline.py e docs/MUDANCAS.md e não rodei nada.

Fluxo atual

1. OCR (python main.py ocr)
- Seleciona as imagens em input/, priorizando as de 300 dpi.
- Amplia a imagem até 1600 px de largura, se ela for menor.
- Roda o PaddleOCR (PPStructureV3, português, PP-OCRv5). O MKLDNN fica desligado porque quebra no Paddle 3.3.1. A correção de deformação (unwarp) também está desligada.
- Reconstrói as tabelas pelas bordas visíveis (detect_cells e ruled_table, com OpenCV). Se a grade não for confiável, mantém a tabela do Paddle.
- Salva JSON, Markdown e um _layout.json compacto em output/intermediarios/<nome>/.

2. HTML (python main.py html)
Há três motores:
- direct (padrão): monta o HTML direto do OCR, sem LLM. Leva cerca de 0,15 s.
- ollama: usa qwen2.5vl:7b localmente.
- openrouter: usa qwen/qwen3.8-27b:free. O raciocínio fica desligado, porque ligado ele truncava o HTML. A API tem 5 tentativas automáticas.

O HTML final vai para output/html/<nome>/. Se já havia um HTML, ele vira .previous.html, e o log fica em .generation.json.

3. Métricas
Cada etapa (via stage()) mede tempo,
- Reorganização: o pacote pipeline/ virou o arquivo único pipeline.py, e o main.py ficou só com a linha de comando. Os arquivos mortos (tests/test.py, qwen-html/ etc.) foram deletados.
- Medição e novos arquivos: o foco é medir e otimizar, com scripts/, runs/, docs/otimizacao.md, docs/RESUMO_EXPERIMENTOS.md, tests/test_pipeline.py e uma amostra em input/amostra_diversa/.

Pontos em aberto

Eles estão em docs/MUDANCAS.md, mas esse texto descreve a estrutura antiga com pipeline/ocr.py, então pode estar defasado.
- Gargalo: o predict do PaddleOCR é mais de 99% do tempo, cerca de 260 s para 3 notas. Mais threads piora, porque o i7-1355U só tem 2 núcleos de desempenho.
- Bug em _overlap: há um overflow de inteiro nas áreas das caixas. Isso pode afetar a associação de texto às células, e ninguém corrigiu ainda.
- Chave da OpenRouter: foi colada no chat e precisa ser trocada.
- Limpeza: runs/ não está no .gitignore, e o requirements.txt está em UTF-16.

Quer que eu corrija o _overlap ou atualize o MUDANCAS.md?