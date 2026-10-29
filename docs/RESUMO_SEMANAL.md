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

## Semana 5 — 19 a 25/10

Foi feito:

- Criei um script (`scripts/avaliar.py`) que compara os HTMLs gerados com os gabaritos do dataset sintético.
- Testei se o LLM ajuda a montar o HTML. Na prática não ajudou: o HTML gerado direto do OCR, sem LLM, saiu igual, e o LLM ainda demora uns 30 a 60 s por nota. Numa das notas ele inventou rótulos que não estavam no OCR.
- Comparei as três entradas do LLM (JSON compacto, JSON completo e Markdown). Deram resultados bem parecidos; o Markdown perde um pouco de texto.
- A reconstrução das tabelas pelas bordas fez bastante diferença: o acerto das células praticamente dobrou e os produtos passaram a sair certos em boa parte das notas.
- A maioria dos erros vem do OCR. Nas notas em paisagem, ele girava algumas linhas por engano e lia lixo no lugar do destinatário. Desliguei essa opção no notebook e os campos da nota em paisagem testada subiram de 66,7% para 88,9%.
- Tentei acelerar o OCR com modelos mais leves, mas a leitura das tabelas piorou, então mantive os atuais.
- Coloquei todo o código da pipeline dentro do notebook do Colab. Agora não precisa mais do `colab.zip`, só enviar a imagem.
- Enxuguei o `pipeline.py`, tirando o Ollama e opções que ninguém usava.
- Organizei o repositório: apaguei arquivos soltos e docs repetidos, movi o `validate_dataset.py` para `scripts/` e atualizei o `.gitignore`.
- Não consegui testar o modelo lendo a imagem direto: o Qwen gratuito saiu do OpenRouter e o Gemma gratuito estava sem cota.

## Semana 6 — 25/10 a 28/10

Foi feito (Larissa):

- Desliguei a orientação de linha do OCR por padrão também no `pipeline.py` e no `main.py`, igual ao notebook.
- O `qwen/qwen3.8-27b` gratuito saiu do OpenRouter e a versão paga virou o padrão. Para não pagar, a pipeline voltou a aceitar o Ollama (`--engine ollama`) com o `qwen2.5vl:7b`, que roda na GPU T4 do Colab sem custo.
- Criei o `ask_vlm`, que gera o HTML direto da imagem, sem OCR (imagem reduzida a 1600 px no lado maior e sempre em PNG).
- O notebook ganhou a célula do Ollama, o passo do VLM e a tabela de tempo por nota (`runs/tempos_por_nota.csv`). Ele agora processa várias notas de uma vez (imagens ou `.zip`), e o `ocr` também aceita `.zip`.
- Rodei no Colab as 19 notas do lote diverso (100 a 300 dpi) com o `qwen2.5vl:7b`, lendo o texto do OCR (LLM) e lendo a imagem (VLM). A primeira tentativa deu erro em todas porque o Ollama estava sem o `llama-server`; corrigi e rodei de novo.
- Acerto nos campos da nota: OCR direto 91,0%, OCR + LLM 91,1% e VLM 94,7%. O LLM não melhorou nada; o VLM acertou mais.
- O VLM respondeu 14 das 19 notas e o LLM 18 de 19. As outras passaram do limite de 8192 tokens (`num_predict`), o HTML ficou pela metade e foi descartado. Cada tentativa gastou ~4,5 minutos à toa. Em 4 das 5 notas cortadas no VLM a nota é pequena, então o modelo provavelmente ficou repetindo texto (não conferi). O corte não parece ligado ao DPI.
- Tempo médio por nota: OCR direto 71 s, OCR + LLM 130 s e VLM 65 s. O VLM é mais rápido por não precisar do OCR, mas as tentativas cortadas fazem o total dele passar o do OCR + LLM.
- Subi o limite do notebook para 16384 tokens, mas ainda não testei de novo. Se for repetição, só aumentar o limite pode não resolver. Relatório completo em `relatorio-larissa.md`.

Foi feito (Guilherme):

- Montou a própria versão do notebook (`guilherme_pipeline_ipynb.ipynb`) com o modelo gratuito `nvidia/nemotron-3-ultra-550b-a55b:free` no OpenRouter.
- Colocou um contrato de campos (`contrato_campos.csv`) no prompt: cada valor da nota sai marcado no HTML com `data-field`, e o notebook confere quais campos do contrato ficaram sem marcação.
- Comparou as mesmas 19 notas nos três caminhos (`relatorio-guilherme.md`): OCR direto 79,0%, OCR + LLM 88,1% e VLM 96,1% (o VLM só respondeu 15 das 19). Ele usa outro conjunto de campos, então os percentuais não são comparáveis com os meus, mas a conclusão é a mesma: o VLM acerta mais, mas falha em algumas notas.
- Tempo médio por nota: OCR direto 71 s, OCR + LLM 184 s e VLM 124 s.
