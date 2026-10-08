● Fiz as três etapas, sem nenhum commit. Os 31 testes passam e o notebook roda do início ao fim. Medido, o pipeline não ficou mais rápido: o que ainda tomava tempo dentro do nosso código
  era o predict do PaddleOCR, e aumentar threads só piorou.

  Antes e depois

  OCR de 3 notas a 300 dpi, com a máquina parada:

  ┌──────────────────────────┬──────────┬──────────────────────────────┐
  │          Etapa           │  Antes   │            Depois            │
  ├──────────────────────────┼──────────┼──────────────────────────────┤
  │ Carregar o modelo        │ 11,5 s   │ 11,8 s                       │
  ├──────────────────────────┼──────────┼──────────────────────────────┤
  │ ocr:predict              │ 270,0 s  │ 260,8 s (repetição: 265,0 s) │
  ├──────────────────────────┼──────────┼──────────────────────────────┤
  │ Reparar tabelas          │ 0,46 s   │ 0,42 s                       │
  ├──────────────────────────┼──────────┼──────────────────────────────┤
  │ Salvar saídas            │ 0,57 s   │ 0,63 s                       │
  ├──────────────────────────┼──────────┼──────────────────────────────┤
  │ HTML direto (9 arquivos) │ 0,15 s   │ 0,14 s                       │
  ├──────────────────────────┼──────────┼──────────────────────────────┤
  │ Pico de RAM              │ ~1,95 GB │ ~1,96 GB                     │
  └──────────────────────────┴──────────┴──────────────────────────────┘

  - O predict do PaddleOCR é mais de 99% do tempo. O resto soma menos de 2 s, então não havia ganho possível no nosso código.
  - Mais threads deixou mais lento:

  | Threads | predict |
  |---------|---------|
  | 2       | 261 s   |
  | 3       | 331 s   |
  | 4       | 358 s   |
  | 6       | 376 s   |
    O i7-1355U só tem 2 núcleos de desempenho; os outros 8 são de eficiência. O padrão continua em 2, agora configurável com --cpu-threads.
  - O MKLDNN quebra no Paddle 3.3.1 (ConvertPirAttribute2RuntimeAttribute not support). Continua desligado, com um comentário no código explicando.
  - O comportamento não mudou. Os HTMLs e JSONs saem idênticos byte a byte. O relatório do validate_dataset.py também saiu idêntico ao da versão anterior.
  - GPU: a máquina não tem GPU NVIDIA, então essas colunas ficam vazias, sem erro.

  O que mudou

  - Métricas (pipeline/metrics.py): medem tempo, CPU, pico de RAM, GPU e tokens/custo por etapa. No fim de cada execução imprimem a tabela e salvam em runs/metrics_<timestamp>.json.
  - Pacote pipeline/:
    - ocr.py reúne o código de OCR que estava em main.py com a reconstrução das tabelas que estava em tests/generate_html.py.
    - render.py gera o HTML direto, via Ollama ou via OpenRouter (cliente OpenAI). O comando agora é python -m pipeline.render --engine openrouter.
    - O main.py caiu de 143 para 65 linhas e só cuida dos argumentos.
  - Linhas no total: o código do pipeline, sem contar as métricas, ficou com 568 linhas contra 535 antes, porque agora inclui o engine da OpenRouter. O ganho foi em organização e remoção
    de duplicação, não em tamanho.
  - Código removido: três funções mortas e uma duplicada em validate_dataset.py. Os arquivos mortos (tests/test.py, tests/or_gen_html.py, qwen-html/) e o generate_html.py antigo foram
    para arquivos_mortos/, que está no .gitignore.
  - Testes: criei tests/test_pipeline.py com 6 testes novos (métricas, simplify_result e OpenRouter com cliente simulado). Agora todos rodam com python -m unittest discover tests.
  - Notebook notebooks/pipeline_openrouter.ipynb: 9 células, modelo configurável na primeira, só importando funções do pacote. Na execução final, a chamada ao qwen/qwen3.8-27b:free levou
    29 s, com 2165 tokens de entrada, 4409 de saída e custo $0.
  - O docs/FLUXO.md foi atualizado com os comandos novos.

  Precisa da sua decisão

  1. Raciocínio do modelo desligado por padrão. Ligado, o Qwen3.8 gastou de 5 a 16 mil tokens só pensando e truncou o HTML duas vezes, inclusive com limite de 16384. Desligado, gerou 4–5
     mil tokens e o HTML passou na validação estrita. Dá para religar com --reasoning no CLI ou REASONING = True no notebook.
  2. Bug antigo em _overlap (pipeline/ocr.py): aparece RuntimeWarning: overflow encountered in scalar multiply. Pelo aviso, as caixas do OCR parecem vir num tipo inteiro pequeno e a
     multiplicação das áreas estoura, o que pode afetar a associação de textos às células. Não confirmei o tipo exato. Corrigir muda a saída das tabelas, então não mexi. Quer que eu
     corrija?
  3. Para o OCR ficar realmente mais rápido, é preciso mudar o comportamento. As opções são:
     - desligar a classificação de orientação;
     - reduzir --det-max-side;
     - usar o modelo de detecção mobile no lugar do server;
     - usar GPU.

     O scripts/benchmark_ocr.py já mede a qualidade de algumas dessas variações. Quer que eu rode?
  4. Arquivos locais:
     - runs/ (26 MB de medições) e output/notebook/ não estão no .gitignore.
     - O requirements.txt está em UTF-16 e não inclui nbconvert/ipykernel, que instalei no .venv.
  5. Troque a chave da OpenRouter. Ela foi colada no chat; localmente está só no .env, que é ignorado pelo git.
  6. Plano gratuito: ele devolve 429 (limite de taxa) com frequência. Coloquei 5 tentativas automáticas no cliente, mas em lotes grandes pode não bastar.