**Resumo do projeto e dos experimentos — Invoice Converter**

> Atualização de organização (28/09/2026): o fluxo mantido é PaddleOCR → JSON
> compacto → Qwen2.5-VL:7b → HTML. Os scripts de experimentação, o gerador de
> DANFEs, o módulo avulso `invoice_formats.py`, o notebook Gemma e as cópias
> da pasta `historico/` foram retirados. O notebook
> `notebooks/NLP_26_2_openrouter_test.ipynb` foi restaurado e permanece disponível. Todos os outputs permanecem preservados e são versionáveis.
> O texto abaixo registra o levantamento anterior; referências a esses códigos
> e a diretórios ignorados descrevem aquele estado, não a estrutura atual.
> O resumo em TXT foi consolidado neste documento. O erro registrado no HTML
> direto foi `prediction aborted, token repeat limit reached (status code: 500)`,
> e não timeout. Consulte o [fluxo atual](FLUXO.md) para executar o projeto.

Levantamento dos códigos, notebooks e resultados disponíveis no repositório em 25/09/2026. O objetivo da PoC é transformar imagens de notas fiscais em documentos digitais legíveis, preservando campos, valores e organização da DANFE.

**A escolha central é usar JSON do OCR como representação intermediária e HTML como saída de apresentação: imagem → PaddleOCR → JSON com conteúdo e estrutura → HTML.** O JSON permite selecionar e rastrear o conteúdo extraído; o HTML permite reconstruir tabelas e organizar a nota para leitura e impressão. Os testes também mostram que enviar o JSON bruto inteiro ao modelo não basta: é necessário selecionar o conteúdo documental e preservar a estrutura das tabelas.

**O que foi desenvolvido**

- Extração local com PaddleOCR/PPStructureV3, com exportação de JSON completo, Markdown, HTML de tabelas e blocos simplificados contendo tipo, posição e conteúdo.
- Ajustes para execução local: duas threads de CPU, MKLDNN desativado e reconhecimento de fórmulas, gráficos e selos desligado. O reconhecimento de tabelas permanece ativo. São ajustes de configuração; não há benchmark salvo que quantifique seu ganho.
- Conversão para uma representação compacta em `invoice_formats.py`, com seções, tabelas e recuperação de textos do OCR que não aparecem nos blocos principais.
- Experimentos de geração via Ollama local, Ollama no Colab e script de acesso ao OpenRouter.
- Notebook de comparação entre imagem direta e dados previamente extraídos pelo OCR, com saídas HTML e Markdown, parâmetros registrados, tratamento de falhas e download em ZIP.
- Notebook com Gemma3:4b e chamadas independentes JSON → HTML e JSON → Markdown.
- Gerador de DANFEs sintéticos em PDF e XML para ampliar os testes com documentos fictícios.
- Organização de scripts, notebooks históricos, resultados e documentação, preservando os artefatos anteriores.

**Diferenças entre os caminhos investigados**

| Caminho | O que o modelo recebe e faz | Principal diferença |
|---|---|---|
| Imagem → modelo → HTML/Markdown | Lê a imagem e produz o documento na mesma chamada | Concentra leitura, interpretação e formatação no modelo visual |
| Imagem → OCR → Markdown → HTML | Recebe a representação exportada pelo OCR | Simples de inspecionar, mas depende da estrutura efetivamente contida no Markdown |
| Imagem → OCR → JSON completo → modelo | Recebe conteúdo, coordenadas e muitos metadados | Maior volume de entrada; os resultados mostram confusão entre dados da nota e coordenadas |
| Imagem → OCR → JSON compacto → modelo | Recebe seções, linhas de tabelas e textos selecionados | Reduz a entrada, mas a transformação em linhas pode perder a informação explícita de células mescladas |
| Imagem → OCR → JSON com HTML das tabelas → HTML | Recebe os blocos documentais e o HTML reconhecido pelo OCR | Mantém `rowspan` e `colspan` disponíveis para reconstruir a DANFE |

No notebook comparativo, “com pré-processamento” significa usar OCR antes do modelo; “sem” significa enviar a imagem diretamente. Isso é diferente de corrigir orientação e deformação da imagem: essas correções estão desligadas no `main.py` atual. Não foi encontrado um par de resultados que permita medir o efeito de ligá-las e desligá-las.

O arquivo `output/img_1.md` já contém uma tabela HTML. Portanto, o experimento histórico Markdown → HTML não equivale a uma comparação entre texto Markdown simples e JSON estruturado.

**Inventário dos testes e resultados encontrados**

| Experimento | Evidência disponível | Resultado observado / alcance da evidência |
|---|---|---|
| Extração com PaddleOCR | `output/img_1_res.json`, `img_1.md`, `img_1_table_1.html`, `img_0_layout.json` | Foram produzidos texto e estrutura tabular; a extração ainda contém erros de OCR |
| Extrações e validação locais adicionais | `output/extracao/off/pagina_001/` e `output/validacao/off/pagina_001/` | Há artefatos de extração, inclusive da segunda imagem; a existência da pasta `validacao` não comprova acurácia dos campos |
| Qwen2.5:7b, Markdown → HTML | Notebook histórico e `output/qwen-html/img_1_rep.html` | Produziu tabela HTML; o arquivo salvo ainda contém delimitadores de código Markdown |
| Qwen2.5:7b, JSON de blocos → HTML | Notebook histórico e `output/qwen-html/nota_fiscal_json.html` | Produziu documento HTML com tabela e CSS; há alterações de grafia em relação ao OCR, portanto não é cópia literal garantida |
| Prompt para preservar DANFE e impressão | `notebooks/historico/og_NLP_26_2_openrouter_test.ipynb` | Foi implementado reforço para preservar células, mesclagens e CSS A4; sem medição isolada de ganho |
| Gemma, arquivos identificados como JSON → HTML/Markdown | `output/gemma-html/img_1_json.html` e `output/gemma-md/img_1_json.md` | Responderam com explicações sobre dados numéricos/sensores, em vez de reconstruir a nota; o arquivo `.html` não contém documento HTML |
| Gemma, arquivos identificados como JSON compacto → HTML/Markdown | `output/gemma-html/img_1_compact.html` e `output/gemma-md/img_1_compact.md` | Passaram a apresentar dados da nota, mas o HTML usa parágrafos e “Table 1”, insere placeholder e termina incompleto; o Markdown também termina no meio de um campo |
| Outros HTMLs Gemma preservados | `output/gemma-html/img_1_rep.html` e `img_2_rep.html` | O primeiro termina no meio de uma célula; o segundo contém tabela e delimitadores Markdown. Não há manifesto que vincule cada arquivo aos parâmetros exatos da geração |
| Gemma3:4b, JSON com conteúdo HTML dos blocos | `notebooks/NLP_Gemma3_4b_JSON_HTML_MD.ipynb` | Implementação voltada à preservação das tabelas; sem saídas de execução salvas no notebook que comprovem o resultado da versão atual |
| Experimentos identificados como Qwen3 | `output/qwen3-html/img_1_res_qwen3.html` e `img_1_compact_qwen3.html` | O primeiro contém apenas “bounding boxes”; o segundo contém tabela parcial, reticências e textos genéricos. Não há registro suficiente para afirmar a versão exata do modelo |
| Qwen2.5-VL:7b, imagem versus OCR | Manifesto da execução de 24/09/2026, detalhado abaixo | Três gerações concluídas e uma falha; há tempos por chamada |
| Conexão e Markdown → HTML com Ollama | `scripts/experimentos/check_ollama.py` e `generate_html.py`, configurados com `qwen3.5:9b` | Scripts manuais disponíveis; sem log preservado que comprove sucesso dessas chamadas |
| Markdown → HTML com OpenRouter | `scripts/experimentos/or_gen_html.py`, configurado com `google/gemma-4-26b-a4b-it:free` | Integração implementada; sem log de execução preservado que comprove sucesso |
| Geração de DANFEs sintéticos | `scripts/danfe.py` | Recurso implementado; não foram encontrados resultados de uma avaliação em lote com essas notas |

Os nomes dos arquivos históricos ajudam a identificar a intenção do experimento, mas não substituem um registro de modelo, prompt e parâmetros. Não foi encontrada uma suíte de testes automatizados nem um relatório consolidado de acurácia. Os antigos arquivos da pasta `tests`, hoje em `scripts/experimentos`, são chamadas manuais a modelos.

**Comparação com tempos registrados**

Execução de 24/09/2026, modelo `qwen2.5vl:7b`, contexto de 16.384 tokens, limite de geração de 8.192 tokens, temperatura 0 e seed 42. Fonte: [manifesto da execução](../output/comparacoes/comparacao_20260924T183034_645056Z/manifest.json).

| Entrada | Saída solicitada | Status registrado | Tempo da tentativa |
|---|---|---|---:|
| Imagem original | HTML | Erro: limite de repetição de tokens atingido, HTTP 500 | 142,243 s |
| Imagem original | Markdown | Concluída | 56,253 s |
| OCR em JSON compacto + Markdown (`mixed`) | HTML | Concluída | 79,799 s |
| OCR em JSON compacto + Markdown (`mixed`) | Markdown | Concluída | 45,745 s |

Nessa execução, o caminho com OCR conseguiu produzir HTML, enquanto o caminho direto da imagem falhou. A geração de Markdown com OCR foi aproximadamente 18,7% mais rápida nessa chamada. Os tempos não incluem a extração local anterior, a instalação ou o download do modelo; não medem o tempo total do fluxo. Como o HTML direto falhou, não há comparação válida de velocidade entre duas gerações HTML bem-sucedidas.

O status `ok` indica que a geração passou pelas verificações de saída do código, não que todos os campos estejam corretos. Por exemplo, o Markdown com OCR reorganiza a nota em uma lista de campos e deixa outros textos separados; o HTML preserva uma apresentação tabular. Não houve avaliação sistemática contra uma transcrição de referência.

Essa execução utilizou **JSON e Markdown juntos**, e não JSON isolado. Ela sustenta a utilidade do OCR como etapa anterior, mas não demonstra, sozinha, superioridade de JSON sobre Markdown nem permite classificar os modelos por qualidade.

**Por que destacar a escolha JSON → HTML**

1. **Separação de responsabilidades.** O OCR extrai o conteúdo e a estrutura; a etapa seguinte organiza a apresentação. O JSON pode ser inspecionado e reutilizado sem executar novamente a leitura da imagem.
2. **Controle sobre o que entra no modelo.** É possível selecionar blocos, textos e tabelas e retirar metadados desnecessários. Nos arquivos locais, o JSON completo ocupa 371.027 bytes, o layout simplificado 5.161 bytes e o compacto 10.169 bytes. São tamanhos de representações diferentes, não medições de tokens ou prova de preservação integral.
3. **Preservação da estrutura da DANFE.** Para reconstruir o layout, o JSON pode carregar o próprio HTML reconhecido nas tabelas. Isso mantém mesclagens explícitas; a representação compacta em `rows` expande as células e guarda o texto apenas na origem, sem conservar os atributos originais de mesclagem.
4. **Saída adequada à leitura e impressão.** HTML suporta tabelas, células mescladas, bordas, hierarquia e CSS de impressão. Markdown permanece útil como saída textual alternativa, mas tabelas Markdown simples não representam essas mesclagens diretamente.
5. **Rastreabilidade.** Manter imagem, JSON de origem e saída facilita investigar se uma divergência veio do OCR ou da geração. O manifesto comparativo registra hashes das entradas e parâmetros para apoiar essa conferência.

Assim, a escolha é **JSON documental selecionado → HTML**, preferencialmente conservando as tabelas do OCR quando a fidelidade de layout for o objetivo. JSON é o contêiner dos dados; não é garantia de qualidade. Os próprios testes com JSON bruto e compacto produziram respostas inadequadas. A seleção da entrada, o prompt e a validação continuam necessários.

No fluxo atual, a geração do HTML ainda é feita por um modelo de linguagem: não se trata de um renderizador determinístico. As instruções para não inventar, corrigir ou omitir dados reduzem a ambiguidade da tarefa, mas os artefatos mostram que não garantem seu cumprimento.

**Estado atual e limites para reprodução**

O projeto demonstra extração local e geração experimental de documentos, com exemplos preservados de acertos e falhas. Ainda faltam métricas de preservação de campos, avaliação de layout, repetição das execuções e uma comparação controlada entre JSON, Markdown e entrada mista.

Há duas diferenças no código que precisam ser consideradas ao reproduzir os experimentos: a introdução do notebook comparativo ainda mostra argumentos de linha de comando e caminhos de uma versão anterior do `main.py`; para executar o script local atual, vale `python main.py`, conforme o [guia do fluxo](FLUXO.md). No notebook Gemma, a entrada atual usa `blocks/content`, mas o prompt de Markdown ainda menciona `sections/rows`. Isso impede tratar as versões atuais como uma reprodução exata de todos os resultados históricos.

Este levantamento foi feito por leitura dos códigos, notebooks, arquivos gerados e manifesto disponível. Não foram executadas novas inferências nem repetida a extração OCR para produzir este resumo. Algumas evidências, especialmente `output/comparacoes/`, são locais e ignoradas pelo Git; para compartilhar esses resultados com outra pessoa, é necessário enviar também os artefatos correspondentes.

Para localizar os materiais, consulte o [README](../README.md), o [índice de notebooks](../notebooks/README.md), o [guia dos resultados](../output/README.md).
