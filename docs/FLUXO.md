# Fluxo do projeto

O projeto transforma imagens de DANFE em HTML. O fluxo local processa vários arquivos em sequência:

```text
imagem → PaddleOCR → JSON → HTML
```

## Rodar tudo localmente

1. Coloque as imagens PNG da amostra em `input/amostra_diversa/captura/`. O programa procura também nas subpastas de resolução. HTMLs, XMLs e o manifesto ficam em `input/amostra_diversa/` como referências e não são enviados ao OCR.
2. Na pasta principal do projeto, rode:

   ```bash
   python main.py
   ```

   O PaddleOCR lê as imagens PNG da amostra. Por padrão, `main.py` processa as primeiras cinco capturas. Use `python main.py --limit 1` para uma imagem ou `python main.py --limit 0` para todas. Para cada captura, cria uma pasta própria em `output/output-sintetico/`, com os resultados do OCR e um JSON compacto por página.

   A correção de deformação (`use_doc_unwarping`) fica desligada: as capturas sintéticas já são planas e essa etapa estava cortando a margem da imagem. A orientação continua ativada.

3. Instale o [Ollama](https://ollama.com/) e baixe o modelo usado pelo script:

   ```bash
   ollama pull qwen2.5vl:7b
   ```

   Deixe o Ollama em execução durante a geração do HTML. Se ele não iniciar automaticamente, abra outro terminal e rode `ollama serve`.

4. Gere três versões de HTML para cada página: uma usando o JSON compacto, uma usando o Markdown do OCR e uma usando o JSON completo:

   ```bash
   python tests/generate_html.py
   ```

   O script lê o JSON compacto para localizar cada página e, na mesma pasta, procura também o Markdown e o JSON completo do PaddleOCR. Cada entrada é enviada ao Ollama separadamente. No fluxo que lê o JSON completo, o script extrai os blocos reconhecidos antes do envio: portanto, a comparação atual não usa o JSON integral com todos os metadados. Os arquivos terminam em `_html.html` (JSON compacto), `_from_md_html.html` (Markdown) e `_from_full_json_html.html` (JSON completo).

   O script usa contexto de 16384 tokens e limite de saída de 8192 tokens (ajustáveis com `--num-ctx` e `--num-predict`). Isso pode aumentar o uso de memória. Antes de salvar, rejeita respostas cortadas, tags estruturais sem fechamento, mudanças nas contagens de tabelas/células/linhas existentes e perda de mais de 5% dos termos da entrada. Essas verificações não garantem fidelidade visual nem detectam todos os erros de conteúdo.

   Ao lado de cada saída, `.response.txt` guarda a resposta original e `.generation.json` guarda o resultado, o motivo de término e os tokens usados. Se a nova tentativa falhar, um HTML anterior é movido para `.previous.html`, evitando que seja avaliado como um resultado novo. O CSS é aplicado somente após essa checagem.

5. Compare OCR e HTML com os gabaritos:

   ```bash
   python tests/validate_dataset.py
   ```

   O relatório `output/output-sintetico/relatorio_validacao.md` compara os três formatos de entrada e os três HTMLs com o gabarito. Aponta texto faltando ou extra, diferenças na estrutura das tabelas e valores do XML ausentes. Também resume os resultados por resolução, aparência e estrutura. Rode esta etapa depois do OCR e da geração HTML. Se ainda não houver resultados, o relatório avisa que a comparação está pendente.

Rode os comandos a partir da pasta principal do projeto. Os scripts locais encontram os arquivos usando a localização do projeto.

## Testar uma imagem sem sobrescrever o lote

```bash
python main.py --file input/amostra_diversa/captura/100/01-16a40-espacada-100/p01.png --limit 1 --output output/diagnostico-sem-unwarping
```

O primeiro teste está registrado em `output/diagnostico-sem-unwarping/COMPARACAO.md`: recuperou a margem e um código cortado, mas ainda houve mistura de células. Compare os arquivos dessa pasta com os antigos em `output/output-sintetico`. Primeiro confira os textos, códigos e tabelas do OCR. Depois gere os três HTMLs do teste:

```bash
python tests/generate_html.py --results-dir output/diagnostico-sem-unwarping --limit 1
```

O validador de dataset continua usando `output/output-sintetico`; ele não inclui essa pasta de diagnóstico. Quando estiver satisfeito com o teste, rode o fluxo normal novamente para atualizar o lote. Para avaliar resolução, selecione capturas de 200 ou 300 dpi com `--file`; as cinco primeiras do lote são de 100 dpi. Prefira a mesma nota e layout quando disponíveis.

## Usar o notebook (opcional)

O notebook `notebooks/NLP_26_2_openrouter_test.ipynb` executa experimentos no Google Colab. Ele não faz a etapa de OCR: primeiro gere o JSON com `python main.py` localmente.

1. Abra o notebook no Colab e execute as células de instalação e inicialização do Ollama.
2. Execute a célula que baixa `qwen2.5:7b`.
3. Para o caminho JSON → HTML, execute a célula 9 e carregue um JSON completo salvo dentro da pasta do documento em `output/output-sintetico/`. A célula seleciona os blocos relevantes e pede ao modelo para gerar o HTML.
4. O HTML será salvo como `nota_fiscal_json.html` no ambiente do Colab e poderá ser baixado pela própria célula.

As células 4 e 7 fazem outro experimento: recebem `img_1.md` e geram HTML a partir de Markdown. Para seguir o fluxo JSON → HTML, use a célula 9.

## Observação

O HTML e o XML de referência servem para avaliação; não são enviados ao modelo. O JSON compacto guarda o tipo, a posição e o conteúdo de cada bloco. A tabela HTML reconhecida pelo PaddleOCR pode estar dentro de `content`, incluindo `rowspan` e `colspan`. A validação textual ignora diferenças de maiúsculas, acentos e pontuação; a validação estrutural conta tabelas, linhas, células e mesclagens, mas não compara a aparência em pixels.
