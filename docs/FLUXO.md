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

   O PaddleOCR lê as imagens PNG da amostra. No momento, `main.py` está limitado às primeiras cinco capturas (`input_files[:5]`) para o teste piloto. Remova esse limite para processar as 306 imagens. Para cada captura, cria uma pasta própria em `output/output-sintetico/`, com os resultados do OCR e um JSON compacto por página.

3. Instale o [Ollama](https://ollama.com/) e baixe o modelo usado pelo script:

   ```bash
   ollama pull qwen2.5vl:7b
   ```

   Deixe o Ollama em execução durante a geração do HTML. Se ele não iniciar automaticamente, abra outro terminal e rode `ollama serve`.

4. Gere três versões de HTML para cada página: uma usando o JSON compacto, uma usando o Markdown do OCR e uma usando o JSON completo:

   ```bash
   python tests/generate_html.py
   ```

   O script lê o JSON compacto para localizar cada página e, na mesma pasta, procura também o Markdown e o JSON completo do PaddleOCR. Cada entrada é enviada ao Ollama separadamente. Os arquivos terminam em `_html.html` (JSON compacto), `_from_md_html.html` (Markdown) e `_from_full_json_html.html` (JSON completo).

5. Compare OCR e HTML com os gabaritos:

   ```bash
   python tests/validate_dataset.py
   ```

   O relatório `output/output-sintetico/relatorio_validacao.md` compara os três formatos de entrada e os três HTMLs com o gabarito. Aponta texto faltando ou extra, diferenças na estrutura das tabelas e valores do XML ausentes. Também resume os resultados por resolução, aparência e estrutura. Rode esta etapa depois do OCR e da geração HTML. Se ainda não houver resultados, o relatório avisa que a comparação está pendente.

Rode os comandos a partir da pasta principal do projeto. Os scripts locais encontram os arquivos usando a localização do projeto.

## Usar o notebook (opcional)

O notebook `notebooks/NLP_26_2_openrouter_test.ipynb` executa experimentos no Google Colab. Ele não faz a etapa de OCR: primeiro gere o JSON com `python main.py` localmente.

1. Abra o notebook no Colab e execute as células de instalação e inicialização do Ollama.
2. Execute a célula que baixa `qwen2.5:7b`.
3. Para o caminho JSON → HTML, execute a célula 9 e carregue um JSON completo salvo dentro da pasta do documento em `output/output-sintetico/`. A célula seleciona os blocos relevantes e pede ao modelo para gerar o HTML.
4. O HTML será salvo como `nota_fiscal_json.html` no ambiente do Colab e poderá ser baixado pela própria célula.

As células 4 e 7 fazem outro experimento: recebem `img_1.md` e geram HTML a partir de Markdown. Para seguir o fluxo JSON → HTML, use a célula 9.

## Observação

O HTML e o XML de referência servem para avaliação; não são enviados ao modelo. O JSON compacto guarda o tipo, a posição e o conteúdo de cada bloco. A tabela HTML reconhecida pelo PaddleOCR pode estar dentro de `content`, incluindo `rowspan` e `colspan`. A validação textual ignora diferenças de maiúsculas, acentos e pontuação; a validação estrutural conta tabelas, linhas, células e mesclagens, mas não compara a aparência em pixels.
