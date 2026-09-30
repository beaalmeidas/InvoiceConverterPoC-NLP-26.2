# Fluxo do projeto

O projeto transforma uma imagem de DANFE em HTML em três etapas:

```text
imagem → PaddleOCR → JSON → HTML
```

## Rodar tudo localmente

1. Coloque a imagem a processar em `input/img_1.jpg`.
2. Na pasta principal do projeto, rode:

   ```bash
   python main.py
   ```

   O PaddleOCR lê a imagem. Os arquivos aparecem em `output/`; `img_0_layout.json` é a versão compacta usada na etapa seguinte. O OCR também salva arquivos JSON, Markdown e HTML.

3. Instale e inicie o [Ollama](https://ollama.com/) e baixe o modelo usado pelo script:

   ```bash
   ollama pull qwen3.5:9b
   ```

4. Gere o HTML a partir do JSON compacto:

   ```bash
   python tests/generate_html.py
   ```

   O script lê `output/img_0_layout.json`, envia o conteúdo ao Ollama e salva o resultado em `output/img_1_from_md.html` (o nome do arquivo é antigo; a entrada atual é JSON).

Rode os comandos a partir da pasta principal do projeto. O `main.py` está configurado para processar `input/img_1.jpg`.

## Usar o notebook

O notebook `notebooks/NLP_26_2_openrouter_test.ipynb` executa experimentos no Google Colab. Ele não faz a etapa de OCR: primeiro gere o JSON com `python main.py` localmente.

1. Abra o notebook no Colab e execute as células de instalação e inicialização do Ollama.
2. Execute a célula que baixa `qwen2.5:7b`.
3. Para o caminho JSON → HTML, execute a célula 9 e carregue o JSON completo `output/img_1_res.json`. A célula seleciona os blocos relevantes e pede ao modelo para gerar o HTML.
4. O HTML será salvo como `nota_fiscal_json.html` no ambiente do Colab e poderá ser baixado pela própria célula.

As células 4 e 7 fazem outro experimento: recebem `img_1.md` e geram HTML a partir de Markdown. Para seguir o fluxo JSON → HTML, use a célula 9.

## Observação

O JSON compacto guarda o tipo, a posição e o conteúdo de cada bloco. A tabela HTML reconhecida pelo PaddleOCR pode estar dentro de `content`, incluindo `rowspan` e `colspan`. O modelo organiza esse conteúdo em HTML; erros já presentes no OCR podem aparecer também no resultado.
