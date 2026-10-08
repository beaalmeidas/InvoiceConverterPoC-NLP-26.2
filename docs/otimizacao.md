# Simplificação da pipeline de OCR e HTML

O objetivo foi aproximar o código do fluxo original: selecionar arquivos,
carregar o PaddleOCR uma vez, processar o lote e salvar as saídas. O `main.py`
passou de **443 para 134 linhas**, sem adicionar arquivos auxiliares.

## Organização atual

- [main.py](../main.py): argumentos, seleção das entradas, configuração direta
  de `PPStructureV3`, processamento e exportação. A função `simplify_result` e
  os nomes `file_output_dir`, `page_number` e `layout_path` seguem a versão inicial.
- [pipeline.py](../pipeline.py) (antes `tests/generate_html.py`): pré-processamento da imagem,
  reconstrução de tabelas, caminhos de saída e geração de HTML direto ou com Qwen.
- [ocr_metrics.py](../tests/ocr_metrics.py): comparações de texto, células e
  produtos, usadas apenas nos testes e na avaliação manual da amostra.

Os arquivos `ocr_quality.py`, `ocr_tables.py` e `output_paths.py` continuam
removidos. A reconstrução das tabelas foi mantida porque participa da geração
correta das células e mesclagens.

## Código removido da execução normal

Foram retirados a geração de evidências por linha de OCR, os arquivos
`*_ocr.json` e `*_native.json`, a coleta de versões das dependências, hashes e
cronômetros por etapa do OCR. O registro `ocr_run.json` guarda apenas o status
e o recorte, para a avaliação manual excluir execuções falhas ou parciais.

As comparações detalhadas de texto e estrutura do HTML também saíram da geração
normal. Elas permanecem nos testes. O gerador ainda rejeita respostas encerradas
pelo limite de tokens e preserva a limpeza de cercas de Markdown.

## Comportamento preservado

O PaddleOCR é carregado uma vez por lote. Foram mantidos português, resolução da
detecção, recorte e ampliação das imagens, correção de orientação, reconhecimento
e reconstrução das tabelas, seleção de arquivos e tratamento de falhas por entrada.

As saídas continuam sendo JSON completo, Markdown, HTML das tabelas e JSON
simplificado por página. Os HTMLs finais ficam em `output/html/` e os demais
arquivos em `output/intermediarios/`.

O notebook mantém suas 12 células e os fluxos de Markdown e JSON, com 79 linhas
de código. As duas chamadas usam o mesmo prompt, cliente Ollama e modelo
`qwen2.5vl:7b`, com `keep_alive="10m"`.

## Prompt

O prompt base pede um documento HTML completo, preservação literal do texto,
associação entre campos e valores e conservação das tabelas, incluindo `td`,
`th`, `rowspan` e `colspan`. Uma instrução curta identifica o formato da entrada
e explica o uso de `content`, `type` e `bbox`, sem mostrar coordenadas.

O JSON é enviado sem espaços de formatação. O limite de geração foi preservado
para evitar truncar notas maiores. A geração direta permanece como padrão;
o Qwen pode ser usado com `--engine ollama`.

## Validação

Os 25 testes restantes passaram após a simplificação. O teste das evidências
por linha foi removido junto com essa funcionalidade. Na comparação local,
os 15 HTMLs da amostra permaneceram idênticos aos anteriores.

A versão final também processou a nota `01-1-padrao-300` com o PaddleOCR real.
O layout simplificado, o Markdown e o HTML da tabela ficaram idênticos aos da
versão de 443 linhas. A geração dos três formatos de HTML concluiu normalmente.

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

A conferência da amostra pode ser executada separadamente:

```bash
.venv/bin/python tests/validate_dataset.py
```

## Medições da refatoração inicial

| Comparação | Antes | Depois | Alcance |
|---|---:|---:|---|
| Preparar, renderizar e validar HTML direto | 0,915 s | 0,519 s | Mediana de sete rodadas com 15 saídas; redução de 43% |
| OCR e exportação | 64,3 s | 56,1 s | Uma nota de 300 dpi, sem carregamento dos modelos |
| Qwen | 76,35 s | 64,99 s | Uma tabela pequena, em CPU, com o mesmo modelo carregado |
| Tokens de entrada do Qwen | 243 | 176 | Mesma tabela com os dois prompts |

Esses tempos foram medidos antes da remoção das validações automáticas e dos
registros detalhados. Não representam um benchmark da versão final simplificada.
As medições únicas de OCR e Qwen não comprovam ganhos recorrentes em notas
completas. Não foi medida a latência total da versão final com Qwen.
