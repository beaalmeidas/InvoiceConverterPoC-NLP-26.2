# Invoice Converter – NLP 26.2 PoC
This is an ongoing trial project for the TAIL-NLP 2026.2 project.

---

## Project structure
```
docs/
   └── FLUXO.md
   └── RESUMO_SEMANAL.md
input/
   └── danfe-1900366884.pdf
   └── img_1.jpg
notebooks/
   └── NLP_26_2_openrouter_test.ipynb
output/
   └── imgs/
   └── img_O_layout.json
   └── img_1_res.json
   └── img_1_table_1.html
   └── img_1.md
qwen-html/
   └── img_1_rep.html
   └── nota_fiscal_json.html
tests/
   └── generate_html.py
   └── or_gen_html.py
   └── test.py                               # for testing local LLms for response
   └── validate_dataset.py
main.py
```
