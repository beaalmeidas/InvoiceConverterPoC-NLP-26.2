import ollama

with open("output/img_0_layout.json", "r", encoding="utf-8") as file:
    JSON = file.read()

# atualização: melhora no prompt
# changing to fit json more
prompt = f"""
You are converting a Brazilian electronic invoice (NF-e) OCR into HTML.

Convert the following JSON output representation into a complete HTML document.

Requirements:
- Generate a full HTML doc, with simple CSS for screen and printing.
- Preserve all information from the input, each block and its order.
- Do not invent information.
- Preserve the hierarchy and organization of the invoice.
- Represent tables as HTML tables.
- Distinguish labels from their corresponding values when possible.
- Keep numeric values, dates, CNPJ, invoice numbers and product information exactly as provided.
- Return only the HTML code.
- Do not summarize, rewrite, correct, or complete the OCR content.
- Do not return MarkDown fences.
- Input is a compact JSON made by PaddleOCR:
    - "type" identifies the type of block;
    - "bbox" has coordinates of the pages and its use is used to order/locate the block
    - "content" has the text recognized or html made by the OCR.
- If "content" has an HTML table, preserve its lines, cells, rowspan and colspan.
- Do not show the "bbox" coordinates.
INPUT:

{JSON}
"""

response = ollama.chat(
    model="qwen3.5:9b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

html = response["message"]["content"]

with open("output/img_1_from_md.html", "w", encoding="utf-8") as file:
    file.write(html)