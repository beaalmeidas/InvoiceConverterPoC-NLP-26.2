import ollama

with open("output/img_1.md", "r", encoding="utf-8") as file:
    markdown = file.read()

prompt = f"""
You are converting a Brazilian electronic invoice (NF-e) into HTML.

Convert the following Markdown representation into a complete HTML document.

Requirements:
- Preserve all information from the input.
- Do not invent information.
- Preserve the hierarchy and organization of the invoice.
- Represent tables as HTML tables.
- Distinguish labels from their corresponding values when possible.
- Keep numeric values, dates, CNPJ, invoice numbers and product information exactly as provided.
- Return only the HTML code.
- Do not use Markdown code fences.

INPUT:

{markdown}
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