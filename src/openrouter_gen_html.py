import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

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

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json={
        "model": "google/gemma-4-26b-a4b-it:free",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
)

print("Status:", response.status_code)

result = response.json()

print("Resposta da API:")
print(result)

response.raise_for_status()

message = result["choices"][0]["message"]

print("\nMessage:")
print(message)

html = message.get("content")

if not html:
    raise ValueError(
        f"O modelo não retornou conteúdo em 'content'. Message recebida: {message}"
    )

with open("output/img_1_from_md.html", "w", encoding="utf-8") as file:
    file.write(html)

print("\nHTML gerado com sucesso.")