from pathlib import Path
import json
import re
import ollama


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "output" / "output-sintetico"
MODEL = "qwen2.5vl:7b"

BASE_PROMPT = """
You are converting OCR output from a Brazilian electronic invoice (NF-e) into HTML.

Create a complete HTML document containing the invoice as a readable, compact table.
- Keep all recognized text exactly as given and in the same order. Do not summarize, correct, or invent text.
- Preserve table rows, cells, rowspan, and colspan. Do not merge unrelated cells.
- Include semantic HTML and a table; return only HTML, without Markdown fences or explanations.
"""

HTML_STYLE = """<style>
@page { size: A4 portrait; margin: 10mm; }
* { box-sizing: border-box; }
body { margin: 0 auto; max-width: 210mm; padding: 12px; color: #111; font: 10px/1.25 Arial, sans-serif; }
table { width: 100%; border-collapse: collapse; table-layout: auto; }
td, th { border: 1px solid #333; padding: 3px 4px; vertical-align: top; overflow-wrap: anywhere; }
tr { break-inside: avoid; }
img { max-width: 100%; height: auto; }
@media screen { body { background: #fff; box-shadow: 0 1px 8px #bbb; margin: 20px auto; } }
@media print { body { width: 100%; max-width: none; padding: 0; font-size: 8px; } }
</style>"""

SOURCE_INSTRUCTIONS = {
    "compact": "The input is compact PaddleOCR JSON. Use each block's content; type identifies the block and bbox is only for position/order. Never show bbox.",
    "markdown": "The input is PaddleOCR Markdown. Preserve its text and any HTML tables included in it.",
    "full_json": "The input is the full PaddleOCR JSON. Use document text and table markup from parsing_res_list/block_content (including inside res.parsing_res_list if wrapped). Ignore model settings, scores, dimensions, and other processing metadata; never turn those numbers into invoice content. Use bounding boxes only to determine reading order.",
}


def source_files(layout_path):
    stem = layout_path.name.rsplit("_page_", 1)[0]
    return {
        "compact": layout_path,
        "markdown": layout_path.parent / f"{stem}.md",
        "full_json": layout_path.parent / f"{stem}_res.json",
    }


def prompt_for(kind, content):
    return (
        f"{BASE_PROMPT}\n\nINPUT TYPE:\n{SOURCE_INSTRUCTIONS[kind]}\n\n"
        f"INPUT:\n\n{content}"
    )


def prepare_content(kind, source_path):
    content = source_path.read_text(encoding="utf-8")
    if kind != "full_json":
        return content

    # O JSON integral contém coordenadas e metadados extensos que não ajudam
    # na reconstrução. Mantém apenas os blocos reconhecidos pelo PaddleOCR.
    data = json.loads(content)
    data = data.get("res", data)
    blocks = data.get("parsing_res_list", [])
    relevant = [
        {
            "type": block.get("block_label"),
            "bbox": block.get("block_bbox"),
            "content": block.get("block_content"),
        }
        for block in blocks
    ]
    return json.dumps(relevant, ensure_ascii=False, separators=(",", ":"))


def add_presentation(html):
    """Aplica estilo uniforme sem alterar o texto nem a estrutura reconhecida."""
    html = re.sub(r"^```(?:html)?\s*|\s*```$", "", html.strip(), flags=re.IGNORECASE)
    if not re.search(r"<html\b", html, flags=re.IGNORECASE):
        html = f"<!doctype html><html><head><meta charset=\"utf-8\"></head><body>{html}</body></html>"
    if re.search(r"</head\s*>", html, flags=re.IGNORECASE):
        html = re.sub(r"</head\s*>", HTML_STYLE + "</head>", html, count=1, flags=re.IGNORECASE)
    else:
        html = re.sub(r"<html\b[^>]*>", lambda match: match.group(0) + "<head><meta charset=\"utf-8\">" + HTML_STYLE + "</head>", html, count=1, flags=re.IGNORECASE)
    return html


def output_path(layout_path, kind):
    stem = layout_path.name.removesuffix("_layout.json")
    suffix = {
        "compact": "_html.html",
        "markdown": "_from_md_html.html",
        "full_json": "_from_full_json_html.html",
    }[kind]
    return layout_path.with_name(stem + suffix)


def main():
    layout_files = sorted(RESULTS_DIR.rglob("*_layout.json"))
    if not layout_files:
        raise FileNotFoundError(
            f"Nenhum JSON compacto (*_layout.json) foi encontrado em {RESULTS_DIR}. Rode main.py primeiro."
        )

    for layout_path in layout_files:
        for kind, source_path in source_files(layout_path).items():
            if not source_path.is_file():
                print(f"Entrada {kind} ausente: {source_path.relative_to(BASE_DIR)}")
                continue

            try:
                content = prepare_content(kind, source_path)
                response = ollama.chat(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt_for(kind, content)}],
                    options={"temperature": 0},
                )
                html = add_presentation(response["message"]["content"])
                if not re.search(r"<table\b", html, flags=re.IGNORECASE):
                    raise ValueError("A resposta do modelo não contém uma tabela HTML; arquivo não salvo.")

                destination = output_path(layout_path, kind)
                destination.write_text(html, encoding="utf-8")
                print(f"HTML ({kind}) salvo: {destination.relative_to(BASE_DIR)}")
            except Exception as exc:
                print(f"Falha no fluxo {kind} para {source_path.name}: {exc}")


if __name__ == "__main__":
    main()
