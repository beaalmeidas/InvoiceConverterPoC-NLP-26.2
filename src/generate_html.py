from pathlib import Path
import argparse
from collections import Counter
from html.parser import HTMLParser
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
- Preserve table rows, cells, rowspan, and colspan. Keep td/th tags unchanged. Do not merge unrelated cells.
- Return a complete HTML document with explicitly closed html, body, table, tr and td/th tags.
- Include every input table and every row. Return only HTML, without Markdown fences or explanations.
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
    "full_json": "The input contains the recognized blocks extracted from the full PaddleOCR JSON. Use content as document text/table markup, type as the block label and bbox only for reading order. Do not print coordinates.",
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


class DocumentParser(HTMLParser):
    tracked = {"html", "body", "table", "thead", "tbody", "tfoot", "tr", "td", "th"}

    def __init__(self):
        super().__init__()
        self.stack = []
        self.errors = []
        self.counts = Counter()
        self.text = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"style", "script"}:
            self.hidden += 1
        if tag in self.tracked:
            self.stack.append(tag)
            self.counts[tag] += 1

    def handle_endtag(self, tag):
        if tag in {"style", "script"}:
            self.hidden = max(0, self.hidden - 1)
        if tag in self.tracked:
            if not self.stack or self.stack[-1] != tag:
                self.errors.append(f"Fechamento inesperado: {tag}")
            else:
                self.stack.pop()

    def handle_data(self, data):
        if not self.hidden:
            self.text.append(data)


def inspect_html(text):
    parser = DocumentParser()
    parser.feed(text)
    parser.close()
    return parser


def source_html(kind, content):
    if kind == "markdown":
        return content
    return "\n".join(str(block.get("content") or "") for block in json.loads(content))


def validate_response(raw, reference, reason):
    if reason == "length":
        raise ValueError("Ollama atingiu o limite de geração (done_reason=length).")
    result = inspect_html(raw)
    if result.stack or result.errors:
        raise ValueError("HTML incompleto ou com tags fora de ordem.")
    if any(result.counts[tag] == 0 for tag in ("html", "body", "table")):
        raise ValueError("Resposta sem documento HTML completo e tabela.")
    source = inspect_html(reference)
    for tag in ("table", "tr", "td", "th"):
        if result.counts[tag] != source.counts[tag] and source.counts[tag]:
            raise ValueError(f"Estrutura alterada: {tag}: entrada={source.counts[tag]}, saída={result.counts[tag]}.")
    expected = Counter(re.findall(r"\w+", " ".join(source.text).casefold()))
    actual = Counter(re.findall(r"\w+", " ".join(result.text).casefold()))
    coverage = sum((expected & actual).values()) / max(1, sum(expected.values()))
    if coverage < 0.95:
        raise ValueError(f"Conteúdo incompleto: apenas {coverage:.1%} dos termos da entrada preservados.")
    return coverage


def main():
    parser = argparse.ArgumentParser(description="Gera HTML e rejeita respostas incompletas")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Máximo de páginas; 0 = todas")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--num-predict", type=int, default=8192)
    args = parser.parse_args()
    if args.limit < 0 or args.num_predict < 1 or args.num_ctx <= args.num_predict:
        parser.error("Limite deve ser >= 0 e num-ctx deve ser maior que num-predict > 0")
    layout_files = sorted(args.results_dir.rglob("*_layout.json"))
    if args.limit:
        layout_files = layout_files[:args.limit]
    if not layout_files:
        raise FileNotFoundError(f"Nenhum JSON compacto encontrado em {args.results_dir}. Rode main.py primeiro.")

    for layout_path in layout_files:
        for kind, source_path in source_files(layout_path).items():
            if not source_path.is_file():
                print(f"Entrada {kind} ausente: {source_path}")
                continue
            destination = output_path(layout_path, kind)
            log_path = destination.with_suffix(".generation.json")
            log = {"source": str(source_path), "model": MODEL,
                   "options": {"temperature": 0, "num_ctx": args.num_ctx, "num_predict": args.num_predict},
                   "status": "failed"}
            try:
                content = prepare_content(kind, source_path)
                print(f"Gerando {kind}: {source_path}", flush=True)
                response = ollama.chat(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt_for(kind, content)}],
                    options=log["options"],
                )
                for key in ("done", "done_reason", "prompt_eval_count", "eval_count", "total_duration", "eval_duration"):
                    log[key] = response.get(key)
                raw = response["message"]["content"].strip()
                # Guarda a resposta original para diagnosticar inclusive falhas de validação.
                destination.with_suffix(".response.txt").write_text(raw, encoding="utf-8")
                raw = re.sub(r"^```(?:html)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
                log["content_coverage"] = validate_response(raw, source_html(kind, content), log.get("done_reason"))
                html = add_presentation(raw)
                temporary = destination.with_suffix(".tmp")
                temporary.write_text(html, encoding="utf-8")
                temporary.replace(destination)
                log["status"] = "accepted"
                print(f"HTML ({kind}) salvo: {destination}", flush=True)
            except Exception as exc:
                log["error"] = str(exc)
                # Evita que o validador conte um resultado antigo como sucesso da nova execução.
                if destination.exists():
                    destination.replace(destination.with_suffix(".previous.html"))
                print(f"Falha no fluxo {kind} para {source_path.name}: {exc}", flush=True)
            finally:
                log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
