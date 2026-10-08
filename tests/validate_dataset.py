"""compara as três saídas do ocr e os htmls com os gabaritos."""

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path
import csv
import json
import re
from html import escape
import unicodedata
import xml.etree.ElementTree as ET
import argparse
import sys


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from bs4 import BeautifulSoup

from pipeline import HTML_SUFFIX, prepare_content, result_directories
DATASET_DIR = BASE_DIR / "input" / "amostra_diversa"
RESULTS_DIR = BASE_DIR / "output" / "output-sintetico"
MANIFEST_PATH = DATASET_DIR / "manifesto.csv"
REPORT_PATH = RESULTS_DIR / "relatorio_validacao.md"
STRUCTURE_TAGS = ("table", "tr", "td", "th")

# variante: (nome da entrada, nome do html)
VARIANTS = {"compact": ("JSON compacto", "HTML compacto"), "markdown": ("Markdown OCR", "HTML do Markdown"),
            "full_json": ("JSON completo", "HTML do JSON completo")}


class VisibleHTMLParser(HTMLParser):
    """lê textos e tabelas; ignora estilos e scripts."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text_parts = []
        self.tags = Counter()
        self.spans = {"rowspan": Counter(), "colspan": Counter()}
        self.skipping = None

    def handle_starttag(self, tag, attrs):
        if self.skipping:
            return
        if tag in {"style", "script", "noscript"}:
            self.skipping = tag
            return
        if tag in STRUCTURE_TAGS:
            self.tags[tag] += 1
        if tag in {"td", "th"}:
            for key, value in attrs:
                if key in self.spans:
                    self.spans[key][value or "1"] += 1

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if self.skipping == tag:
            self.skipping = None

    def handle_data(self, data):
        if not self.skipping and data.strip():
            self.text_parts.append(data.strip())


def parse_html(source):
    parser = VisibleHTMLParser()
    parser.feed(source)
    return {
        "text": " ".join(parser.text_parts),
        "tags": parser.tags,
        "spans": parser.spans,
    }


def unpack_result(data):
    if isinstance(data, str):
        data = json.loads(data)
    return data.get("res", data)


def visible_text(content):
    return parse_html(str(content or ""))["text"]


def tokens(text):
    text = unicodedata.normalize("NFKD", text).casefold()
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", text)


class DocumentParser(HTMLParser):
    tracked = {"html", "body", "table", "thead", "tbody", "tfoot", "tr", "td", "th"}

    def __init__(self):
        super().__init__()
        self.stack = []
        self.errors = []
        self.counts = Counter()

    def handle_starttag(self, tag, attrs):
        if tag in self.tracked:
            self.stack.append(tag)
            self.counts[tag] += 1

    def handle_endtag(self, tag):
        if tag in self.tracked:
            if not self.stack or self.stack[-1] != tag:
                self.errors.append(f"Fechamento inesperado: {tag}")
            else:
                self.stack.pop()

def inspect_html(text):
    parser = DocumentParser()
    parser.feed(text)
    parser.close()
    return parser


def source_html(kind, content):
    if kind == "markdown":
        return content
    return "\n".join(
        str(block.get("content") or "") if block.get("type") == "table"
        else escape(str(block.get("content") or ""))
        for block in json.loads(content)
    )


def validate_response(raw, reference, reason):
    if reason == "length":
        raise ValueError("Ollama atingiu o limite de geração (done_reason=length).")
    result = inspect_html(raw)
    if result.stack or result.errors:
        raise ValueError("HTML incompleto ou com tags fora de ordem.")
    if any(result.counts[tag] == 0 for tag in ("html", "body")):
        raise ValueError("Resposta sem documento HTML completo.")
    source = inspect_html(reference)
    for tag in ("table", "tr", "td", "th"):
        if result.counts[tag] != source.counts[tag] and source.counts[tag]:
            raise ValueError(f"Estrutura alterada: {tag}: entrada={source.counts[tag]}, saída={result.counts[tag]}.")
    def document_signature(markup):
        soup = BeautifulSoup(markup, "html.parser")
        for hidden in soup.find_all(["head", "style", "script"]):
            hidden.decompose()
        text = " ".join(soup.get_text(" ", strip=True).split())
        cells = [(tag.name, tag.get("rowspan", "1"), tag.get("colspan", "1"),
                  " ".join(tag.get_text(" ", strip=True).split()))
                 for tag in soup.find_all(["table", "tr", "td", "th"])]
        return text, cells

    expected_text, expected_cells = document_signature(reference)
    actual_text, actual_cells = document_signature(raw)
    if expected_text != actual_text:
        raise ValueError("Conteúdo alterado, omitido, acrescentado ou fora de ordem.")
    if expected_cells != actual_cells:
        raise ValueError("Células, mesclagens ou ordem das tabelas alteradas.")
    return 1.0


def recall(reference, candidate):
    expected, actual = Counter(tokens(reference)), Counter(tokens(candidate))
    return sum((expected & actual).values()) / max(1, sum(expected.values()))


class RowsParser(HTMLParser):
    """separa tabelas e células; ignora células com tabelas internas."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self.rows, self.cells, self.finished = [], [], [], []
        self.next_table = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            self.next_table += 1
            self.tables.append(self.next_table)
            if self.cells:
                self.cells[-1]["nested"] = True
        elif tag == "tr":
            self.rows.append({"table": self.tables[-1] if self.tables else 0, "cells": []})
        elif tag in {"td", "th"}:
            self.cells.append({"text": [], "nested": False,
                               "colspan": attrs.get("colspan", "1"), "rowspan": attrs.get("rowspan", "1")})

    def handle_data(self, data):
        if self.cells:
            self.cells[-1]["text"].append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cells:
            cell = self.cells.pop()
            cell["text"] = " ".join(" ".join(cell["text"]).split())
            if self.rows:
                self.rows[-1]["cells"].append(cell)
        elif tag == "tr" and self.rows:
            self.finished.append(self.rows.pop())
        elif tag == "table" and self.tables:
            self.tables.pop()


def column_kind(text):
    label = "".join(tokens(text))
    if label in {"codigo", "cod", "codproduto", "codigoproduto", "codigodoproduto", "codigodoprodutoservico"}:
        return "code"
    if label.startswith("descri"):
        return "description"
    if label in {"qtd", "qtde", "quant", "quantidade"}:
        return "quantity"
    if label in {"vunit", "vlunit", "vlrunit", "valorunit", "valorunitario", "vlrunitario"}:
        return "unit_price"
    if label in {"vtotal", "vltotal", "vlrtotal", "valortotal", "valortotalbruto"}:
        return "total"
    return None


def product_rows(html):
    parser = RowsParser()
    parser.feed(html)
    headers, records = {}, []
    for row in parser.finished:
        cells = row["cells"]
        if not cells or any(c["nested"] or c["colspan"] != "1" or c["rowspan"] != "1" for c in cells):
            continue
        mapping = {column_kind(c["text"]): i for i, c in enumerate(cells) if column_kind(c["text"])}
        if "code" in mapping and "description" in mapping:
            headers[row["table"]] = mapping
        elif row["table"] in headers:
            records.append({kind: cells[i]["text"] for kind, i in headers[row["table"]].items() if i < len(cells)})
    return records


def decimal_cell(text):
    text = text.strip()
    if not re.fullmatch(r"-?\d[\d.,]*", text):
        return None
    if "," in text:
        text = text.replace(".", "").replace(",", ".")
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def xml_products(path):
    root = ET.parse(path).getroot()
    products = []
    for node in root.iter():
        if node.tag.split("}")[-1] != "prod":
            continue
        values = {child.tag.split("}")[-1]: (child.text or "").strip() for child in node}
        products.append({key: values.get(tag, "") for key, tag in
                         {"code": "cProd", "description": "xProd", "quantity": "qCom",
                          "unit_price": "vUnCom", "total": "vProd"}.items()})
    return products


def check_products(products, html):
    rows = product_rows(html)
    used, checks = set(), []
    for product in products:
        # exige código exato, com pontuação e zeros iniciais
        candidates = [(i, row) for i, row in enumerate(rows)
                      if i not in used and row.get("code", "").strip() == product["code"]]
        def field_matches(row):
            return {key: decimal_cell(row.get(key, "")) == Decimal(product[key])
                    for key in ("quantity", "unit_price", "total") if product[key]}
        best = max(candidates, key=lambda pair: sum(field_matches(pair[1]).values()), default=None)
        row = best[1] if best else {}
        if best:
            used.add(best[0])
        fields = field_matches(row)
        checks.append({"code": product["code"], "code_in_product_column": best is not None,
                       "fields": fields, "row_exact": best is not None and len(fields) == 3 and all(fields.values()),
                       "expected": product, "recognized": row})
    return checks


def write_ocr_report(manifest, dataset_dir, results_dir, artifact_paths):
    documents = []
    for row in manifest:
        captures = [p.strip() for p in row.get("caminho_imagem", "").split(";") if p.strip()]
        paths = [artifact_paths(p) for p in captures]
        if not paths or any(p is None or not p["source"]["full_json"].is_file() for p in paths):
            continue
        raw, structured, html = [], [], []
        skip = False
        for p in paths:
            full = p["source"]["full_json"]
            metadata_path = full.parent / "ocr_run.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata.get("crop") or metadata.get("status") != "ok":
                    skip = True
                    break
            data = unpack_result(json.loads(full.read_text(encoding="utf-8")))
            raw.extend(data.get("overall_ocr_res", {}).get("rec_texts", []))
            markup = "\n".join(str(b.get("block_content") or "") for b in data.get("parsing_res_list", []))
            structured.append(visible_text(markup))
            html.append(markup)
        if skip:
            continue
        reference = visible_text((dataset_dir / row["caminho_html"]).read_text(encoding="utf-8"))
        products = xml_products(dataset_dir / "xml" / (row["xml_id"] + ".xml"))
        checks = check_products(products, "\n".join(html))
        raw_text, structured_text = " ".join(raw), " ".join(structured)
        documents.append({"doc_id": row["doc_id"], "xml_id": row["xml_id"], "dpi": row["captura_dpi"],
                          "raw_recall": recall(reference, raw_text),
                          "structured_recall": recall(reference, structured_text),
                          "raw_to_structure_coverage": recall(raw_text, structured_text),
                          "product_checks": checks,
                          "exact_product_rows": sum(c["row_exact"] for c in checks),
                          "product_count": len(products)})
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "ocr_quality.json").write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# OCR quality", "", "Complete documents only; cropped and failed runs excluded.", "",
             "| Document | Raw recall | Structured recall | Raw → structure coverage | Exact product rows |",
             "|---|---:|---:|---:|---:|"]
    for d in documents:
        lines.append(f"| {d['doc_id']} | {d['raw_recall']:.1%} | {d['structured_recall']:.1%} | "
                     f"{d['raw_to_structure_coverage']:.1%} | {d['exact_product_rows']}/{d['product_count']} |")
    lines += ["", "Text recall ignores case, accents and punctuation; it does not verify field placement.",
              "Exact product rows require an exact code and numerically equal quantity, unit price and total in their recognized columns.",
              "Unrecognized headers or merged cells fail conservatively; inspect the JSON before interpreting a failure as a wrong value.",
              "These checks do not certify every invoice field. No reference data is sent to OCR."]
    (results_dir / "ocr_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return documents


def normalized_text(text):
    return " ".join(tokens(text))


def score_text(reference, candidate):
    expected = Counter(tokens(reference))
    actual = Counter(tokens(candidate))
    overlap = sum((expected & actual).values())
    expected_total = sum(expected.values())
    actual_total = sum(actual.values())
    recall = overlap / expected_total if expected_total else 1.0
    precision = overlap / actual_total if actual_total else (1.0 if not expected_total else 0.0)
    return {
        "precision": precision,
        "recall": recall,
        "missing": expected - actual,
        "extra": actual - expected,
    }


def format_tokens(counter, limit=8):
    common = counter.most_common(limit)
    values = [f"`{token}`" + (f" ×{count}" if count > 1 else "") for token, count in common]
    if len(counter) > limit:
        values.append(f"… mais {len(counter) - limit} termos")
    return ", ".join(values) if values else "—"


def capture_paths(row):
    return [part.strip() for part in row.get("caminho_imagem", "").split(";") if part.strip()]


def artifact_paths(capture_path):
    parts = Path(capture_path).parts
    if not parts or parts[0] != "captura":
        return None
    relative = Path(*parts[1:])
    intermediates_dir, html_dir = result_directories(RESULTS_DIR)
    folder = intermediates_dir / relative.parent / f"{relative.stem}_{relative.suffix[1:].lower()}"
    html_folder = html_dir / relative.parent / f"{relative.stem}_{relative.suffix[1:].lower()}"
    stem = relative.stem
    return {
        "source": {"compact": folder / f"{stem}_page_001_layout.json", "markdown": folder / f"{stem}.md",
                   "full_json": folder / f"{stem}_res.json"},
        "html": {kind: html_folder / f"{stem}_page_001{suffix}" for kind, suffix in HTML_SUFFIX.items()},
    }


def read_source_text(path, variant):
    if variant == "markdown":
        return visible_text(path.read_text(encoding="utf-8", errors="replace"))
    blocks = json.loads(prepare_content(variant, path))
    return " ".join(visible_text(block["content"]) for block in blocks if block.get("content") is not None)


def read_html_stats(path):
    return parse_html(path.read_text(encoding="utf-8", errors="replace"))






def local_child(parent, name):
    if parent is None:
        return None
    return next((node for node in parent if node.tag.split("}")[-1] == name), None)


def xml_text(parent, *path):
    node = parent
    for name in path:
        node = local_child(node, name)
    return (node.text or "").strip() if node is not None and node.text else ""


def read_xml_fields(xml_path):
    root = ET.parse(xml_path).getroot()
    nfe = next((node for node in root.iter() if node.tag.split("}")[-1] == "NFe"), root)
    inf = local_child(nfe, "infNFe")
    ide = local_child(inf, "ide")
    emit = local_child(inf, "emit")
    dest = local_child(inf, "dest")
    total = local_child(local_child(inf, "total"), "ICMSTot")
    key = xml_text(root, "protNFe", "infProt", "chNFe")
    if not key and inf is not None:
        key = inf.attrib.get("Id", "").removeprefix("NFe")

    fields = []
    def add(label, value, kind="text"):
        if value:
            fields.append((label, value, kind))

    add("chave de acesso", key, "digits")
    add("CNPJ do emitente", xml_text(emit, "CNPJ"), "digits")
    add("nome do emitente", xml_text(emit, "xNome"))
    add("CNPJ/CPF do destinatário", xml_text(dest, "CNPJ") or xml_text(dest, "CPF"), "digits")
    add("nome do destinatário", xml_text(dest, "xNome"))
    add("natureza da operação", xml_text(ide, "natOp"))
    add("valor total da nota", xml_text(total, "vNF"), "money")
    if inf is not None:
        for detail in (node for node in inf if node.tag.split("}")[-1] == "det"):
            product = local_child(detail, "prod")
            add("código de produto", xml_text(product, "cProd"))
            add("descrição de produto", xml_text(product, "xProd"))
    return fields


def xml_value_present(text, value, kind):
    if kind == "digits":
        expected = re.sub(r"\D", "", value)
        return bool(expected) and expected in re.sub(r"\D", "", text)
    if kind == "money":
        try:
            cents = str(int((Decimal(value) * 100).to_integral_value()))
        except (InvalidOperation, ValueError):
            return False
        return cents in re.sub(r"\D", "", text)
    expected = normalized_text(value)
    return bool(expected) and expected in normalized_text(text)


def fmt_pct(value):
    return f"{value * 100:.1f}%"





def main():
    global RESULTS_DIR, REPORT_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--doc-id", action="append", help="avalia só estes documentos do manifesto")
    parser.add_argument("--ocr-only", action="store_true", help="avalia texto bruto, blocos e produtos sem exigir html")
    args = parser.parse_args()
    RESULTS_DIR = args.results_dir.resolve()
    intermediates_dir, html_dir = result_directories(RESULTS_DIR)
    REPORT_PATH = intermediates_dir / "relatorio_validacao.md"
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifesto não encontrado: {MANIFEST_PATH}")
    with MANIFEST_PATH.open(encoding="utf-8-sig", newline="") as file:
        manifest = list(csv.DictReader(file))
    if args.doc_id:
        unknown = set(args.doc_id) - {r["doc_id"] for r in manifest}
        if unknown:
            parser.error(f"doc_id desconhecido: {sorted(unknown)}")
        manifest = [r for r in manifest if r["doc_id"] in args.doc_id]
    documents = write_ocr_report(manifest, DATASET_DIR, intermediates_dir, artifact_paths)
    print(f"OCR: {len(documents)} documento(s) comparado(s); {intermediates_dir / 'ocr_quality.md'}")
    if args.ocr_only:
        return

    patterns = {
        "compact": "*_page_001_layout.json",
        "markdown": "*.md",
        "full_json": "*_res.json",
        "html_compact": "*_page_001_html.html",
        "html_markdown": "*_from_md_html.html",
        "html_full_json": "*_from_full_json_html.html",
    }
    counts = {key: sum(1 for _ in (html_dir if key.startswith("html_") else intermediates_dir).rglob(pattern))
              for key, pattern in patterns.items()}
    # exclui relatórios da busca por markdown do ocr
    counts["markdown"] = sum(1 for path in intermediates_dir.rglob("*.md") if path.name not in {REPORT_PATH.name, "ocr_quality.md"})
    if not any(counts.values()):
        REPORT_PATH.write_text(
            "# Validação da amostra DANFE\n\n"
            f"Manifesto: {len(manifest)} documentos; {sum(len(capture_paths(row)) for row in manifest)} imagens.\n\n"
            "Ainda não há resultados do OCR em `output/output-sintetico/`, então não foi possível comparar os três fluxos.\n\n"
            "Rode `python main.py ocr`, `python main.py html` e depois `python tests/validate_dataset.py`.\n",
            encoding="utf-8",
        )
        print(f"Relatório inicial salvo em {REPORT_PATH}")
        return

    results = []
    error_docs = defaultdict(set)
    for row in manifest:
        doc_id = row.get("doc_id", "(sem id)")
        issues = []
        pages = []
        for capture_path in capture_paths(row):
            paths = artifact_paths(capture_path)
            if paths is None:
                issues.append(("caminho de captura inválido", capture_path))
                continue
            image_path = DATASET_DIR / capture_path
            pages.append({"capture": capture_path, "image": image_path, **paths})

        any_output = any(
            path.is_file()
            for page in pages
            for collection in (page["source"], page["html"])
            for path in collection.values()
        )
        if not any_output and all(page["image"].is_file() for page in pages):
            issues.append(("ainda não processado", f"{len(pages)} imagem(ns) aguardando execução"))
        else:
            for page in pages:
                if not page["image"].is_file():
                    issues.append(("PNG ausente", page["capture"]))
                for variant in VARIANTS:
                    if not page["source"][variant].is_file():
                        issues.append((f"{VARIANTS[variant][0]} ausente", page["capture"]))
                    if not page["html"][variant].is_file():
                        issues.append((f"{VARIANTS[variant][1]} ausente", page["capture"]))

        reference_path = DATASET_DIR / row.get("caminho_html", "")
        xml_id = row.get("xml_id", "")
        xml_path = DATASET_DIR / "xml" / f"{xml_id}.xml"
        reference = parse_html(reference_path.read_text(encoding="utf-8", errors="replace")) if reference_path.is_file() else None
        if reference is None:
            issues.append(("HTML de referência ausente", row.get("caminho_html", "")))
        xml_fields = []
        if not xml_path.is_file():
            issues.append(("XML de referência ausente", f"xml/{xml_id}.xml"))
        else:
            try:
                xml_fields = read_xml_fields(xml_path)
            except (ET.ParseError, OSError) as exc:
                issues.append(("XML inválido", str(exc)))

        source_texts = {variant: [] for variant in VARIANTS}
        html_data = {
            variant: {"texts": [], "tags": Counter(), "spans": {"rowspan": Counter(), "colspan": Counter()}}
            for variant in VARIANTS
        }
        for page in pages:
            for variant in VARIANTS:
                source_path = page["source"][variant]
                if source_path.is_file():
                    try:
                        source_texts[variant].append(read_source_text(source_path, variant))
                    except (OSError, json.JSONDecodeError, TypeError, AttributeError, ET.ParseError) as exc:
                        issues.append((f"{VARIANTS[variant][0]} inválido", f"{page['capture']}: {exc}"))
                html_path = page["html"][variant]
                if html_path.is_file():
                    try:
                        stats = read_html_stats(html_path)
                        html_data[variant]["texts"].append(stats["text"])
                        html_data[variant]["tags"].update(stats["tags"])
                        for kind in ("rowspan", "colspan"):
                            html_data[variant]["spans"][kind].update(stats["spans"][kind])
                    except OSError as exc:
                        issues.append((f"{VARIANTS[variant][1]} ilegível", str(exc)))

        scores = {}
        structure = {}
        for variant in VARIANTS:
            source_complete = bool(pages) and all(page["source"][variant].is_file() for page in pages)
            html_complete = bool(pages) and all(page["html"][variant].is_file() for page in pages)
            source_text = " ".join(source_texts[variant])
            output_text = " ".join(html_data[variant]["texts"])
            scores[f"source_{variant}"] = score_text(reference["text"], source_text) if reference and source_complete else None
            scores[f"html_{variant}"] = score_text(reference["text"], output_text) if reference and html_complete else None

            structure[variant] = None
            if reference and html_complete:
                generated = html_data[variant]
                same_tags = all(generated["tags"][tag] == reference["tags"][tag] for tag in STRUCTURE_TAGS)
                same_spans = all(generated["spans"][kind] == reference["spans"][kind] for kind in ("rowspan", "colspan"))
                structure[variant] = same_tags and same_spans
                if not structure[variant]:
                    delta = [
                        f"{tag} {generated['tags'][tag]}/{reference['tags'][tag]}"
                        for tag in STRUCTURE_TAGS if generated["tags"][tag] != reference["tags"][tag]
                    ]
                    if not same_spans:
                        delta.append("rowspan/colspan divergem")
                    issues.append((f"estrutura diferente ({VARIANTS[variant][1]})", ", ".join(delta)))

            source_stage = f"source_{variant}"
            html_stage = f"html_{variant}"
            for stage, candidate in ((source_stage, source_text), (html_stage, output_text)):
                metric = scores[stage]
                if metric is None:
                    continue
                if metric["missing"]:
                    issues.append((f"texto faltando ({stage})", format_tokens(metric["missing"])))
                if metric["extra"]:
                    issues.append((f"texto extra ({stage})", format_tokens(metric["extra"])))

        xml_hits = Counter()
        xml_checks = Counter()
        if reference:
            for label, value, kind in xml_fields:
                # conta só valores presentes no html de referência
                if not xml_value_present(reference["text"], value, kind):
                    continue
                for variant in VARIANTS:
                    source_stage = f"source_{variant}"
                    html_stage = f"html_{variant}"
                    if scores[source_stage] is not None:
                        xml_checks[source_stage] += 1
                        if xml_value_present(" ".join(source_texts[variant]), value, kind):
                            xml_hits[source_stage] += 1
                    if scores[html_stage] is not None:
                        xml_checks[html_stage] += 1
                        if xml_value_present(" ".join(html_data[variant]["texts"]), value, kind):
                            xml_hits[html_stage] += 1

        for stage in list(xml_checks):
            if xml_hits[stage] < xml_checks[stage]:
                issues.append((f"valores XML ausentes ({stage})", f"{xml_checks[stage] - xml_hits[stage]} de {xml_checks[stage]}"))

        for category in {category for category, _ in issues}:
            error_docs[category].add(doc_id)
        results.append({
            "doc_id": doc_id,
            "metadata": row,
            "scores": scores,
            "structure": structure,
            "xml_hits": xml_hits,
            "xml_checks": xml_checks,
            "issues": issues,
        })

    report = [
        "# Validação dos fluxos da amostra DANFE",
        "",
        f"- Documentos no manifesto: **{len(manifest)}**",
        f"- Imagens listadas: **{sum(len(capture_paths(row)) for row in manifest)}**",
        "",
        "## Arquivos e recall textual por etapa",
        "",
        "O recall é a proporção de palavras do HTML de referência encontradas em cada etapa.",
        "",
        "| Fluxo | Arquivos | Documentos comparados | Recall médio |",
        "|---|---:|---:|---:|",
    ]
    stage_names = {}
    for variant in VARIANTS:
        stage_names[f"source_{variant}"] = VARIANTS[variant][0]
        stage_names[f"html_{variant}"] = VARIANTS[variant][1]
    for stage, name in stage_names.items():
        stage_scores = [result["scores"][stage] for result in results if result["scores"][stage] is not None]
        report.append(
            f"| {name} | {counts['compact'] if stage == 'source_compact' else counts['markdown'] if stage == 'source_markdown' else counts['full_json'] if stage == 'source_full_json' else counts['html_' + stage.removeprefix('html_')]} | "
            f"{len(stage_scores)} | {fmt_pct(sum(score['recall'] for score in stage_scores) / len(stage_scores)) if stage_scores else '—'} |"
        )

    report += ["", "## Estrutura HTML", "", "| HTML gerado | Documentos comparados | Estrutura igual ao gabarito |", "|---|---:|---:|"]
    for variant in VARIANTS:
        values = [result["structure"][variant] for result in results if result["structure"][variant] is not None]
        report.append(
            f"| {VARIANTS[variant][1]} | {len(values)} | "
            f"{fmt_pct(sum(values) / len(values)) if values else '—'} |"
        )

    report += ["", "## Valores do XML", "", "Presença lexical dos campos do XML que também aparecem no HTML de referência.", "", "| Etapa | Encontrados / conferidos |", "|---|---:|"]
    for stage, name in stage_names.items():
        checked = sum(result["xml_checks"][stage] for result in results)
        hits = sum(result["xml_hits"][stage] for result in results)
        report.append(f"| {name} | {hits}/{checked} |")

    report += ["", "## Resumo de erros", "", "| Tipo | Documentos afetados |", "|---|---:|"]
    if error_docs:
        report += [f"| {category} | {len(doc_ids)} |" for category, doc_ids in sorted(error_docs.items(), key=lambda item: (-len(item[1]), item[0]))]
    else:
        report.append("| Nenhum erro detectado | 0 |")

    for dimension, label in (("captura_dpi", "DPI"), ("aparencia", "Aparência"), ("estrutura", "Estrutura")):
        report += ["", f"## Recall por {label.lower()}", "", "| " + label + " | " + " | ".join(stage_names.values()) + " |", "|---|" + "---:|" * len(stage_names)]
        groups = defaultdict(list)
        for result in results:
            groups[result["metadata"].get(dimension, "(sem valor)")].append(result)
        for value, group in sorted(groups.items()):
            cells = [value]
            for stage in stage_names:
                values = [r["scores"][stage]["recall"] for r in group if r["scores"][stage] is not None]
                cells.append(fmt_pct(sum(values) / len(values)) if values else "—")
            report.append("| " + " | ".join(cells) + " |")

    report += ["", "## Detalhes dos documentos com diferenças", ""]
    detailed = [result for result in results if result["issues"] and not all(issue[0] == "ainda não processado" for issue in result["issues"])]
    if not detailed:
        report.append("Nenhum erro nos documentos processados.")
    else:
        for result in detailed:
            row = result["metadata"]
            report.append(f"### `{result['doc_id']}` — {row.get('captura_dpi', '?')} dpi, {row.get('aparencia', '?')}, {row.get('estrutura', '?')}")
            for stage, name in stage_names.items():
                score = result["scores"][stage]
                if score is not None:
                    report.append(f"- {name}: recall {fmt_pct(score['recall'])}, precisão {fmt_pct(score['precision'])}")
                    if score["missing"]:
                        report.append(f"  - Faltando: {format_tokens(score['missing'])}")
                    if score["extra"]:
                        report.append(f"  - Extra: {format_tokens(score['extra'])}")
            for category, detail in result["issues"]:
                if category != "ainda não processado":
                    report.append(f"- **{category}:** {detail}")
            report.append("")

    report += [
        "## Como interpretar",
        "",
        "- Os três arquivos de entrada são comparados pelo texto documental; no JSON completo, metadados do PaddleOCR são ignorados na pontuação textual.",
        "- A comparação textual ignora maiúsculas, acentos e pontuação. Diferenças no fluxo OCR indicam leitura/representação; diferenças que aparecem só no HTML apontam para a geração.",
        "- A comparação estrutural conta tabelas, linhas, células, `rowspan` e `colspan`; não compara a aparência visual em pixels.",
        "- Os campos XML são conferidos por presença textual quando também aparecem no HTML de referência; é uma checagem lexical, não uma validação da associação visual do campo.",
        "- Documentos sem arquivos de saída são contabilizados como ainda não processados, e não como erros de qualidade.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"Relatório salvo em {REPORT_PATH}")


if __name__ == "__main__":
    main()
