"""Compare the three OCR inputs and generated HTMLs with the DANFE gold files."""

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path
import csv
import json
import re
import unicodedata
import xml.etree.ElementTree as ET


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "input" / "amostra_diversa"
RESULTS_DIR = BASE_DIR / "output" / "output-sintetico"
MANIFEST_PATH = DATASET_DIR / "manifesto.csv"
REPORT_PATH = RESULTS_DIR / "relatorio_validacao.md"
STRUCTURE_TAGS = ("table", "tr", "td", "th")

VARIANTS = {
    "compact": {"source_name": "JSON compacto", "html_name": "HTML do JSON compacto"},
    "markdown": {"source_name": "Markdown do OCR", "html_name": "HTML do Markdown"},
    "full_json": {"source_name": "JSON completo", "html_name": "HTML do JSON completo"},
}


class VisibleHTMLParser(HTMLParser):
    """Collect visible text and table structure, ignoring CSS and scripts."""

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


def normalize_tokens(text):
    text = unicodedata.normalize("NFKD", text).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.findall(r"[a-z0-9]+", text)


def normalized_text(text):
    return " ".join(normalize_tokens(text))


def score_text(reference, candidate):
    expected = Counter(normalize_tokens(reference))
    actual = Counter(normalize_tokens(candidate))
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
    folder = RESULTS_DIR / relative.parent / f"{relative.stem}_{relative.suffix[1:].lower()}"
    stem = relative.stem
    compact = folder / f"{stem}_page_001_layout.json"
    base = folder / f"{stem}_page_001"
    return {
        "source": {
            "compact": compact,
            "markdown": folder / f"{stem}.md",
            "full_json": folder / f"{stem}_res.json",
        },
        "html": {
            "compact": folder / f"{base.name}_html.html",
            "markdown": folder / f"{base.name}_from_md_html.html",
            "full_json": folder / f"{base.name}_from_full_json_html.html",
        },
    }


def html_fragment_text(source):
    return parse_html(source)["text"]


def read_compact_text(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("blocks", data.get("parsing_res_list", []))
    contents = []
    for block in data if isinstance(data, list) else []:
        content = block.get("content", "") if isinstance(block, dict) else block
        if content is not None:
            contents.append(html_fragment_text(str(content)))
    return " ".join(contents)


def read_full_json_text(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("res", data)
    blocks = data.get("parsing_res_list", []) if isinstance(data, dict) else []
    contents = []
    for block in blocks:
        content = block.get("block_content", "")
        if content is not None:
            contents.append(html_fragment_text(str(content)))
    return " ".join(contents)


def read_source_text(path, variant):
    if variant == "compact":
        return read_compact_text(path)
    if variant == "markdown":
        return html_fragment_text(path.read_text(encoding="utf-8", errors="replace"))
    return read_full_json_text(path)


def read_html_stats(path):
    return parse_html(path.read_text(encoding="utf-8", errors="replace"))


def merge_stats(target, source):
    target["texts"].append(source["text"])
    target["tags"].update(source["tags"])
    for kind in ("rowspan", "colspan"):
        target["spans"][kind].update(source["spans"][kind])


def empty_merged_stats():
    return {"texts": [], "tags": Counter(), "spans": {"rowspan": Counter(), "colspan": Counter()}}


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


def expected_source_names(variant):
    return {"compact": "JSON compacto", "markdown": "Markdown OCR", "full_json": "JSON completo"}[variant]


def expected_html_names(variant):
    return {"compact": "HTML compacto", "markdown": "HTML do Markdown", "full_json": "HTML do JSON completo"}[variant]


def group_metrics(results, dimension):
    groups = defaultdict(list)
    for result in results:
        groups[result["metadata"].get(dimension, "(sem valor)")].append(result)
    lines = [
        f"| {dimension} | " + " | ".join(expected_source_names(v) for v in VARIANTS) + " | "
        + " | ".join(expected_html_names(v) for v in VARIANTS) + " |",
        "|---|" + "---:|" * 6,
    ]
    for value in sorted(groups):
        group = groups[value]
        cells = [str(value)]
        for stage in [f"source_{v}" for v in VARIANTS] + [f"html_{v}" for v in VARIANTS]:
            scores = [item["scores"][stage]["recall"] for item in group if item["scores"].get(stage)]
            cells.append(fmt_pct(sum(scores) / len(scores)) if scores else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifesto não encontrado: {MANIFEST_PATH}")
    with MANIFEST_PATH.open(encoding="utf-8-sig", newline="") as file:
        manifest = list(csv.DictReader(file))

    patterns = {
        "compact": "*_page_001_layout.json",
        "markdown": "*.md",
        "full_json": "*_res.json",
        "html_compact": "*_page_001_html.html",
        "html_markdown": "*_from_md_html.html",
        "html_full_json": "*_from_full_json_html.html",
    }
    counts = {key: sum(1 for _ in RESULTS_DIR.rglob(pattern)) for key, pattern in patterns.items()}
    # Não conta o relatório como Markdown do OCR.
    counts["markdown"] = sum(1 for path in RESULTS_DIR.rglob("*.md") if path.name != REPORT_PATH.name)
    if not any(counts.values()):
        REPORT_PATH.write_text(
            "# Validação da amostra DANFE\n\n"
            f"Manifesto: {len(manifest)} documentos; {sum(len(capture_paths(row)) for row in manifest)} imagens.\n\n"
            "Ainda não há resultados do OCR em `output/output-sintetico/`, então não foi possível comparar os três fluxos.\n\n"
            "Rode `python main.py`, `python src/generate_html.py` e depois `python src/validate_dataset.py`.\n",
            encoding="utf-8",
        )
        print(f"Relatório inicial salvo em {REPORT_PATH.relative_to(BASE_DIR)}")
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
                        issues.append((f"{expected_source_names(variant)} ausente", page["capture"]))
                    if not page["html"][variant].is_file():
                        issues.append((f"{expected_html_names(variant)} ausente", page["capture"]))

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
                        issues.append((f"{expected_source_names(variant)} inválido", f"{page['capture']}: {exc}"))
                html_path = page["html"][variant]
                if html_path.is_file():
                    try:
                        stats = read_html_stats(html_path)
                        html_data[variant]["texts"].append(stats["text"])
                        html_data[variant]["tags"].update(stats["tags"])
                        for kind in ("rowspan", "colspan"):
                            html_data[variant]["spans"][kind].update(stats["spans"][kind])
                    except OSError as exc:
                        issues.append((f"{expected_html_names(variant)} ilegível", str(exc)))

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
                    issues.append((f"estrutura diferente ({expected_html_names(variant)})", ", ".join(delta)))

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
                # Conta apenas valores que também aparecem no HTML de referência.
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
        stage_names[f"source_{variant}"] = expected_source_names(variant)
        stage_names[f"html_{variant}"] = expected_html_names(variant)
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
            f"| {expected_html_names(variant)} | {len(values)} | "
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
    print(f"Relatório salvo em {REPORT_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
