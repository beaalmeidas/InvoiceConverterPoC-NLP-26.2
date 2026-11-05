"""ocr de notas fiscais (paddleocr) → html direto ou via llm (openrouter), com métricas por etapa."""
from contextlib import contextmanager
from datetime import datetime
from functools import cache
from html import escape
import json
import os
from pathlib import Path
import re
import threading
import time

from bs4 import BeautifulSoup
import psutil

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
RUNS_DIR = BASE_DIR / "runs"
SUPPORTED_INPUTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pdf"}
LLM_URLS = {"openrouter": "https://openrouter.ai/api/v1", "ollama": "http://localhost:11434/v1"}
MODELS = {"openrouter": "qwen/qwen3.8-27b",  # pago; a versão :free saiu da openrouter
          "ollama": "qwen2.5vl:7b"}  # roda na gpu do colab, sem custo
VLM_MAX_SIDE = 1600  # o qwen2.5-vl gasta ~1 token por 28×28 px; a captura de 300 dpi estoura o contexto
HTML_SUFFIX = {"compact": "_html.html", "markdown": "_from_md_html.html", "full_json": "_from_full_json_html.html"}

# ---------- métricas ----------

records, _active = [], []


@contextmanager
def stage(name):
    """mede tempo, cpu, pico de ram e tokens de llm de uma etapa."""
    record = {"stage": name, "peak_ram_mb": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    stop, process = threading.Event(), psutil.Process()

    def sample():  # ru_maxrss marcou 7,9 GB onde a amostragem do rss viu 1,9 GB
        while True:
            record["peak_ram_mb"] = max(record["peak_ram_mb"], process.memory_info().rss / 2**20)
            if stop.wait(.05):
                return

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    _active.append(record)
    cpu, started = time.process_time(), time.perf_counter()
    try:
        yield record
    finally:
        _active.pop()
        stop.set()
        sampler.join()
        record.update(seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu)
        records.append(record)


def record_llm(tokens_in, tokens_out, cost=0.0):
    if _active:
        _active[-1]["tokens_in"] += tokens_in or 0
        _active[-1]["tokens_out"] += tokens_out or 0
        _active[-1]["cost_usd"] += float(cost or 0)


def summary():
    import pandas as pd

    table = pd.DataFrame(records).groupby("stage", sort=False).agg(
        calls=("seconds", "size"), seconds=("seconds", "sum"), cpu_seconds=("cpu_seconds", "sum"),
        peak_ram_mb=("peak_ram_mb", "max"), tokens_in=("tokens_in", "sum"), tokens_out=("tokens_out", "sum"),
        cost_usd=("cost_usd", "sum"))
    table.insert(2, "cpu_%", 100 * table.pop("cpu_seconds") / table.seconds)
    return table


def report(label="pipeline"):
    """imprime o resumo, salva em runs/metrics_<timestamp>.json e zera as medições."""
    if not records:
        return None
    table = summary()
    print(table.round(4).to_string())
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"metrics_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    path.write_text(json.dumps({"label": label, "cpu_count": os.cpu_count(), "stages": records,
                                "summary": table.reset_index().to_dict("records")},
                               indent=2, default=lambda value: value.item()), encoding="utf-8")
    records.clear()
    print(f"Métricas: {path}")
    return path

# ---------- ocr ----------


def extract_zip(archive_path, destination):
    """extrai um zip de notas (ignora o __MACOSX do mac) e devolve a pasta extraída."""
    import zipfile

    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination, [m for m in archive.namelist() if not m.startswith("__MACOSX")])
    return destination


def result_directories(results_dir):
    """separa intermediários e htmls finais em output/intermediarios e output/html."""
    root = Path(results_dir).resolve()
    if not root.is_relative_to(OUTPUT_DIR):
        return root / "intermediarios", root / "html"
    relative = root.relative_to(OUTPUT_DIR)
    if relative.parts and relative.parts[0] in {"html", "intermediarios"}:
        relative = Path(*relative.parts[1:])
    return OUTPUT_DIR / "intermediarios" / relative, OUTPUT_DIR / "html" / relative


def input_order(path, root):
    """prioriza imagens com maior resolução."""
    first = path.relative_to(root).parts[0]
    return -int(first) if first in {"100", "150", "200", "300"} else 0, str(path.relative_to(root))


def select_inputs(root, dpi=None, limit=0):
    files = sorted((path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_INPUTS
                    and (not dpi or path.relative_to(root).parts[0] == str(dpi))), key=lambda p: input_order(p, root))
    return files[:limit] if limit else files


# textline_orientation=False: ligado, gira por engano linhas das notas em paisagem e o ocr lê lixo
# (campos 66,7% → 88,9% na nota 02-1-cinza-300, mesmo tempo)
def load_engine(lang="pt", doc_orientation=True, textline_orientation=False, det_max_side=1280, cpu_threads=2):
    from paddleocr import PPStructureV3

    return PPStructureV3(
        lang=None if lang == "default" else lang, ocr_version=None if lang == "default" else "PP-OCRv5",
        use_doc_orientation_classify=doc_orientation, use_textline_orientation=textline_orientation,
        use_doc_unwarping=False,  # capturas sintéticas já são planas
        enable_mkldnn=False,  # onednn falha no paddle 3.3.1 (ConvertPirAttribute2RuntimeAttribute)
        cpu_threads=cpu_threads,  # 2 foi o mais rápido no i7-1355U
        use_formula_recognition=False, use_chart_recognition=False, use_seal_recognition=False,
        use_table_recognition=True, text_det_limit_side_len=det_max_side, text_det_limit_type="max")


def simplify_result(res):
    """json compacto: só tipo, posição e conteúdo de cada bloco."""
    data = json.loads(res.json) if isinstance(res.json, str) else res.json
    return [{"id": i, "type": block.get("block_label"), "bbox": block.get("block_bbox"),
             "content": block.get("block_content")}
            for i, block in enumerate(data.get("res", data).get("parsing_res_list", []))]


def prepare_image(source, destination, min_width=0):
    """amplia até min_width; imagem já larga volta como está."""
    from PIL import Image

    with Image.open(source) as image:
        if image.width >= min_width:
            return source
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.resize((min_width, round(image.height * min_width / image.width)),
                     Image.Resampling.LANCZOS).convert("RGB").save(destination)
        return destination


def process_file(engine, input_path, input_root, output, min_width=1600, ruled_tables=True):
    """roda o ocr em uma imagem ou pdf, salva as saídas e retorna (pasta, status)."""
    relative = input_path.relative_to(input_root)
    folder = result_directories(output)[0] / relative.parent / f"{input_path.stem}_{input_path.suffix[1:].lower()}"
    folder.mkdir(parents=True, exist_ok=True)
    status, started = "ok", time.perf_counter()
    try:
        source = input_path
        if input_path.suffix.lower() != ".pdf":
            with stage("ocr:preparar_imagem"):
                source = prepare_image(input_path, folder / "_inputs" / f"{input_path.stem}.png", min_width)
        with stage("ocr:predict"):
            pages = list(engine.predict(str(source)))
        if not pages:
            raise ValueError("OCR não retornou páginas")
        for page, res in enumerate(pages, start=1):
            if ruled_tables:
                with stage("ocr:reparar_tabelas"):
                    repair_tables(res, source)
            with stage("ocr:salvar"):
                res.save_to_json(save_path=str(folder))
                res.save_to_markdown(save_path=str(folder))
                res.save_to_html(save_path=str(folder))
                (folder / f"{input_path.stem}_page_{page:03d}_layout.json").write_text(
                    json.dumps(simplify_result(res), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        status = "failed"
        print(f"Falha em {input_path.name}: {exc}", flush=True)
    (folder / "ocr_run.json").write_text(json.dumps({"status": status, "seconds": round(time.perf_counter() - started, 3)}),
                                         encoding="utf-8")
    return folder, status

# ---------- tabelas pelas bordas visíveis ----------


def _boundaries(values, tolerance):
    """agrupa coordenadas próximas e devolve a média de cada grupo."""
    groups = []
    for value in sorted(values):
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [round(sum(group) / len(group)) for group in groups]


def _overlap(box, cell):
    x1, y1, x2, y2 = box
    a, b, c, d = cell
    return max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b)) / max(1, (x2-x1)*(y2-y1))


def detect_cells(image):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = gray.shape
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(20, width // 40)), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(20, height // 60), 1), np.uint8))
    borders = cv2.morphologyEx(horizontal | vertical, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(255 - borders)
    return [[int(x), int(y), int(x+w), int(y+h)] for x, y, w, h, area in stats[1:]
            if w >= 12 and h >= 8 and area / (w*h) > .8 and x > 0 and y > 0 and x+w < width and y+h < height]


def ruled_table(image, ocr, bbox, cells=None):
    """reconstrói a tabela pela grade visível; sem bordas confiáveis, retorna None."""
    cells = [cell for cell in (detect_cells(image) if cells is None else cells) if _overlap(cell, bbox) > .9]
    if len(cells) < 8:
        return None
    eligible = [(i, text, box) for i, (text, box) in enumerate(zip(ocr.get("rec_texts", []), ocr.get("rec_boxes", [])))
                if text.strip() and _overlap(box, bbox) > .5]
    if not eligible:
        return None
    assigned, loose = [[] for _ in cells], []
    for i, text, box in eligible:
        hits = sorted([(_overlap(box, cell), j) for j, cell in enumerate(cells)], reverse=True)
        if hits[0][0] >= .55 and (len(hits) == 1 or hits[1][0] < .25):
            assigned[hits[0][1]].append((box[1], box[0], i, text))
        else:
            loose.append((box[1], i, text))
    matched = len(eligible) - len(loose)
    if matched / len(eligible) < .65:
        return None

    tolerance = max(3, round(image.shape[1] / 700))
    xs = _boundaries([v for c in cells for v in (c[0], c[2])], tolerance)
    ys = _boundaries([v for c in cells for v in (c[1], c[3])], tolerance)
    nearest = lambda value, edges: min(range(len(edges)), key=lambda i: abs(edges[i] - value))
    grid = [(nearest(c[0], xs), nearest(c[1], ys), nearest(c[2], xs), nearest(c[3], ys)) for c in cells]
    if any(left >= right or top >= bottom for left, top, right, bottom in grid):
        return None
    # faixas horizontais que nenhuma célula atravessa; faixas vizinhas com as mesmas colunas viram uma tabela
    cuts = [y for y in range(len(ys)) if not any(top < y < bottom for _, top, _, bottom in grid)]
    bands = []
    for lo, hi in zip(cuts, cuts[1:]):
        members = [i for i, (_, top, _, bottom) in enumerate(grid) if lo <= top < bottom <= hi]
        if not members:
            continue
        signature = tuple(sorted({v for i in members for v in (grid[i][0], grid[i][2])}))
        if bands and bands[-1]["end"] == lo and bands[-1]["signature"] == signature:
            bands[-1].update(end=hi, members=bands[-1]["members"] + members)
        else:
            bands.append({"start": lo, "end": hi, "signature": signature, "members": members})

    fragments = []
    for band in bands:
        members = band["members"]
        if not any(assigned[i] for i in members):
            continue
        columns = list(band["signature"])
        rows = sorted({v for i in members for v in (grid[i][1], grid[i][3])})
        html = ["<table><tbody>"]
        for top in rows[:-1]:
            html.append("<tr>")
            occupied = {x for i in members if grid[i][1] < top < grid[i][3]
                        for x in range(columns.index(grid[i][0]), columns.index(grid[i][2]))}
            starts = {columns.index(grid[i][0]): i for i in members if grid[i][1] == top}
            x = 0
            while x < len(columns)-1:
                if x in occupied or x not in starts:
                    html.append("" if x in occupied else "<td></td>")
                    x += 1
                    continue
                i = starts[x]
                left, _, right, bottom = grid[i]
                colspan, rowspan = columns.index(right) - columns.index(left), rows.index(bottom) - rows.index(top)
                attrs = (f' colspan="{colspan}"' if colspan > 1 else "") + (f' rowspan="{rowspan}"' if rowspan > 1 else "")
                html.append(f"<td{attrs}>{'<br>'.join(escape(line[3]) for line in sorted(assigned[i]))}</td>")
                x += colspan
            html.append("</tr>")
        html.append("</tbody></table>")
        fragments.append((ys[band["start"]], "".join(html)))
    fragments.extend((y, f"<p>{escape(text)}</p>") for y, _, text in loose)  # textos sem posição segura
    return {"html": "\n".join(fragment for _, fragment in sorted(fragments, key=lambda part: part[0])),
            "cells": cells, "matched_lines": matched, "loose_line_ids": [i for _, i, _ in loose]}


def repair_tables(res, input_path):
    blocks = [block for block in res["parsing_res_list"] if block.label == "table"]
    if not blocks:
        return
    import cv2

    preprocessed = res.get("doc_preprocessor_res", {})
    image = preprocessed.get("output_img")
    if image is None and not preprocessed.get("angle", 0):
        image = cv2.imread(str(input_path))
    if image is None:
        return
    cells = detect_cells(image)
    for block, table in zip(blocks, res["table_res_list"]):
        if repaired := ruled_table(image, res["overall_ocr_res"], block.bbox, cells):
            block.content = table["pred_html"] = repaired["html"]
            table["cell_box_list"] = repaired["cells"]

# ---------- html ----------

BASE_PROMPT = """Return only a complete HTML document with closed tags, no Markdown.
Preserve all recognized text exactly and in order; never correct, omit or invent content.
Keep labels with values and the relative layout of fields and sections.
Preserve every table, row, td/th, rowspan and colspan; do not merge unrelated cells.
"""

MARKERS_PROMPT = """Field tagging: every value that matches a field of the CONTRACT below must carry data-field="<field>" with the exact field name.
- Put the attribute on the smallest element that holds only the value (td, th, p or span), never on a label, header or section.
- If a cell holds label and value together, wrap only the value in <span data-field="...">; the label stays outside and all text and order stay unchanged.
- Label prefixes and currency symbols stay outside the span ("Num.", "Venc.", "Valor R$", "Fone:", "BAIRRO"); the span holds only the value itself.
- One element, one field. If one text holds several fields (page "1/2", the issuer block, an installment line), wrap each part in its own span. Never nest tagged elements.
- A value broken by <br> or spread over several lines (access key, address, nature of the operation, a CNPJ cut at the line end) is wrapped by ONE span that includes the <br>; never use two spans for one value.
- Tag every CONTRACT field whose value is printed. Use only field names from the CONTRACT; never invent names or add text just to hold a tag.
- If the same value is printed in more than one place (for example invoice.number and invoice.series in the receipt stub and in the header), tag every occurrence, each inside the section where it is printed.
- Decide by the section where the value is printed: the same label can be an item (item.*, one row per product), a totals field (totals.*), or belong to issuer, recipient or carrier.
- item.*, payment.*, billing.* and volumes.* repeat: tag every row or occurrence. If the products of the note arrive as loose <p> paragraphs instead of table rows, do not tag item.* at all: a wrong tag is worse than none. A single stray fragment between otherwise intact item rows is not that case: leave the fragment untagged and keep it where it is.
- A detached digit (operation type, freight type) is tagged in place with data-field on its own element; never move it into another box.
- Never delete, shorten or reorder any text, label or cell of the input; only add data-field attributes and spans.
- Respect "only if": CNPJ/CPF of recipient and carrier is one box (count DIGITS and ignore punctuation: 11 digits is *.cpf, 14 digits is *.cnpj, empty is *.document); item CST vs CSOSN follows the column header; exit vs entry date/time follows invoice.operation_type (1 saída, 0 entrada), the digit between "0 - ENTRADA" and "1 - SAÍDA" (it may appear detached from that box as a loose <p>).
- Fields marked (NFC-e) exist only on NFC-e receipts; DANFE uses invoice.issue_date.
- Label hints (DANFE): "V. ICMS UF REMET." is totals.icms_uf_origin; "V. ICMS UF DEST." is totals.icms_uf_destination; "V. IMP. IMPORTAÇÃO" is totals.import_tax; "V. APROX. DO TRIBUTO" is totals.approximate_tax; "OUTRAS DESP." is totals.other_expenses; "VALOR DA CONFINS" and "VALOR DO COFINS" are totals.cofins. In the items table: "CS", "CST" and "CSOSN" follow the column header; "ALIQ.ICMS" is item.icms_rate; "ALIQ.IPI" is item.ipi_rate; "BC ICMS" is item.icms_base.

CONTRACT (field — description):
invoice.access_key — access key (44 digits, grouped in sets of 4)
invoice.barcode — barcode of the access key (image)
invoice.qrcode — NFC-e QR Code (image)
invoice.consult_url — access URL for key-based lookup (NFC-e)
invoice.number — invoice number
invoice.series — series
invoice.operation_nature — nature of the operation
invoice.operation_type — operation type: 0 incoming, 1 outgoing
invoice.issue_date — issue date
invoice.issue_datetime — issue date and time (NFC-e)
invoice.exit_date — exit date (only if: ide/tpNF = 1)
invoice.exit_time — exit time (only if: ide/tpNF = 1)
invoice.entry_date — entry date (only if: ide/tpNF = 0)
invoice.entry_time — entry time (only if: ide/tpNF = 0)
invoice.protocol — authorization protocol number
invoice.authorization_datetime — authorization date and time
issuer.name — issuer's legal business name
issuer.cnpj — issuer's CNPJ
issuer.state_registration — issuer's state registration
issuer.state_registration_st — tax substitute's state registration
issuer.municipal_registration — municipal registration (with ISSQN)
issuer.address — street, number, and complement
issuer.full_address — street, neighborhood, city, and state on one line (NFC-e)
issuer.neighborhood — neighborhood
issuer.zip_code — ZIP code
issuer.city — city
issuer.state — state
issuer.phone — phone number
recipient.name — recipient's name or legal business name
recipient.cpf — recipient's CPF (only if: dest/CPF is present)
recipient.cnpj — recipient's CNPJ (only if: dest/CNPJ is present)
recipient.document — recipient's document, when the invoice contains neither CPF nor CNPJ (nothing printed) (only if: neither dest/CPF nor dest/CNPJ is present)
recipient.state_registration — recipient's state registration
recipient.address — street, number, and complement
recipient.neighborhood — neighborhood
recipient.zip_code — ZIP code
recipient.city — city
recipient.state — state
recipient.phone — phone number
item.code — product code
item.description — description
item.ncm — NCM code
item.cst — ICMS tax status code (only if: emit/CRT is neither 1 nor 4)
item.csosn — ICMS tax code for Simples Nacional (only if: emit/CRT = 1 or 4)
item.origin — origin of the goods
item.cfop — CFOP code
item.unit — commercial unit
item.quantity — quantity
item.unit_price — unit price
item.total_price — total item price
item.discount — item discount
item.icms_base — ICMS tax calculation base
item.icms — ICMS amount
item.icms_rate — ICMS tax rate
item.ipi — IPI amount
item.ipi_rate — IPI tax rate
item.approximate_tax — approximate tax amount for the item
totals.icms_base — ICMS tax calculation base
totals.icms — ICMS amount
totals.icms_st_base — ICMS ST tax calculation base
totals.icms_st — ICMS ST amount
totals.import_tax — import tax
totals.icms_uf_origin — ICMS tax for the originating state
totals.icms_uf_destination — ICMS tax for the destination state
totals.fcp — FCP amount
totals.pis — PIS amount
totals.cofins — COFINS amount
totals.products — total value of products
totals.shipping — shipping cost
totals.insurance — insurance
totals.discount — discount
totals.other_expenses — other ancillary expenses
totals.additions — additional charges: shipping + insurance + other expenses (NFC-e)
totals.ipi — IPI amount
totals.approximate_tax — approximate tax amount
totals.invoice — total invoice amount
totals.items_count — number of items (NFC-e)
payment.method — payment method
payment.amount — amount paid
payment.change — change
billing.number — installment number
billing.due_date — installment due date
billing.amount — installment amount
carrier.freight_type — freight type
carrier.name — carrier's legal business name
carrier.cpf — carrier's CPF (only if: transporta/CPF is present)
carrier.cnpj — carrier's CNPJ (only if: transporta/CNPJ is present)
carrier.document — carrier's document, when neither CPF nor CNPJ is present (nothing printed) (only if: neither CPF nor CNPJ is present)
carrier.state_registration — carrier's state registration
carrier.address — carrier's address
carrier.city — carrier's city
carrier.state — carrier's state
carrier.vehicle_plate — vehicle license plate
carrier.vehicle_state — vehicle registration state
carrier.antt_code — ANTT code (RNTC)
volumes.quantity — number of packages
volumes.kind — type
volumes.brand — brand
volumes.numbering — numbering
volumes.gross_weight — gross weight
volumes.net_weight — net weight
issqn.services_total — total value of services
issqn.base — ISSQN tax calculation base
issqn.value — ISSQN amount
additional.complementary_info — complementary information
additional.fisco_info — additional information of interest to tax authorities
page.current — current page
page.total — total number of pages
issuer.logo — issuer's logo (image; none of the dataset documents contain one)

Structure tagging: besides data-field, mark the structure of the note with data-section, data-item and data-index. They only add attributes and the wrapper elements named below: text, order, tables and data-field attributes stay unchanged.

0. Root: put the whole document body inside one root element per printed sheet, chosen by document type: <div class="page"> for a DANFE (full-size sheet; one root per sheet, in order), <div class="documento"> for a DANFE Simplificado (label), <div class="areaNfce"> for an NFC-e receipt. Every printed text and image goes inside it.

1. data-section="<name>": every visible text and every barcode/QR Code image must be inside exactly one section, and sections never nest. Use the name of the box where the text is printed.
DANFE sheet (portrait or landscape):
- receipt: receipt stub (canhoto: "RECEBEMOS DE ... OS PRODUTOS", receipt date and signature)
- header: issuer identification, DANFE title box, access key, barcode, sheet number
- operation: nature of the operation and authorization protocol
- registration: state registration, state registration ST and CNPJ of the issuer
- recipient: recipient/sender box
- billing: invoice and installments (fatura, duplicatas)
- tax: tax calculation and invoice totals
- transport: carrier and transported volumes
- items: products/services table, header and all rows
- issqn: ISSQN calculation
- additional_info: additional data (complementary and fisco information)
- footer: page footer, anything below the last box
DANFE Simplificado (label): notice (warning text), title (document title), totals, protocol (authorization protocol); the other boxes reuse the names above when they match.
NFC-e: contingency (contingency warning), payment (payment methods and change), consultation (access key, QR Code, consultation URL), consumer (consumer identification), messages (messages of interest, approximate taxes); the other boxes reuse header, items and totals.
- Put data-section on the smallest existing element that holds the whole box (section, div or table). If a table holds boxes of different sections, put it on each <tr> instead, never on the table. If no element fits, wrap the box's blocks in <section data-section="...">. Never split a table, move cells or add text.
- A box may be spread over several tables and loose <p> (recipient, tax, transport and items often are): wrap all of them in one <section data-section="...">, keeping their order.
- Labels, titles and small text belong to the box they sit in. A box title printed as a <p> above a table (FATURA, CALCULO DO IMPOSTO, DADOS DO PRODUTO/SERVIÇO...) belongs to the box it heads. Choose the section by meaning, not by the physical neighbor: a loose <p> displaced by the OCR (for example "CONTROLE DO FISCO" or the operation-type digit after the header table) goes in the section it belongs to (header). Text that fits no name goes in the nearest box. Never leave text outside a section and never invent a name.
- If the same box continues after a page break, tag each part with the same name.

2. data-item="<n>": only for item.* fields. n is the product's sequence number (nItem): 1 for the first product, in printed order, over the whole note, continuing across tables and sheets. Each field reads the closest data-item above it, so put it on the element that holds the whole row of the item (the <tr>), never on a single cell. If one item is printed on several <tr> (NFC-e: a line with code and description, another with quantity and price), put the same data-item on every <tr> of it. Never on header, total, title or empty rows. Map the columns by position under the header; empty header cells are alignment columns. Empty cells of a row stay empty and untagged. A stray fragment between item rows (a loose <p> or text outside any row) is never tagged: it would be read as belonging to the item above it. If the items cannot be isolated per row (loose <p>, several items in one cell), tag no data-item and no item.* field.

3. data-index="<n>": only for billing.* and payment.* fields. n is the order of the installment (billing) or of the payment method (payment), from 1 in printed order over the whole note, counted separately for each. Put it on the smallest element that holds ALL fields of ONE occurrence and nothing from another. Vertical layout (one occurrence per row): the <tr>. Horizontal layout (several occurrences in one <tr>, one per <td>): each occurrence's own <td>, never the <tr>, which would give every occurrence the same index. Never on headers, titles, empty cells or empty rows. Never use it for items, volumes or anything else, and not on the change (payment.change), which belongs to no payment method.

Example 1 (vertical layout):
<div class="page">
<section data-section="header"><p>DANFE Nº <span data-field="invoice.number">123</span></p></section>
<table>
<tr data-section="registration"><td>INSCRIÇÃO ESTADUAL</td><td data-field="issuer.state_registration">12345</td></tr>
<tr data-section="recipient"><td>DESTINATÁRIO</td><td data-field="recipient.name">CLIENTE LTDA</td></tr>
</table>
<section data-section="billing"><table><tr><th>FATURA</th></tr>
<tr data-index="1"><td data-field="billing.number">001</td><td data-field="billing.amount">10,00</td></tr>
<tr data-index="2"><td data-field="billing.number">002</td><td data-field="billing.amount">20,00</td></tr></table></section>
<section data-section="items"><table><tr><th>CÓDIGO</th><th>DESCRIÇÃO</th></tr>
<tr data-item="1"><td data-field="item.code">001</td><td data-field="item.description">PARAFUSO</td></tr>
<tr data-item="2"><td data-field="item.code">002</td><td data-field="item.description">PORCA</td></tr></table></section>
</div>

Example 2 (horizontal installments, box spread over several tables, detached digit, stray fragment):
<div class="page">
<section data-section="header"><table><tr><td>DANFE</td><td>Nº <span data-field="invoice.number">000.428.157</span></td></tr></table><p>CONTROLE DO FISCO</p><p data-field="invoice.operation_type">1</p></section>
<section data-section="billing"><p>FATURA</p><table><tr>
<td data-index="1">Num. <span data-field="billing.number">001</span><br>Venc. <span data-field="billing.due_date">12/11/2024</span><br>Valor R$ <span data-field="billing.amount">376,54</span></td>
<td></td>
<td data-index="2">Num. <span data-field="billing.number">002</span><br>Venc. <span data-field="billing.due_date">12/12/2024</span><br>Valor R$ <span data-field="billing.amount">376,54</span></td>
</tr></table></section>
<section data-section="items"><p>DADOS DO PRODUTO/SERVIÇO</p>
<table><tr><th>CÓDIGO</th><th>DESCRIÇÃO</th><th>VLR.ICMS</th></tr></table>
<table><tr data-item="1"><td data-field="item.code">P1</td><td data-field="item.description">VODKA 1L</td><td></td></tr></table>
<p>112, 720, 00</p>
<table><tr data-item="2"><td data-field="item.code">P2</td><td data-field="item.description">REFRESCO</td><td data-field="item.icms">0,26</td></tr></table></section>
</div>"""

HTML_STYLE = """<style>
@page { size: A4 landscape; margin: 10mm; }
* { box-sizing: border-box; }
html { background: #eef1f5; }
body { margin: 24px auto; max-width: 1440px; padding: 24px;
       background: #fff; color: #202936; font: 14px/1.5 Arial, sans-serif;
       box-shadow: 0 4px 24px #18263914; }
.invoice-block + .invoice-block { margin-top: 20px; }
.table-scroll { overflow-x: auto; max-width: 100%; }
table { width: 100%; border-collapse: collapse; table-layout: auto; text-align: left; }
td, th { border: 1px solid #c4ccd5; padding: 8px 10px; vertical-align: top;
         overflow-wrap: break-word; min-width: 48px; }
th { background: #e8edf3; font-weight: 700; }
tr:nth-child(even) > td { background: #f7f9fc; }
td.numeric, th.numeric { white-space: nowrap; text-align: right; font-variant-numeric: tabular-nums; }
tr { break-inside: avoid; }
p { margin: 0 0 12px; white-space: pre-wrap; }
img { max-width: 100%; height: auto; }
@media (max-width: 600px) { body { margin: 0; padding: 12px; } }
@media print {
  html, body { background: #fff; }
  body { margin: 0; padding: 0; max-width: none; font-size: 8pt; box-shadow: none; }
  .table-scroll { overflow: visible; }
  td, th { padding: 3px; min-width: 0; }
  td.numeric, th.numeric { white-space: normal; }
}
</style>"""


def source_path(layout_path, kind):
    stem = layout_path.name.rsplit("_page_", 1)[0]
    return {"compact": layout_path, "markdown": layout_path.with_name(f"{stem}.md"),
            "full_json": layout_path.with_name(f"{stem}_res.json")}[kind]


def output_path(layout_path, kind, results_dir):
    intermediates_dir, html_dir = result_directories(results_dir)
    name = layout_path.name.removesuffix("_layout.json") + HTML_SUFFIX[kind]
    return (html_dir / layout_path.relative_to(intermediates_dir)).with_name(name)


def prepare_content(kind, path):
    content = path.read_text(encoding="utf-8")
    if kind == "markdown":
        return content
    data = json.loads(content)
    if kind == "full_json":
        data = [{"type": block.get("block_label"), "bbox": block.get("block_bbox"), "content": block.get("block_content")}
                for block in data.get("res", data).get("parsing_res_list", [])]
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def clean_fragment(content, keep_classes=False):
    """remove estilos e documentos aninhados; preserva células e mesclagens."""
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup.find_all(["head", "style", "script", "meta", "link"]):
        tag.decompose()
    for tag in soup.find_all(["html", "body"]):
        tag.unwrap()
    allowed = {"rowspan", "colspan", "data-field", "data-section", "data-item", "data-index"}
    if keep_classes:
        allowed.add("class")
    for tag in soup.find_all(True):
        tag.attrs = {key: value for key, value in tag.attrs.items() if key in allowed}
    return soup


def render_source(kind, content):
    """html direto do ocr, sem llm."""
    fragments = [content] if kind == "markdown" else [
        str(block.get("content") or "") if block.get("type") == "table"
        else f"<p>{escape(str(block.get('content') or ''))}</p>" for block in json.loads(content)]
    soup = clean_fragment("\n".join(f"<section>{part}</section>" for part in fragments))
    for section in soup.find_all("section", recursive=False):
        section["class"] = "invoice-block"
    return add_presentation(soup)


def add_presentation(html):
    """aplica o estilo sem alterar o conteúdo das células."""
    soup = html if not isinstance(html, str) else clean_fragment(
        re.sub(r"^```(?:html)?\s*|\s*```$", "", html.strip(), flags=re.IGNORECASE), keep_classes=True)
    for cell in soup.find_all(["td", "th"]):
        if re.fullmatch(r"[+−-]?[\d.,%/ -]+", cell.get_text(strip=True)):
            cell["class"] = ["numeric"]
    for table in soup.find_all("table"):
        table.wrap(soup.new_tag("div", attrs={"class": "table-scroll", "tabindex": "0", "role": "region",
                                              "aria-label": "Tabela da nota fiscal"}))
    return ('<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Nota fiscal</title>{HTML_STYLE}</head><body>{soup}</body></html>')


@cache
def _client(engine="openrouter"):
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(BASE_DIR / ".env")
    key = os.environ["OPENROUTER_API_KEY"] if engine == "openrouter" else "ollama"  # o ollama ignora a chave
    return OpenAI(base_url=LLM_URLS[engine], api_key=key, max_retries=5)


def _ask(engine, model, content, num_predict=16384, reasoning=False):
    """uma chamada de chat; content é texto ou lista de partes (texto e imagem)."""
    response = _client(engine).chat.completions.create(
        model=model, temperature=0, max_tokens=num_predict, messages=[{"role": "user", "content": content}],
        # raciocínio gasta 5–16 mil tokens e trunca; o ollama não conhece o campo
        extra_body={"reasoning": {"enabled": reasoning}} if engine == "openrouter" else None)
    record_llm(response.usage.prompt_tokens, response.usage.completion_tokens, getattr(response.usage, "cost", 0))
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("HTML truncado: aumente num-predict.")
    if not choice.message.content:
        raise ValueError("O modelo não retornou conteúdo.")
    return choice.message.content


def build_instructions(markers=True):
    return BASE_PROMPT + (MARKERS_PROMPT if markers else "") + "\n"


def build_prompt(kind, content, markers=True):
    source = ("Input: PaddleOCR Markdown with HTML tables." if kind == "markdown" else
              "Input: OCR JSON blocks. Use content as text/table markup, type as label, bbox for layout only; hide coordinates.")
    return f"{build_instructions(markers)}{source}\n\n{content}"


def ask_llm(kind, content, model=None, num_predict=16384, reasoning=False, engine="openrouter", prompt=None):
    return _ask(engine, model or MODELS[engine], prompt or build_prompt(kind, content), num_predict, reasoning)


def ask_vlm(image_path, model=None, num_predict=16384, engine="openrouter", markers=True):
    """html direto da imagem, sem ocr; a imagem vai reduzida a VLM_MAX_SIDE e em png."""
    import base64
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Imagem ilegível (pdf não é aceito pelo vlm): {image_path}")
    if (scale := VLM_MAX_SIDE / max(image.shape[:2])) < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    data = base64.b64encode(cv2.imencode(".png", image)[1]).decode()
    return _ask(engine, model or MODELS[engine], [
        {"type": "text", "text": f"{build_instructions(markers)}Input: image of a Brazilian invoice (DANFE). Transcribe every printed text exactly."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}], num_predict)


def write_html(layout_path, kind, results_dir, engine="direct", model=None, num_predict=16384, reasoning=False, markers=True):
    """gera um html final (engine direct, openrouter ou ollama); o log fica nos intermediários."""
    source = source_path(layout_path, kind)
    if not source.is_file():
        print(f"Entrada {kind} ausente: {source}")
        return None
    destination = output_path(layout_path, kind, results_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    log = {"source": str(source), "engine": engine, "status": "failed"}
    with stage(f"html:{engine}:{kind}") as record:
        try:
            content = prepare_content(kind, source)
            if engine == "direct":
                html = render_source(kind, content)
            else:
                raw = ask_llm(kind, content, model, num_predict, reasoning, engine,
                              prompt=build_prompt(kind, content, markers))
                html = add_presentation(raw)
                log["data_field_raw"] = raw.count("data-field=")
                log["data_field_final"] = html.count("data-field=")
            destination.write_text(html, encoding="utf-8")
            if engine != "direct":
                log.update(model=model or MODELS[engine], **{k: record[k] for k in ("tokens_in", "tokens_out", "cost_usd")})
            log["status"] = "accepted"
            print(f"HTML ({kind}): {destination}", flush=True)
        except Exception as exc:
            log["error"] = str(exc)
            print(f"Falha no fluxo {kind} para {source.name}: {exc}", flush=True)
    log["duration_seconds"] = round(record["seconds"], 3)
    layout_path.with_name(destination.name).with_suffix(".generation.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log
