"""ocr de notas fiscais (paddleocr) → html direto ou via llm, com métricas por etapa."""
from contextlib import contextmanager
from datetime import datetime
from functools import cache
from html import escape
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

from bs4 import BeautifulSoup
import psutil

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
RUNS_DIR = BASE_DIR / "runs"
SUPPORTED_INPUTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pdf"}
MODELS = {"ollama": "qwen2.5vl:7b", "openrouter": "qwen/qwen3.8-27b:free"}
HTML_SUFFIX = {"compact": "_html.html", "markdown": "_from_md_html.html", "full_json": "_from_full_json_html.html"}

# ---------- métricas ----------

records, _active = [], []


def _sample_ram(record, stop):
    process = psutil.Process()
    while True:
        record["peak_ram_mb"] = max(record["peak_ram_mb"], process.memory_info().rss / 2**20)
        if stop.wait(.05):
            return


@contextmanager
def stage(name):
    """mede tempo, cpu, pico de ram/vram e tokens de llm de uma etapa."""
    record = {"stage": name, "peak_ram_mb": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_ram, args=(record, stop), daemon=True)
    sampler.start()
    _active.append(record)
    cpu, started = time.process_time(), time.perf_counter()
    try:
        yield record
    finally:
        _active.pop()
        stop.set()
        sampler.join()
        paddle = sys.modules.get("paddle")
        gpu = paddle and paddle.device.is_compiled_with_cuda()
        # ru_maxrss marcou 7,9 GB onde a amostragem do rss viu 1,9 GB; por isso a amostragem
        record.update(seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu,
                      peak_vram_mb=paddle.device.cuda.max_memory_allocated() / 2**20 if gpu else None)
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
        peak_ram_mb=("peak_ram_mb", "max"), peak_vram_mb=("peak_vram_mb", "max"),
        tokens_in=("tokens_in", "sum"), tokens_out=("tokens_out", "sum"), cost_usd=("cost_usd", "sum"))
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
    path.write_text(json.dumps({"label": label, "cpu_count": os.cpu_count(),
                                "summary": table.reset_index().to_dict("records"), "stages": records},
                               indent=2, default=lambda value: value.item()), encoding="utf-8")
    records.clear()
    print(f"Métricas: {path}")
    return path

# ---------- ocr ----------


def result_directories(results_dir):
    """separa intermediários e htmls finais em output/intermediarios e output/html."""
    root = Path(results_dir).resolve()
    if root.is_relative_to(OUTPUT_DIR):
        relative = root.relative_to(OUTPUT_DIR)
        if relative.parts and relative.parts[0] in {"html", "intermediarios"}:
            relative = Path(*relative.parts[1:])
        return OUTPUT_DIR / "intermediarios" / relative, OUTPUT_DIR / "html" / relative
    return root / "intermediarios", root / "html"


def input_order(path, root):
    """prioriza imagens com maior resolução."""
    first = path.relative_to(root).parts[0]
    return -int(first) if first in {"100", "150", "200", "300"} else 0, str(path.relative_to(root))


def select_inputs(root, dpi=None, limit=0):
    files = sorted((path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_INPUTS
                    and (not dpi or path.relative_to(root).parts[0] == str(dpi))),
                   key=lambda path: input_order(path, root))
    return files[:limit] if limit else files


def load_engine(lang="pt", doc_orientation=True, textline_orientation=True, unwarp=False,
                det_max_side=1280, cpu_threads=2):
    from paddleocr import PPStructureV3

    return PPStructureV3(
        lang=None if lang == "default" else lang,
        ocr_version=None if lang == "default" else "PP-OCRv5",
        use_doc_orientation_classify=doc_orientation,
        use_textline_orientation=textline_orientation,
        use_doc_unwarping=unwarp,
        enable_mkldnn=False,  # onednn falha no paddle 3.3.1 (ConvertPirAttribute2RuntimeAttribute)
        cpu_threads=cpu_threads,  # 2 foi o mais rápido no i7-1355U; mais threads ficou mais lento
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
        use_table_recognition=True,
        text_det_limit_side_len=det_max_side,
        text_det_limit_type="max",
    )


def simplify_result(res):
    """json compacto: só tipo, posição e conteúdo de cada bloco."""
    data = json.loads(res.json) if isinstance(res.json, str) else res.json
    return [{"id": i, "type": block.get("block_label"), "bbox": block.get("block_bbox"),
             "content": block.get("block_content")}
            for i, block in enumerate(data.get("res", data).get("parsing_res_list", []))]


def prepare_image(source, destination, min_width=0, crop=None):
    """recorta e amplia até min_width; sem mudança, devolve a própria origem."""
    from PIL import Image

    with Image.open(source) as image:
        original_size = image.size
        if crop:
            x1, y1, x2, y2 = crop
            if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                raise ValueError("Recorte fora dos limites da imagem")
            image = image.crop(crop)
        if image.width < min_width:
            image = image.resize((min_width, round(image.height * min_width / image.width)),
                                 Image.Resampling.LANCZOS)
        if not crop and image.size == original_size:
            return source
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(destination)
        return destination


def process_file(engine, input_path, input_root, output, min_width=1600, crop=None,
                 table_mode="structure", ruled_tables=True, unwarp=False):
    """roda o ocr em uma imagem ou pdf, salva as saídas e retorna (pasta, status)."""
    intermediates_dir, _ = result_directories(output)
    relative = input_path.relative_to(input_root)
    folder = intermediates_dir / relative.parent / f"{input_path.stem}_{input_path.suffix[1:].lower()}"
    folder.mkdir(parents=True, exist_ok=True)
    status = "ok"
    try:
        source = input_path
        if input_path.suffix.lower() != ".pdf" and (crop or min_width):
            with stage("ocr:preparar_imagem"):
                source = prepare_image(input_path, folder / "_inputs" / f"{input_path.stem}.png", min_width, crop)
        with stage("ocr:predict"):
            pages = list(engine.predict(str(source), use_wired_table_cells_trans_to_html=table_mode == "cells"))
        if not pages:
            raise ValueError("OCR não retornou páginas")
        for page, res in enumerate(pages, start=1):
            if ruled_tables:
                with stage("ocr:reparar_tabelas"):
                    repair_tables(res, source, unwarp)
            with stage("ocr:salvar"):
                res.save_to_json(save_path=str(folder))
                res.save_to_markdown(save_path=str(folder))
                res.save_to_html(save_path=str(folder))
                (folder / f"{input_path.stem}_page_{page:03d}_layout.json").write_text(
                    json.dumps(simplify_result(res), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        status = "failed"
        print(f"Falha em {input_path.name}: {exc}", flush=True)
    (folder / "ocr_run.json").write_text(json.dumps({"status": status, "crop": crop}), encoding="utf-8")
    return folder, status

# ---------- tabelas pelas bordas visíveis ----------


def _boundaries(values, tolerance):
    groups = []
    for value in sorted(values):
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [round(sum(group) / len(group)) for group in groups]


def _nearest(value, boundaries):
    return min(range(len(boundaries)), key=lambda i: abs(boundaries[i] - value))


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
            if w >= 12 and h >= 8 and area / (w*h) > .8
            and x > 0 and y > 0 and x+w < width and y+h < height]


def ruled_table(image, ocr, bbox, cells=None):
    """reconstrói a tabela pela grade visível; sem bordas confiáveis, retorna None."""
    cells = detect_cells(image) if cells is None else cells
    cells = [cell for cell in cells if _overlap(cell, bbox) > .9]
    if len(cells) < 8:
        return None
    eligible = [(i, text, box) for i, (text, box) in enumerate(zip(ocr.get("rec_texts", []), ocr.get("rec_boxes", [])))
                if text.strip() and _overlap(box, bbox) > .5]
    if not eligible:
        return None
    assigned = [[] for _ in cells]
    loose = []
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
    grid = [(_nearest(c[0], xs), _nearest(c[1], ys), _nearest(c[2], xs), _nearest(c[3], ys)) for c in cells]
    if any(left >= right or top >= bottom for left, top, right, bottom in grid):
        return None
    cuts = [y for y in range(len(ys)) if not any(top < y < bottom for _, top, _, bottom in grid)]
    bands = []
    for lo, hi in zip(cuts, cuts[1:]):
        members = [i for i, (_, top, _, bottom) in enumerate(grid) if lo <= top < bottom <= hi]
        if not members:
            continue
        signature = tuple(sorted({v for i in members for v in (grid[i][0], grid[i][2])}))
        if bands and bands[-1]["end"] == lo and bands[-1]["signature"] == signature:
            bands[-1]["members"].extend(members)
            bands[-1]["end"] = hi
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
                if x in occupied:
                    x += 1
                    continue
                if x not in starts:
                    html.append("<td></td>")
                    x += 1
                    continue
                i = starts[x]
                left, _, right, bottom = grid[i]
                colspan = columns.index(right) - columns.index(left)
                rowspan = rows.index(bottom) - rows.index(top)
                attrs = (f' colspan="{colspan}"' if colspan > 1 else "") + (f' rowspan="{rowspan}"' if rowspan > 1 else "")
                html.append(f"<td{attrs}>{'<br>'.join(escape(line[3]) for line in sorted(assigned[i]))}</td>")
                x += colspan
            html.append("</tr>")
        html.append("</tbody></table>")
        fragments.append((ys[band["start"]], "".join(html)))
    # textos sem posição segura ficam fora das células
    fragments.extend((y, f"<p>{escape(text)}</p>") for y, _, text in loose)
    return {"html": "\n".join(fragment for _, fragment in sorted(fragments, key=lambda part: part[0])),
            "cells": cells, "matched_lines": matched, "loose_line_ids": [i for _, i, _ in loose]}


def repair_tables(res, input_path, unwarp=False):
    blocks = [block for block in res["parsing_res_list"] if block.label == "table"]
    if not blocks:
        return
    import cv2

    preprocessed = res.get("doc_preprocessor_res", {})
    image = preprocessed.get("output_img")
    if image is None and not preprocessed.get("angle", 0) and not unwarp:
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
    allowed = {"rowspan", "colspan", "class"} if keep_classes else {"rowspan", "colspan"}
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
def _client(engine):
    if engine == "ollama":
        import ollama
        return ollama.Client()
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(BASE_DIR / ".env")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], max_retries=5)


def ask_llm(kind, content, engine="openrouter", model=None, num_ctx=16384, num_predict=8192, reasoning=False):
    source = ("Input: PaddleOCR Markdown with HTML tables." if kind == "markdown" else
              "Input: OCR JSON blocks. Use content as text/table markup, type as label, bbox for layout only; hide coordinates.")
    messages = [{"role": "user", "content": f"{BASE_PROMPT}{source}\n\n{content}"}]
    model = model or MODELS[engine]
    if engine == "ollama":
        response = _client(engine).chat(model=model, messages=messages, keep_alive="10m", options={
            "temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict})
        record_llm(response.get("prompt_eval_count"), response.get("eval_count"))
        truncated, text = response.get("done_reason") == "length", response["message"]["content"]
    else:
        response = _client(engine).chat.completions.create(
            model=model, messages=messages, temperature=0, max_tokens=num_predict,
            extra_body={"reasoning": {"enabled": reasoning}})  # raciocínio gasta 5–16 mil tokens e trunca
        usage = response.usage
        record_llm(usage.prompt_tokens, usage.completion_tokens, getattr(usage, "cost", 0))
        truncated, text = response.choices[0].finish_reason == "length", response.choices[0].message.content
    if truncated:
        raise ValueError("HTML truncado: aumente num-predict e num-ctx.")
    if not text:
        raise ValueError("O modelo não retornou conteúdo.")
    return text


def write_html(layout_path, kind, results_dir, engine="direct", model=None, num_ctx=16384, num_predict=8192,
               reasoning=False):
    """gera um html final; o anterior vira .previous.html e o log fica nos intermediários."""
    source = source_path(layout_path, kind)
    if not source.is_file():
        print(f"Entrada {kind} ausente: {source}")
        return None
    destination = output_path(layout_path, kind, results_dir)
    base = layout_path.with_name(destination.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.replace(base.with_suffix(".previous.html"))
    log = {"source": str(source), "engine": engine, "status": "failed"}
    with stage(f"html:{engine}:{kind}") as record:
        try:
            content = prepare_content(kind, source)
            html = (render_source(kind, content) if engine == "direct" else
                    add_presentation(ask_llm(kind, content, engine, model, num_ctx, num_predict, reasoning)))
            destination.write_text(html, encoding="utf-8")
            if engine != "direct":
                log.update(model=model or MODELS[engine], tokens_in=record["tokens_in"],
                           tokens_out=record["tokens_out"], cost_usd=record["cost_usd"])
            log["status"] = "accepted"
            print(f"HTML ({kind}): {destination}", flush=True)
        except Exception as exc:
            log["error"] = str(exc)
            print(f"Falha no fluxo {kind} para {source.name}: {exc}", flush=True)
    log["duration_seconds"] = round(record["seconds"], 3)
    base.with_suffix(".generation.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log
