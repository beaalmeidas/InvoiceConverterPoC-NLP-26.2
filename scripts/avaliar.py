"""avalia P1 (llm vs direto), P2 (representação), P3 (ruled_table vs paddle) e atribui erros por etapa.

  python scripts/avaliar.py                    → output/avaliacao/relatorio.md e metricas.json
  python scripts/avaliar.py --gerar MODELO     → antes, gera as 5 condições com um mesmo llm (openrouter)
"""
from collections import Counter
from decimal import Decimal
import csv
import json
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE_DIR), str(BASE_DIR / "scripts")]
from bs4 import BeautifulSoup
import pandas as pd
import pipeline
from validate_dataset import (DATASET_DIR, check_products, decimal_cell, normalized_text, read_source_text,
                              read_xml_fields, score_text, tokens, xml_products, xml_value_present)

LLM_DIR = pipeline.OUTPUT_DIR / "output-sintetico"                # 5 notas com html via openrouter
OUT = pipeline.OUTPUT_DIR / "avaliacao"
NATIVE_DIR = OUT / "paddle-nativo"                                # ocr com --no-ruled-tables
GEN_DIR = OUT / "llm"
KINDS = {"compact": "JSON blocos", "markdown": "Markdown", "full_json": "JSON completo"}
LABELS = {**KINDS, "lines": "Linhas OCR + bbox", "image": "Imagem (VLM)"}
SOURCES = {"lines": "Input: OCR text lines as JSON [{t: text, b: [x1,y1,x2,y2] pixels}]. Use b only to rebuild rows, "
                    "columns and tables; hide coordinates.",
           "image": "Input: image of a Brazilian invoice (DANFE). Transcribe every printed text exactly."}
MANIFEST = {r["doc_id"]: r for r in csv.DictReader((DATASET_DIR / "manifesto.csv").open(encoding="utf-8-sig"))}
METRICS = ["recall", "precision", "fields", "rows", "cell_f1"]
STAGES = ["ok", "ocr", "montagem", "geração"]


def body_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["head", "style", "script", "title"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def cells(html):
    """textos normalizados das células folha (sem tabela interna), sem vazias."""
    return Counter(text for cell in BeautifulSoup(html, "html.parser").find_all(["td", "th"]) if not cell.find("table")
                   if (text := normalized_text(cell.get_text(" ", strip=True))))


def gold(doc_id):
    row = MANIFEST[doc_id]
    html = (DATASET_DIR / row["caminho_html"]).read_text(encoding="utf-8", errors="replace")
    text, xml = body_text(html), DATASET_DIR / "xml" / f"{row['xml_id']}.xml"
    return {"text": text, "cells": cells(html), "products": xml_products(xml),
            "fields": [f for f in read_xml_fields(xml) if xml_value_present(text, f[1], f[2])]}


def score(g, html):
    text, found = body_text(html), cells(html)
    products = check_products(g["products"], html)
    both = sum(found.values()) + sum(g["cells"].values())
    return {**{k: score_text(g["text"], text)[k] for k in ("recall", "precision")},
            "fields": sum(xml_value_present(text, v, k) for _, v, k in g["fields"]) / max(1, len(g["fields"])),
            "rows": sum(p["row_exact"] for p in products) / max(1, len(products)),
            "cell_f1": 2 * sum((g["cells"] & found).values()) / max(1, both), "n_cells": sum(found.values()),
            "_text": text, "_products": products}


def hallucination(ocr, text):
    """tokens (e números) da saída que não existem nas linhas do ocr."""
    out, src = Counter(tokens(text)), Counter(tokens(ocr))
    nums = Counter({t: n for t, n in out.items() if any(c.isdigit() for c in t)})
    return {"aluc_tokens": sum((out - src).values()) / max(1, sum(out.values())),
            "aluc_numeros": sum((nums - src).values()) / max(1, sum(nums.values())),
            "aluc_exemplos": [t for t, _ in (out - src).most_common(6)]}


def attribute(g, lines, rep_text, before, final):
    """cada campo/produto errado no final vai para a primeira etapa que o perdeu.

    lines=None: sem etapa de ocr (vlm). before=None: sem montagem determinística (llm montou)."""
    ocr, blame = " ".join(lines or []), Counter()

    def stage(right_at_end, read_by_ocr, kept_by_assembly):
        return ("ok" if right_at_end else "ocr" if lines is not None and not read_by_ocr
                else "montagem" if not kept_by_assembly else "geração")

    for _, value, kind in g["fields"]:
        blame[stage(xml_value_present(final["_text"], value, kind), xml_value_present(ocr, value, kind),
                    xml_value_present(rep_text, value, kind))] += 1
    for i, prod in enumerate(final["_products"]):
        exp = prod["expected"]
        read = exp["code"] in ocr and all(any(decimal_cell(line) == Decimal(exp[k]) for line in lines or [])
                                          for k in ("quantity", "unit_price", "total"))
        blame[stage(prod["row_exact"], read, before is None or before["_products"][i]["row_exact"])] += 1
    return blame


def outputs(layout):
    """(motor, tipo, html, custo) de cada html desta nota: direto, llm do pipeline e llm de --gerar."""
    for kind in KINDS:
        content = pipeline.prepare_content(kind, pipeline.source_path(layout, kind))
        yield "direct", kind, pipeline.render_source(kind, content), {}
        path = pipeline.output_path(layout, kind, LLM_DIR)
        log = json.loads(layout.with_name(path.name).with_suffix(".generation.json").read_text())
        if log.get("engine") == "openrouter" and path.is_file():
            yield log["model"], kind, path.read_text(encoding="utf-8"), {
                "tokens_in": log["tokens_in"], "tokens_out": log["tokens_out"], "seconds": log["duration_seconds"]}
    for path in sorted(GEN_DIR.glob(f"*/{layout.parent.parent.name}/*.html")):
        log = json.loads(path.with_suffix(".json").read_text())
        yield log["model"], path.stem, path.read_text(encoding="utf-8"), {k: log[k] for k in ("tokens_in", "tokens_out", "seconds")}


def native_and_ruled(folder):
    """html direto do paddle puro e do mesmo ocr reparado pelas bordas."""
    import cv2

    res = json.loads(next(folder.glob("*_res.json")).read_text(encoding="utf-8"))
    blocks = [{"type": b["block_label"], "bbox": b["block_bbox"], "content": b["block_content"]}
              for b in res["parsing_res_list"]]
    native = pipeline.render_source("compact", json.dumps(blocks))
    image = cv2.imread(res["input_path"])
    detected = pipeline.detect_cells(image)
    tables = [b for b in blocks if b["type"] == "table"]
    for b in tables:
        if fixed := pipeline.ruled_table(image, res["overall_ocr_res"], b["bbox"], detected):
            b.update(content=fixed["html"], repaired=True)
    repaired = f"{sum('repaired' in b for b in tables)}/{len(tables)}"
    return native, pipeline.render_source("compact", json.dumps(blocks)), repaired


def generate(model):
    """mesmo llm para as 5 entradas; pula o que já existe. falhas ficam no .json ao lado."""
    import base64

    client = pipeline._client()
    for layout in sorted(pipeline.result_directories(LLM_DIR)[0].rglob("*_layout.json")):
        res = json.loads(pipeline.source_path(layout, "full_json").read_text(encoding="utf-8"))
        ocr = res["overall_ocr_res"]
        lines = json.dumps([{"t": t, "b": b} for t, b in zip(ocr["rec_texts"], ocr["rec_boxes"])],
                           ensure_ascii=False, separators=(",", ":"))
        image = base64.b64encode(Path(res["input_path"]).read_bytes()).decode()
        for kind in LABELS:
            dest = GEN_DIR / model.replace("/", "_") / layout.parent.parent.name / f"{kind}.html"
            if dest.is_file():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            log = {"model": model, "kind": kind}
            with pipeline.stage(f"avaliacao:{kind}") as record:
                try:
                    if kind in KINDS:
                        text = pipeline.ask_llm(kind, pipeline.prepare_content(kind, pipeline.source_path(layout, kind)), model)
                    else:
                        extra = ({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}
                                 if kind == "image" else {"type": "text", "text": lines})
                        r = client.chat.completions.create(model=model, temperature=0, max_tokens=8192, messages=[
                            {"role": "user", "content": [{"type": "text", "text": pipeline.BASE_PROMPT + SOURCES[kind]}, extra]}])
                        pipeline.record_llm(r.usage.prompt_tokens, r.usage.completion_tokens, getattr(r.usage, "cost", 0))
                        text = r.choices[0].message.content
                        if r.choices[0].finish_reason == "length" or not text:
                            raise ValueError(f"resposta truncada ou vazia ({r.choices[0].finish_reason})")
                    dest.write_text(pipeline.add_presentation(text), encoding="utf-8")
                    log["status"] = "accepted"
                except Exception as exc:
                    log.update(status="failed", error=str(exc)[:300])
            log.update({k: record[k] for k in ("tokens_in", "tokens_out", "cost_usd", "seconds")})
            dest.with_suffix(".json").write_text(json.dumps(log, indent=2), encoding="utf-8")
            print(dest, log["status"], flush=True)
    pipeline.report("avaliacao-geracao")


def pct(v):
    return "—" if pd.isna(v) else f"{100 * v:.1f}%"


def markdown(df, percent=()):
    df = df.reset_index()
    for col in percent:
        df[col] = df[col].map(pct)
    return ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns),
            *("| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False))]


def evaluate():
    rows = []
    for layout in sorted(pipeline.result_directories(LLM_DIR)[0].rglob("*_layout.json")):
        doc_id = layout.parent.parent.name
        g, lines = gold(doc_id), json.loads(pipeline.source_path(layout, "full_json").read_text())["overall_ocr_res"]["rec_texts"]
        ocr = " ".join(lines)
        raw = score(g, f"<p>{ocr}</p>")
        rows.append({"doc": doc_id, "condition": "Linhas OCR (texto bruto)", "engine": "ocr",
                     **{k: raw[k] for k in ("recall", "precision", "fields")}})
        direct = {}
        for engine, kind, html, cost in outputs(layout):
            s = score(g, html)
            if engine == "direct":
                direct[kind] = s
            rep_text = read_source_text(pipeline.source_path(layout, kind), kind) if kind in KINDS else ocr
            blame = attribute(g, None if kind == "image" else lines, rep_text, direct.get(kind), s)
            rows.append({"doc": doc_id, "condition": LABELS[kind], "engine": engine, **cost, **{k: blame[k] for k in STAGES},
                         **{k: v for k, v in s.items() if not k.startswith("_")},
                         **(hallucination(ocr, s["_text"]) if engine != "direct" and kind != "image" else {})})
    p3 = []
    for folder in sorted(p.parent for p in pipeline.result_directories(NATIVE_DIR)[0].rglob("p01_res.json")):
        doc_id = folder.parent.name
        g = gold(doc_id)
        native, ruled, repaired = native_and_ruled(folder)
        for label, html in (("Paddle nativo", native), ("ruled_table", ruled)):
            s = score(g, html)
            p3.append({"doc": doc_id, "estrutura": MANIFEST[doc_id]["estrutura"], "condition": label, "repaired": repaired,
                       "gold_cells": sum(g["cells"].values()), **{k: v for k, v in s.items() if not k.startswith("_")}})
    return pd.DataFrame(rows), pd.DataFrame(p3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, p3 = evaluate()
    (OUT / "metricas.json").write_text(json.dumps({"p1_p2": df.to_dict("records"), "p3": p3.to_dict("records")},
                                                  ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    by = df.groupby(["condition", "engine"], sort=False)
    llm = df.dropna(subset=["tokens_in"]).groupby(["condition", "engine"], sort=False)
    p3["cells_ratio"] = p3.n_cells / p3.gold_cells
    pair = p3.pivot(index=["doc", "estrutura", "repaired"], columns="condition", values=["cell_f1", "rows"])
    pair.columns = [f"{m} {c}" for m, c in pair.columns]
    md = ["# Avaliação P1–P3", "", f"Notas P1/P2: {df.doc.nunique()} · notas P3: {p3.doc.nunique()}", "",
          "## P1/P2 — médias por condição", "",
          *markdown(by[METRICS + ["aluc_tokens", "aluc_numeros"]].mean(), METRICS + ["aluc_tokens", "aluc_numeros"]),
          "", "## Atribuição de erros (campos XML + linhas de produto)", "",
          *markdown(df.dropna(subset=["ok"]).groupby(["condition", "engine"], sort=False)[STAGES].sum().astype(int)),
          "", "## Exemplos de tokens alucinados (LLM)", "",
          *(f"- `{r.doc}` / {r.condition}: {', '.join(r.aluc_exemplos) or '—'}"
            for r in df.dropna(subset=["aluc_exemplos"]).itertuples()),
          "", "## Custo LLM por chamada (médias)", "",
          *markdown(llm[["tokens_in", "tokens_out", "seconds"]].mean().round(1)),
          "", "## P3 — ruled_table vs HTML do Paddle", "",
          *markdown(p3.groupby("condition")[METRICS[:1] + METRICS[2:] + ["cells_ratio"]].mean().round({"cells_ratio": 2}), METRICS[:1] + METRICS[2:]),
          "", *markdown(pair, list(pair.columns))]
    (OUT / "relatorio.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    if "--gerar" in sys.argv:
        generate(sys.argv[sys.argv.index("--gerar") + 1])
    main()
