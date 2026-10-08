"""compara configurações do ocr em uma nota, sem enviar gabaritos ao ocr."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BASE_DIR), str(BASE_DIR / "tests")]
from pipeline import result_directories
from validate_dataset import check_products, recall, unpack_result, visible_text, xml_products


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", default="02-1-padrao-300")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "output" / "ocr-experiments")
    parser.add_argument("--cases", nargs="+", choices=("baseline-100", "baseline-300", "pt-300", "doc-off-300", "upright-300", "table-crop-300"),
                        default=["baseline-100", "baseline-300", "pt-300", "doc-off-300", "upright-300"])
    parser.add_argument("--crop", nargs=4, type=int, help="recorte em pixels da imagem original para table-crop-300")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    dataset = BASE_DIR / "input" / "amostra_diversa"
    rows = list(csv.DictReader((dataset / "manifesto.csv").open(encoding="utf-8-sig")))
    row = next((r for r in rows if r["doc_id"] == args.doc_id), None)
    if row is None or row["captura_dpi"] != "300" or ";" in row["caminho_imagem"]:
        parser.error("Choose a single-page 300-DPI document in the manifest")
    if "table-crop-300" in args.cases and not args.crop:
        parser.error("table-crop-300 requires --crop X1 Y1 X2 Y2")
    output = args.output.resolve()
    report_dir, _ = result_directories(output)
    report_dir.mkdir(parents=True, exist_ok=True)
    original = dataset / row["caminho_imagem"]
    relative = original.relative_to(dataset / "captura")
    reference_html = (dataset / row["caminho_html"]).read_text(encoding="utf-8")
    reference = visible_text(reference_html)
    products = xml_products(dataset / "xml" / (row["xml_id"] + ".xml"))
    settings = {"baseline-100": [], "baseline-300": [], "pt-300": ["--lang", "pt"],
                "doc-off-300": ["--lang", "pt", "--no-doc-orientation"],
                "upright-300": ["--lang", "pt", "--no-doc-orientation", "--no-textline-orientation"],
                "table-crop-300": ["--lang", "pt", "--no-doc-orientation", "--no-textline-orientation", "--crop", *map(str, args.crop or [])]}
    summaries = []
    for name in args.cases:
        case_dir = output / name
        case_intermediates, _ = result_directories(case_dir)
        case_intermediates.mkdir(parents=True, exist_ok=True)
        source, input_root = original, dataset / "captura"
        if name == "baseline-100":
            from PIL import Image
            input_root = report_dir / "controlled-inputs"
            source = input_root / relative
            if not args.summarize_only:
                source.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(original) as image:
                    image.resize((round(image.width / 3), round(image.height / 3)), Image.Resampling.LANCZOS).save(source)
        command = [sys.executable, str(BASE_DIR / "main.py"), "ocr", "--input", str(input_root),
                   "--file", str(source), "--output", str(case_dir), *settings[name]]
        if not args.summarize_only:
            # evita misturar resultados novos e antigos
            if list(case_intermediates.rglob("*_res.json")):
                parser.error(f"{case_dir} already has OCR results; use a fresh --output or --summarize-only")
            metadata = {"case": name, "command": command, "started_at": datetime.now(timezone.utc).isoformat(),
                        "original_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "resolution_note": "100 DPI derived by 3x downsampling the SAME 300-DPI image"}
            started = time.monotonic()
            print(f"Running {name}", flush=True)
            env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True")
            with (case_intermediates / "run.log").open("w") as log:
                process = subprocess.run(command, cwd=BASE_DIR, env=env, stdout=log, stderr=subprocess.STDOUT)
            metadata.update(returncode=process.returncode, duration_seconds=round(time.monotonic() - started, 3))
            (case_intermediates / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            if process.returncode:
                print(f"FAILED {name}: see {case_intermediates / 'run.log'}", flush=True)
        folder = case_intermediates / relative.parent / (original.stem + "_png")
        full = folder / (original.stem + "_res.json")
        if not full.exists():
            summaries.append({"case": name, "status": "missing_or_failed"})
            continue
        run_path = case_intermediates / "experiment.json"
        run = json.loads(run_path.read_text()) if run_path.exists() else {}
        if run.get("returncode") != 0 or run.get("input_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
            summaries.append({"case": name, "status": "unverified_or_failed_run"})
            continue
        data = unpack_result(json.loads(full.read_text(encoding="utf-8")))
        raw = " ".join(data.get("overall_ocr_res", {}).get("rec_texts", []))
        html = "\n".join(str(b.get("block_content") or "") for b in data.get("parsing_res_list", []))
        checks = check_products(products, html)
        crop = name == "table-crop-300"
        summaries.append({"case": name, "status": "ok", "scope": "product table only" if crop else "full page",
                          "raw_recall": None if crop else recall(reference, raw),
                          "structured_recall": None if crop else recall(reference, visible_text(html)),
                          "raw_to_structure_coverage": recall(raw, visible_text(html)),
                          "exact_product_rows": sum(c["row_exact"] for c in checks), "product_count": len(products),
                          "product_checks": checks,
                          "duration_seconds": run.get("duration_seconds")})
        # salva os resultados após cada caso
        (report_dir / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Controlled OCR comparison", "", f"Invoice: `{args.doc_id}`; XML: `{row['xml_id']}`.", "",
             "100 DPI is downsampled from this same 300-DPI image. Other settings change one at a time.",
             "Cropped-table results are not scored against the whole page.", "",
             "| Case | Raw recall | Structured recall | Exact product rows |", "|---|---:|---:|---:|"]
    for s in summaries:
        if s["status"] != "ok":
            lines.append(f"| {s['case']} | {s['status']} | — | — |")
            continue
        pct = lambda value: "—" if value is None else f"{value:.1%}"
        lines.append(f"| {s['case']} | {pct(s['raw_recall'])} | {pct(s['structured_recall'])} | {s['exact_product_rows']}/{s['product_count']} |")
    lines += ["", "This is a single-invoice experiment, not a dataset-wide accuracy claim.",
              "Product checks require recognized column headers; inspect comparison.json for individual field results."]
    (report_dir / "comparison.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report_dir / "COMPARISON.md")


if __name__ == "__main__":
    main()
