"""invoice ocr and html generation.

  python main.py ocr  [--input FOLDER_OR_ZIP] [--limit 1] → output/intermediarios/...
  python main.py html [--engine openrouter|ollama] → output/html/...
"""
from pathlib import Path
import argparse

import pipeline

RESULTS_DIR = pipeline.OUTPUT_DIR / "output-sintetico"
INPUT_DIR = pipeline.BASE_DIR / "input" / "amostra_diversa" / "captura"


def run_ocr(parser, args):
    args.input = args.input.resolve()
    if args.input.suffix.lower() == ".zip":  # extracts and processes the folder; outputs look like those of a folder
        args.input = pipeline.extract_zip(args.input, pipeline.OUTPUT_DIR / "entradas" / args.input.stem)
    if args.limit < 0 or args.min_width < 0 or args.det_max_side < 32 or args.cpu_threads < 1:
        parser.error("limit and min-width must be >= 0; det-max-side >= 32; cpu-threads >= 1")
    if args.file:
        files = [args.file.resolve()]
        if not files[0].is_file() or files[0].suffix.lower() not in pipeline.SUPPORTED_INPUTS:
            parser.error("--file must be an existing image or PDF")
        if not files[0].is_relative_to(args.input):
            parser.error("--file must be inside --input")
    else:
        files = pipeline.select_inputs(args.input, args.dpi, args.limit)
    if not files:
        raise FileNotFoundError(f"No image or PDF found in {args.input}")
    with pipeline.stage("ocr:load_model"):
        engine = pipeline.load_engine(args.lang, args.doc_orientation, args.textline_orientation,
                                      args.det_max_side, args.cpu_threads)
    failures = 0
    for path in files:
        print(f"Processing: {path.name}", flush=True)
        failures += pipeline.process_file(engine, path, args.input, args.output, args.min_width,
                                          args.ruled_tables)[1] != "ok"
    return failures


def run_html(parser, args):
    if args.limit < 0 or args.num_predict < 1:
        parser.error("limit must be >= 0 and num-predict > 0")
    layouts = sorted(pipeline.result_directories(args.output)[0].rglob("*_layout.json"))
    layouts = layouts[:args.limit] if args.limit else layouts
    if not layouts:
        raise FileNotFoundError(f"No compact JSON found in {args.output}. Run `main.py ocr` first.")
    logs = [pipeline.write_html(layout, kind, args.output, args.engine, args.model, args.num_predict, args.reasoning)
            for layout in layouts for kind in pipeline.HTML_SUFFIX]
    return sum(log is not None and log["status"] != "accepted" for log in logs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    ocr = commands.add_parser("ocr", help="batch ocr")
    ocr.add_argument("--input", type=Path, default=INPUT_DIR)
    ocr.add_argument("--output", type=Path, default=RESULTS_DIR)
    ocr.add_argument("--file", type=Path)
    ocr.add_argument("--limit", type=int, default=5, help="maximum files; 0 = all")
    ocr.add_argument("--lang", choices=("default", "pt"), default="pt")
    ocr.add_argument("--dpi", type=int, choices=(100, 150, 200, 300))
    ocr.add_argument("--min-width", type=int, default=1600)
    ocr.add_argument("--det-max-side", type=int, default=1280)
    ocr.add_argument("--cpu-threads", type=int, default=2)
    ocr.add_argument("--doc-orientation", action=argparse.BooleanOptionalAction, default=True)
    ocr.add_argument("--textline-orientation", action=argparse.BooleanOptionalAction, default=False)
    ocr.add_argument("--ruled-tables", action=argparse.BooleanOptionalAction, default=True)
    html = commands.add_parser("html", help="html from the ocr")
    html.add_argument("--engine", choices=("direct", *pipeline.LLM_URLS), default="direct",
                      help="direct keeps the ocr; openrouter (paid) and ollama (local) use an llm")
    html.add_argument("--model", help=f"default: {pipeline.MODELS}")
    html.add_argument("--output", "--results-dir", type=Path, default=RESULTS_DIR)
    html.add_argument("--limit", type=int, default=0, help="maximum pages; 0 = all")
    html.add_argument("--num-predict", type=int, default=16384)
    html.add_argument("--reasoning", action="store_true", help="openrouter: turns reasoning on (risk of truncation)")
    args = parser.parse_args(argv)
    failures = (run_ocr if args.command == "ocr" else run_html)(parser, args)
    pipeline.report(args.command)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
