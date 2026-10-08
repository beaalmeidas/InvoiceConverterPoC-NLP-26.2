"""ocr de notas fiscais e geração de html.

  python main.py ocr  [--limit 1]               → output/intermediarios/...
  python main.py html [--engine openrouter]     → output/html/...
"""
from pathlib import Path
import argparse

import pipeline

RESULTS_DIR = pipeline.OUTPUT_DIR / "output-sintetico"
INPUT_DIR = pipeline.BASE_DIR / "input" / "amostra_diversa" / "captura"


def run_ocr(parser, args):
    args.input = args.input.resolve()
    if args.limit < 0 or args.min_width < 0 or args.det_max_side < 32 or args.cpu_threads < 1:
        parser.error("limit e min-width devem ser >= 0; det-max-side >= 32; cpu-threads >= 1")
    if args.crop and (not args.file or args.file.suffix.lower() == ".pdf"):
        parser.error("--crop exige --file com uma imagem rasterizada")
    if args.file:
        files = [args.file.resolve()]
        if not files[0].is_file() or files[0].suffix.lower() not in pipeline.SUPPORTED_INPUTS:
            parser.error("--file deve ser uma imagem ou PDF existente")
        if not files[0].is_relative_to(args.input):
            parser.error("--file deve estar dentro de --input")
    else:
        files = pipeline.select_inputs(args.input, args.dpi, args.limit)
    if not files:
        raise FileNotFoundError(f"Nenhuma imagem ou PDF encontrado em {args.input}")
    with pipeline.stage("ocr:carregar_modelo"):
        engine = pipeline.load_engine(args.lang, args.doc_orientation, args.textline_orientation, args.unwarp,
                                      args.det_max_side, args.cpu_threads)
    failures = 0
    for path in files:
        print(f"Processando: {path.name}", flush=True)
        failures += pipeline.process_file(engine, path, args.input, args.output, args.min_width, args.crop,
                                          args.table_mode, args.ruled_tables, args.unwarp)[1] != "ok"
    return failures


def run_html(parser, args):
    if args.limit < 0 or args.num_predict < 1 or args.num_ctx <= args.num_predict:
        parser.error("Limite deve ser >= 0 e num-ctx deve ser maior que num-predict > 0")
    layouts = sorted(pipeline.result_directories(args.output)[0].rglob("*_layout.json"))
    layouts = layouts[:args.limit] if args.limit else layouts
    if not layouts:
        raise FileNotFoundError(f"Nenhum JSON compacto encontrado em {args.output}. Rode `main.py ocr` primeiro.")
    logs = [pipeline.write_html(layout, kind, args.output, args.engine, args.model, args.num_ctx, args.num_predict,
                                args.reasoning) for layout in layouts for kind in pipeline.HTML_SUFFIX]
    return sum(log is not None and log["status"] != "accepted" for log in logs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    ocr = commands.add_parser("ocr", help="ocr em lote")
    ocr.add_argument("--input", type=Path, default=INPUT_DIR)
    ocr.add_argument("--output", type=Path, default=RESULTS_DIR)
    ocr.add_argument("--file", type=Path)
    ocr.add_argument("--limit", type=int, default=5, help="máximo de arquivos; 0 = todos")
    ocr.add_argument("--lang", choices=("default", "pt"), default="pt")
    ocr.add_argument("--dpi", type=int, choices=(100, 150, 200, 300))
    ocr.add_argument("--min-width", type=int, default=1600)
    ocr.add_argument("--det-max-side", type=int, default=1280)
    ocr.add_argument("--cpu-threads", type=int, default=2)
    ocr.add_argument("--crop", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"))
    ocr.add_argument("--doc-orientation", action=argparse.BooleanOptionalAction, default=True)
    ocr.add_argument("--textline-orientation", action=argparse.BooleanOptionalAction, default=True)
    ocr.add_argument("--unwarp", action=argparse.BooleanOptionalAction, default=False,
                     help="capturas sintéticas já são planas")
    ocr.add_argument("--table-mode", choices=("structure", "cells"), default="structure")
    ocr.add_argument("--ruled-tables", action=argparse.BooleanOptionalAction, default=True)
    html = commands.add_parser("html", help="html a partir do ocr")
    html.add_argument("--engine", choices=("direct", "ollama", "openrouter"), default="direct",
                      help="direct preserva o ocr; ollama e openrouter usam um llm")
    html.add_argument("--model", help=f"padrão: {pipeline.MODELS}")
    html.add_argument("--output", "--results-dir", type=Path, default=RESULTS_DIR)
    html.add_argument("--limit", type=int, default=0, help="máximo de páginas; 0 = todas")
    html.add_argument("--num-ctx", type=int, default=16384)
    html.add_argument("--num-predict", type=int, default=8192)
    html.add_argument("--reasoning", action="store_true", help="openrouter: liga o raciocínio (risco de truncar)")
    args = parser.parse_args(argv)
    failures = (run_ocr if args.command == "ocr" else run_html)(parser, args)
    pipeline.report(args.command)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
