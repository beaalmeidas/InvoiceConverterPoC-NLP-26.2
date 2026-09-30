from paddleocr import PPStructureV3
from pathlib import Path
import json
import argparse

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input" / "amostra_diversa" / "captura"
OUTPUT_DIR = BASE_DIR / "output"
RESULTS_DIR = OUTPUT_DIR / "output-sintetico"
SUPPORTED_INPUTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".pdf"}

# transforma output numa saída menor e mais fácil de se usar/manipular 
# com intuito de averiguar melhoria na extração se o json tiver simplificado
def simplify_result(res):
    result = res.json
    if isinstance(result, str):
        result = json.loads(result)
    # Compatível com os dois formatos usados pelo PaddleOCR: com ou sem wrapper "res".
    result = result.get("res", result)
    blocks = result.get("parsing_res_list", [])

    # pega apenas blocos estruturais essenciais do json ao invés de todo o conteúdo
    return [
        {
            "id": i,
            "type": block.get("block_label"),
            "bbox": block.get("block_bbox"),
            "content": block.get("block_content"),
        }
        for i, block in enumerate(blocks)
    ]


def main():
    parser = argparse.ArgumentParser(description="OCR de DANFEs em lote")
    parser.add_argument("--input", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--file", type=Path, help="Uma imagem dentro da pasta de entrada")
    parser.add_argument("--limit", type=int, default=5, help="Máximo de imagens; 0 = todas")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit deve ser zero ou positivo")
    args.input = args.input.resolve()

    args.output.mkdir(parents=True, exist_ok=True)

    input_files = sorted(
        path for path in args.input.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUTS
    )

    if args.file:
        selected = args.file.resolve()
        if not selected.is_file() or selected.suffix.lower() not in SUPPORTED_INPUTS:
            parser.error("--file deve ser uma imagem ou PDF existente")
        if not selected.is_relative_to(args.input):
            parser.error("--file deve estar dentro de --input")
        input_files = [selected]
    if args.limit:
        input_files = input_files[:args.limit]

    if not input_files:
        raise FileNotFoundError(
            f"Nenhuma imagem ou PDF compatível foi encontrado em {args.input}"
        )

    pipeline = PPStructureV3(
        # Capturas sintéticas já são planas: não aplicar correção de deformação.
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        enable_mkldnn=False,
        # minha cpu tava atingindo 100$ da capacidade ai reduzi o paralelismo pro codigo rodar mais levinho
        cpu_threads=2,

        # desliga outros módulos do ppstructurev3 pra otimizar o processamento
        use_formula_recognition=False,
        use_chart_recognition=False,
        use_seal_recognition=False,
        use_table_recognition=True,
    )

    for input_path in input_files:
        # cada tipo de arquivo ganha uma pasta própria para evitar colisões de nomes.
        relative_path = input_path.relative_to(args.input.resolve())
        file_output_dir = (
            args.output
            / relative_path.parent
            / f"{input_path.stem}_{input_path.suffix[1:].lower()}"
        )
        file_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processando: {input_path.name}", flush=True)

        try:
            results = pipeline.predict(str(input_path))
            for page_number, res in enumerate(results, start=1):
                res.save_to_json(save_path=str(file_output_dir))
                res.save_to_markdown(save_path=str(file_output_dir))
                res.save_to_html(save_path=str(file_output_dir))

                # JSON compacto: um arquivo por página/documento processado.
                simplified = simplify_result(res)
                layout_path = file_output_dir / f"{input_path.stem}_page_{page_number:03d}_layout.json"
                layout_path.write_text(
                    json.dumps(simplified, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception as exc:
            # uma entrada com erro não impede o processamento dos demais arquivos.
            print(f"Falha ao processar {input_path.name}: {exc}")


if __name__ == "__main__":
    main()
