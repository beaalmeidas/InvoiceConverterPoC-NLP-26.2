from paddleocr import PPStructureV3
from pathlib import Path
import json

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


pipeline = PPStructureV3(
    # atualização -> testando parâmetros (orientation, unwarping) com True ativado para ver se melhorias aparecem
    # 1⁰ teste: ambos true
    # 2⁰ teste: um true outro false
    # 3⁰ teste: um false outro true
    # 4⁰ teste: ambos false (como no default)
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    enable_mkldnn=False,
    # minha cpu tava atingindo 100$ da capacidade ai reduzi o paralelismo pro codigo rodar mais levinho
    cpu_threads=2,

    # desliga outros módulos do ppstructurev3 pra otimizar o processamento
    use_formula_recognition=False,
    use_chart_recognition=False,
    use_seal_recognition=False,
    use_table_recognition=True,
)

OUTPUT_DIR.mkdir(exist_ok=True)

input_files = sorted(
    path for path in INPUT_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in SUPPORTED_INPUTS
)

input_files = input_files[:5]

if not input_files:
    raise FileNotFoundError(
        f"Nenhuma imagem ou PDF compatível foi encontrado em {INPUT_DIR}"
    )

for input_path in input_files:
    # cada tipo de arquivo ganha uma pasta própria para evitar colisões de nomes.
    relative_path = input_path.relative_to(INPUT_DIR)
    file_output_dir = (
        RESULTS_DIR
        / relative_path.parent
        / f"{input_path.stem}_{input_path.suffix[1:].lower()}"
    )
    file_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processando: {input_path.name}")

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
