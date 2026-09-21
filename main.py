from paddleocr import PPStructureV3
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# transforma output numa saída menor e mais fácil de se usar/manipular 
# com intuito de averiguar melhoria na extração se o json tiver simplificado
def simplify_result(res):
    result = res.json

    blocks = result["res"].get("parsing_res_list", [])

    # pega apenas blocos estruturais essenciais do json ao invés de todo o conteúdo
    return [
        {
            "type": block.get("block_label"),
            "bbox": block.get("block_bbox"),
            "content": block.get("block_content"),
        }
        for block in blocks
    ]


pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
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

output = pipeline.predict(str(BASE_DIR / "input" / "img_1.jpg"))

OUTPUT_DIR.mkdir(exist_ok=True)

for i, res in enumerate(output):
    res.save_to_json(save_path=str(OUTPUT_DIR))
    res.save_to_markdown(save_path=str(OUTPUT_DIR))
    res.save_to_html(save_path=str(OUTPUT_DIR))

    # versão limpa pra testar e ver se há ganho de desempenho
    simplified = simplify_result(res)

    with open(OUTPUT_DIR / f"img_{i}_layout.json", "w",encoding="utf-8"
    ) as f:
        json.dump(simplified, f, ensure_ascii=False, indent=2)
