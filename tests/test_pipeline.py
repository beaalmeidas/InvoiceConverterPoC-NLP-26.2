import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup
import cv2
import numpy as np
from PIL import Image

import main
import pipeline
from validate_dataset import check_products, decimal_cell, product_rows, source_html, validate_response

TABLE = ('<html><body><table><tr><td rowspan="2">Produto</td><td colspan="2">10,00</td></tr>'
         '<tr><td>2</td><td>20,00</td></tr></table></body></html>')


def fake_openrouter(content, finish_reason="stop"):
    response = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=40, cost=0.002))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))


class MetricsTests(unittest.TestCase):
    def test_stages_are_aggregated_with_llm_usage_and_saved(self):
        with TemporaryDirectory() as folder, patch.object(pipeline, "RUNS_DIR", Path(folder)):
            for _ in range(2):
                with pipeline.stage("llm"):
                    pipeline.record_llm(10, 5, 0.5)
            with pipeline.stage("ocr"):
                pipeline.record_llm(None, None)
            saved = json.loads(pipeline.report("teste").read_text())
        llm, ocr = saved["summary"]
        self.assertEqual((llm["calls"], llm["tokens_in"], llm["tokens_out"], llm["cost_usd"]), (2, 20, 10, 1.0))
        self.assertGreater(ocr["peak_ram_mb"], 0)
        self.assertEqual((len(saved["stages"]), pipeline.records), (3, []))

    def test_llm_usage_outside_a_stage_is_ignored(self):
        pipeline.record_llm(10, 5)
        self.assertIsNone(pipeline.report())


class OcrInputTests(unittest.TestCase):
    def test_limited_batch_prefers_high_resolution(self):
        root = Path("/dataset")
        paths = [root / f"{dpi}/nota.png" for dpi in (100, 300, 150, 200)]
        self.assertEqual(sorted(paths, key=lambda p: pipeline.input_order(p, root))[0], root / "300/nota.png")

    def test_prepare_image_enlarges_crops_and_never_reduces(self):
        with TemporaryDirectory() as folder:
            small, large, out = Path(folder, "small.png"), Path(folder, "large.png"), Path(folder, "out.png")
            Image.new("RGB", (400, 600), "white").save(small)
            Image.new("RGB", (2000, 2500)).save(large)
            original = small.read_bytes()
            self.assertEqual(Image.open(pipeline.prepare_image(small, out, 1600)).size, (1600, 2400))
            self.assertEqual(small.read_bytes(), original)
            self.assertEqual(Image.open(pipeline.prepare_image(small, out, 1600, (100, 100, 300, 200))).size,
                             (1600, 800))
            out.unlink()
            self.assertEqual(pipeline.prepare_image(large, out, 1600), large)
            self.assertFalse(out.exists())
            with self.assertRaisesRegex(ValueError, "limites"):
                pipeline.prepare_image(small, out, 1600, (-1, 0, 100, 100))

    def test_simplify_result_accepts_wrapped_json_string(self):
        block = {"block_label": "text", "block_bbox": [1, 2, 3, 4], "block_content": "Nota"}
        res = SimpleNamespace(json=json.dumps({"res": {"parsing_res_list": [block]}}))
        self.assertEqual(pipeline.simplify_result(res),
                         [{"id": 0, "type": "text", "bbox": [1, 2, 3, 4], "content": "Nota"}])

    def test_output_paths_map_to_html_and_intermediarios(self):
        intermediate, html = pipeline.result_directories(pipeline.OUTPUT_DIR / "ocr-melhoria" / "final")
        self.assertEqual(intermediate, pipeline.OUTPUT_DIR / "intermediarios/ocr-melhoria/final")
        self.assertEqual(html, pipeline.OUTPUT_DIR / "html/ocr-melhoria/final")
        self.assertEqual(pipeline.result_directories(intermediate), (intermediate, html))


class RuledTableTests(unittest.TestCase):
    def sample(self):
        image = np.full((200, 620, 3), 255, np.uint8)
        xs, ys = [10, 130, 250, 370, 490, 610], [10, 60, 110, 160]
        for x in xs:
            cv2.line(image, (x, ys[0]), (x, ys[-1]), (0, 0, 0), 2)
        for y in ys:
            cv2.line(image, (xs[0], y), (xs[-1], y), (0, 0, 0), 2)
        values = [["CÓDIGO", "DESCRIÇÃO", "QTD.", "VLR.UNIT.", "VLR.TOTAL"],
                  ["001", "A & B", "2,0000", "10,00", "20,00"],
                  ["002", "Produto 2", "3", "5,50", "16,50"]]
        boxes = [[xs[col]+10, ys[row]+10, xs[col]+90, ys[row]+30] for row in range(3) for col in range(5)]
        return image, {"rec_texts": [text for row in values for text in row], "rec_boxes": boxes}

    def test_products_keep_values_in_their_columns_and_leading_zeros(self):
        image, ocr = self.sample()
        result = pipeline.ruled_table(image, ocr, [0, 0, 620, 200])
        products = [{"code": "001", "description": "A & B", "quantity": "2", "unit_price": "10", "total": "20"},
                    {"code": "002", "description": "Produto 2", "quantity": "3", "unit_price": "5.5", "total": "16.5"}]
        self.assertTrue(all(row["row_exact"] for row in check_products(products, result["html"])))
        self.assertIn("A &amp; B", result["html"])
        self.assertEqual(result["matched_lines"], 15)

    def test_ambiguous_text_is_preserved_outside_cells(self):
        image, ocr = self.sample()
        ocr["rec_texts"].append("Dado ambíguo")
        ocr["rec_boxes"].append([100, 115, 160, 145])
        result = pipeline.ruled_table(image, ocr, [0, 0, 620, 200])
        soup = BeautifulSoup(result["html"], "html.parser")
        self.assertEqual(soup.find("p").get_text(), "Dado ambíguo")
        self.assertNotIn("Dado ambíguo", soup.find("table").get_text())
        self.assertEqual(result["loose_line_ids"], [15])

    def test_without_borders_or_text_coverage_keeps_neural_fallback(self):
        image, ocr = self.sample()
        ocr["rec_boxes"] = [[0, 170, 600, 190]] * len(ocr["rec_texts"])
        self.assertIsNone(pipeline.ruled_table(image, ocr, [0, 0, 620, 200]))
        image, ocr = self.sample()
        image[:] = 255
        self.assertIsNone(pipeline.ruled_table(image, ocr, [0, 0, 620, 200]))

    def test_visible_rowspan_is_preserved(self):
        image, ocr = self.sample()
        cv2.line(image, (12, 60), (127, 60), (255, 255, 255), 4)
        merged = BeautifulSoup(pipeline.ruled_table(image, ocr, [0, 0, 620, 200])["html"], "html.parser").find(
            "td", rowspan="2")
        self.assertIn("CÓDIGO", merged.get_text())
        self.assertIn("001", merged.get_text())


class ProductRowTests(unittest.TestCase):
    product = {"code": "001", "description": "Produto", "quantity": "2", "unit_price": "10", "total": "20"}

    def html(self, values):
        header = "<tr><td>CÓDIGO</td><td>DESCRIÇÃO</td><td>QTD.</td><td>VLR.UNIT.</td><td>VLR.TOTAL</td></tr>"
        return "<table>" + header + "<tr>" + "".join(f"<td>{v}</td>" for v in values) + "</tr></table>"

    def test_exact_numbers_in_correct_columns_only(self):
        self.assertTrue(check_products([self.product], self.html(["001", "Produto", "2,0000", "10,00", "20,00"]))[0]["row_exact"])
        swapped = check_products([self.product], self.html(["001", "Produto", "10,00", "2,00", "20,00"]))[0]
        self.assertFalse(swapped["row_exact"] or swapped["fields"]["quantity"])

    def test_same_code_cannot_satisfy_two_products(self):
        result = check_products([self.product, self.product], self.html(["001", "Produto", "2", "10", "20"]))
        self.assertEqual(sum(r["row_exact"] for r in result), 1)

    def test_code_substrings_and_merged_numbers_do_not_pass(self):
        self.assertFalse(check_products([self.product], self.html(["1001", "Produto", "2", "10", "20"]))[0]["row_exact"])
        self.assertIsNone(decimal_cell("10,00 20,00"))

    def test_nested_tables_keep_inner_product_columns(self):
        html = "<table><tr><td>Layout</td><td>" + self.html(["001", "Produto", "2", "10", "20"]) + "</td></tr></table>"
        self.assertEqual(len(product_rows(html)), 1)
        self.assertTrue(check_products([self.product], html)[0]["row_exact"])


class HtmlTests(unittest.TestCase):
    def test_json_preparation_preserves_content(self):
        blocks = [{"type": "text", "bbox": [1, 2, 30, 40], "content": "A & B\n001"},
                  {"type": "table", "bbox": [1, 40, 100, 200], "content": TABLE}]
        full = {"res": {"parsing_res_list": [{"block_label": b["type"], "block_bbox": b["bbox"], "block_content": b["content"]}
                                             for b in blocks], "overall_ocr_res": {"rec_texts": ["duplicado"]}}}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "source.json"
            for kind, data in [("compact", blocks), ("full_json", full)]:
                with self.subTest(kind=kind):
                    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                    prepared = pipeline.prepare_content(kind, path)
                    self.assertEqual(json.loads(prepared), blocks)
                    self.assertEqual(validate_response(pipeline.render_source(kind, prepared),
                                                       source_html(kind, prepared), None), 1)

    def test_multiple_tables_and_literal_text_are_preserved(self):
        content = json.dumps([{"type": "table", "content": TABLE}, {"type": "text", "content": "A & B <entrega> 001"},
                              {"type": "table", "content": TABLE}])
        html = pipeline.render_source("compact", content)
        soup = BeautifulSoup(html, "html.parser")
        self.assertEqual((len(soup.find_all("html")), len(soup.find_all("body")), len(soup.select(".table-scroll"))),
                         (1, 1, 2))
        self.assertIn("A & B <entrega> 001", soup.get_text())
        self.assertEqual(validate_response(html, source_html("compact", content), None), 1)

    def test_paddle_markdown_wrappers_do_not_override_presentation(self):
        source = f'<div style="text-align: center">{TABLE}</div>\nRodapé'
        html = pipeline.render_source("markdown", source)
        self.assertNotIn("text-align: center", html)
        self.assertEqual(validate_response(html, source, None), 1)

    def test_validation_rejects_changed_spans_values_and_truncation(self):
        html = pipeline.render_source("markdown", TABLE)
        with self.assertRaisesRegex(ValueError, "mesclagens"):
            validate_response(html.replace('rowspan="2"', 'rowspan="3"'), TABLE, None)
        for altered in (html.replace("10,00", "20,00", 1).replace("20,00</td></tr>", "10,00</td></tr>"),
                        html.replace("10,00", "10.00"), html.replace("</body>", "<p>Inventado</p></body>"),
                        "<html><body><table><tr><td>Produto"):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                validate_response(altered, TABLE, None)
        with self.assertRaisesRegex(ValueError, "limite"):
            validate_response(html, TABLE, "length")

    def test_text_only_document(self):
        content = json.dumps([{"type": "text", "content": "Nota sem tabela: 001"}])
        self.assertEqual(validate_response(pipeline.render_source("compact", content),
                                           source_html("compact", content), None), 1)


class OpenRouterTests(unittest.TestCase):
    def test_response_and_usage_are_recorded(self):
        with patch.object(pipeline, "_client", lambda engine: fake_openrouter("```html\n<p>Nota</p>\n```")), \
             pipeline.stage("llm") as record:
            html = pipeline.ask_llm("compact", "[]")
        pipeline.records.clear()
        self.assertIn("<p>Nota</p>", pipeline.add_presentation(html))
        self.assertEqual((record["tokens_in"], record["tokens_out"], record["cost_usd"]), (100, 40, 0.002))

    def test_truncated_or_empty_response_fails(self):
        for content, reason in (("<p>Nota", "length"), (None, "stop")):
            with self.subTest(reason=reason), self.assertRaises(ValueError), \
                 patch.object(pipeline, "_client", lambda engine: fake_openrouter(content, reason)):
                pipeline.ask_llm("compact", "[]")


class EndToEndTests(unittest.TestCase):
    def test_cli_keeps_html_separate_from_logs_and_backups(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            inputs, outputs = root / "input", root / "output"
            inputs.mkdir()
            Image.new("RGB", (100, 100)).save(inputs / "p01.png")
            table = "<table><tr><td>Produto</td><td>10,00</td></tr></table>"
            data = {"parsing_res_list": [{"block_label": "table", "block_bbox": [0, 0, 100, 100],
                                          "block_content": table}], "overall_ocr_res": {}}

            class Result:
                json = data

                def save_to_json(self, save_path):
                    (Path(save_path) / "p01_res.json").write_text(json.dumps(data))

                def save_to_markdown(self, save_path):
                    (Path(save_path) / "p01.md").write_text(table)

                def save_to_html(self, save_path):
                    (Path(save_path) / "p01_table_1.html").write_text(table)

            engine = SimpleNamespace(predict=lambda *args, **kwargs: [Result()])
            with patch.dict(sys.modules, {"paddleocr": SimpleNamespace(PPStructureV3=lambda **kwargs: engine)}), \
                 patch.object(pipeline, "RUNS_DIR", root / "runs"):
                main.main(["ocr", "--input", str(inputs), "--output", str(outputs), "--min-width", "0",
                           "--no-ruled-tables"])
                main.main(["html", "--output", str(outputs)])
                with patch.object(pipeline, "_client", lambda engine: fake_openrouter(table)):
                    main.main(["html", "--output", str(outputs), "--engine", "openrouter"])
            intermediate, html = pipeline.result_directories(outputs)
            self.assertEqual(len(list(html.rglob("*.html"))), 3)
            self.assertFalse(list(html.rglob("*.json")))
            self.assertEqual(len(list(intermediate.rglob("*.previous.html"))), 3)
            log = json.loads((intermediate / "p01_png/p01_page_001_html.generation.json").read_text())
            self.assertEqual((log["status"], log["model"], log["tokens_in"]), ("accepted", pipeline.MODELS["openrouter"], 100))
            self.assertTrue((intermediate / "p01_png/ocr_run.json").exists())
            self.assertEqual(len(list((root / "runs").glob("metrics_*.json"))), 3)


if __name__ == "__main__":
    unittest.main()
