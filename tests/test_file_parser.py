from __future__ import annotations

import io
import unittest
import zipfile

from long_context_agent.file_parser import (
    DocumentParseError,
    parse_uploaded_document,
)
from long_context_agent.web_server import ResearchApplication


def make_docx() -> bytes:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>项目报告</w:t></w:r></w:p>
    <w:p><w:r><w:t>Word中的唯一项目代号是青峦-7429。</w:t></w:r></w:p>
    <w:tbl><w:tr><w:tc><w:p><w:r><w:t>字段</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>值</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
    <w:sectPr/>
  </w:body>
</w:document>"""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
    return output.getvalue()


def make_pptx() -> bytes:
    slide_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>季度汇报</a:t></a:r></a:p><a:p><a:r><a:t>PPT中的验收口令是银杏-5813。</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
</p:sld>"""
    notes_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>需要复核预算。</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:notes>"""
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide_xml)
        archive.writestr("ppt/notesSlides/notesSlide1.xml", notes_xml)
    return output.getvalue()


def make_pdf() -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream = b"BT /F1 12 Tf 72 720 Td (PDF archive code HF-90617-Z) Tj ET"
    objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{number} 0 obj\n".encode("ascii"))
        content.extend(obj)
        content.extend(b"\nendobj\n")
    xref = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(content)


class FileParserTests(unittest.TestCase):
    def test_docx_preserves_heading_text_and_table(self) -> None:
        parsed = parse_uploaded_document("report.docx", make_docx())
        self.assertEqual(parsed.source_format, "docx")
        self.assertIn("# 项目报告", parsed.text)
        self.assertIn("青峦-7429", parsed.text)
        self.assertIn("字段 | 值", parsed.text)
        self.assertEqual(parsed.metadata["tables"], 1)

    def test_pptx_preserves_slide_number_and_speaker_notes(self) -> None:
        parsed = parse_uploaded_document("review.pptx", make_pptx())
        self.assertIn("## 第 1 页", parsed.text)
        self.assertIn("银杏-5813", parsed.text)
        self.assertIn("### 演讲者备注", parsed.text)
        self.assertEqual(parsed.metadata["slides"], 1)

    def test_pdf_preserves_page_number_and_text_layer(self) -> None:
        parsed = parse_uploaded_document("archive.pdf", make_pdf())
        self.assertIn("## 第 1 页", parsed.text)
        self.assertIn("HF-90617-Z", parsed.text)
        self.assertEqual(parsed.metadata["pages"], 1)

    def test_pdf_content_read_bypasses_llm_and_returns_extracted_text(self) -> None:
        application = ResearchApplication()
        application.load_upload("archive.pdf", make_pdf())
        result = application.answer({"mode": "live", "question": "请输出 PDF 全部内容。"})
        self.assertEqual(result["stop_reason"], "direct_document_read")
        self.assertIn("PDF archive code HF-90617-Z", result["answer"])
        self.assertEqual(result["capacity_report"]["model_calls"], 0)
        self.assertTrue(result["document_read"]["enabled"])
        self.assertFalse(result["document_read"]["has_more"])

    def test_long_document_direct_read_is_paginated(self) -> None:
        application = ResearchApplication()
        application.load_text("long.pdf", "第一页内容\n" + "长文档内容" * 4_000)
        first = application.answer({"mode": "offline", "question": "读取全文"})
        self.assertTrue(first["document_read"]["has_more"])
        second = application.answer({
            "mode": "offline",
            "question": "读取全文",
            "document_read": True,
            "read_offset": first["document_read"]["next_offset"],
        })
        self.assertGreater(second["document_read"]["start_character"], 0)
        self.assertNotEqual(first["answer"], second["answer"])

    def test_binary_upload_enters_shared_retrieval_index(self) -> None:
        application = ResearchApplication()
        document = application.load_upload("report.docx", make_docx())
        self.assertEqual(document["source_format"], "docx")
        self.assertGreater(document["chunks"], 0)
        self.assertGreater(document["sections"], 0)

    def test_rejects_unknown_or_corrupt_files(self) -> None:
        with self.assertRaises(DocumentParseError):
            parse_uploaded_document("malware.exe", b"not allowed")
        with self.assertRaises(DocumentParseError):
            parse_uploaded_document("broken.docx", b"not a zip")


if __name__ == "__main__":
    unittest.main()
