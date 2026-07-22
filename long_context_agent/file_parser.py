from __future__ import annotations

import io
import posixpath
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET


MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 120 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 8_000
MAX_EXTRACTED_CHARACTERS = 20_000_000
MAX_PDF_PAGES = 3_000

TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log"}
NATIVE_OFFICE_EXTENSIONS = {".docx", ".pptx"}
LEGACY_OFFICE_EXTENSIONS = {".doc", ".ppt", ".rtf", ".odt", ".odp"}
SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS
    | NATIVE_OFFICE_EXTENSIONS
    | LEGACY_OFFICE_EXTENSIONS
    | {".pdf"}
)

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


class DocumentParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedDocument:
    name: str
    text: str
    source_format: str
    metadata: dict = field(default_factory=dict)


def _validate_size(data: bytes) -> None:
    if not data:
        raise DocumentParseError("文件内容为空。")
    if len(data) > MAX_FILE_BYTES:
        raise DocumentParseError("文件超过 20 MB 限制。")


def _limit_extracted_text(text: str) -> str:
    normalized = text.replace("\x00", "").strip()
    if not normalized:
        raise DocumentParseError("没有从文件中提取到可索引文字。")
    if len(normalized) > MAX_EXTRACTED_CHARACTERS:
        raise DocumentParseError("文件解压后的文字超过 2,000 万字符安全限制。")
    return normalized


def _open_safe_zip(data: bytes) -> zipfile.ZipFile:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise DocumentParseError("Office 文件结构无效或已经损坏。") from exc
    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        archive.close()
        raise DocumentParseError("Office 文件内部条目过多，已拒绝解析。")
    if sum(member.file_size for member in members) > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        archive.close()
        raise DocumentParseError("Office 文件解压体积超过 120 MB 安全限制。")
    return archive


def _xml_root(archive: zipfile.ZipFile, member: str) -> ET.Element:
    try:
        return ET.fromstring(archive.read(member))
    except KeyError as exc:
        raise DocumentParseError(f"Office 文件缺少必要结构：{member}") from exc
    except ET.ParseError as exc:
        raise DocumentParseError(f"Office XML 结构损坏：{member}") from exc


def _word_paragraph_text(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{{{_W_NS}}}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{{{_W_NS}}}tab":
            parts.append("\t")
        elif node.tag in {f"{{{_W_NS}}}br", f"{{{_W_NS}}}cr"}:
            parts.append("\n")
    return "".join(parts).strip()


def _heading_level_from_name(value: str) -> int | None:
    match = re.search(r"(?:heading|标题)\s*([1-6])", value, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _word_heading_styles(archive: zipfile.ZipFile) -> dict[str, int]:
    if "word/styles.xml" not in archive.namelist():
        return {}
    root = _xml_root(archive, "word/styles.xml")
    styles: dict[str, int] = {}
    for style in root.findall(f".//{{{_W_NS}}}style"):
        if style.get(f"{{{_W_NS}}}type") != "paragraph":
            continue
        style_id = style.get(f"{{{_W_NS}}}styleId", "")
        name = style.find(f"{{{_W_NS}}}name")
        label = name.get(f"{{{_W_NS}}}val", "") if name is not None else ""
        level = _heading_level_from_name(f"{style_id} {label}")
        if style_id and level:
            styles[style_id] = level
    return styles


def _word_heading_level(paragraph: ET.Element, heading_styles: dict[str, int]) -> int | None:
    outline = paragraph.find(f"{{{_W_NS}}}pPr/{{{_W_NS}}}outlineLvl")
    if outline is not None:
        try:
            level = int(outline.get(f"{{{_W_NS}}}val", "")) + 1
            if 1 <= level <= 6:
                return level
        except ValueError:
            pass
    style = paragraph.find(f"{{{_W_NS}}}pPr/{{{_W_NS}}}pStyle")
    if style is None:
        return None
    value = style.get(f"{{{_W_NS}}}val", "")
    return heading_styles.get(value) or _heading_level_from_name(value)


def parse_docx(name: str, data: bytes) -> ParsedDocument:
    with _open_safe_zip(data) as archive:
        root = _xml_root(archive, "word/document.xml")
        body = root.find(f"{{{_W_NS}}}body")
        if body is None:
            raise DocumentParseError("Word 文件缺少正文结构。")
        heading_styles = _word_heading_styles(archive)
        blocks: list[str] = []
        paragraphs = 0
        tables = 0
        for child in body:
            if child.tag == f"{{{_W_NS}}}p":
                text = _word_paragraph_text(child)
                if not text:
                    continue
                level = _word_heading_level(child, heading_styles)
                blocks.append(f"{'#' * level} {text}" if level else text)
                paragraphs += 1
            elif child.tag == f"{{{_W_NS}}}tbl":
                tables += 1
                rows: list[str] = []
                for row in child.findall(f".//{{{_W_NS}}}tr"):
                    cells = []
                    for cell in row.findall(f"{{{_W_NS}}}tc"):
                        cell_text = " ".join(
                            text
                            for text in (
                                _word_paragraph_text(paragraph)
                                for paragraph in cell.findall(f".//{{{_W_NS}}}p")
                            )
                            if text
                        )
                        cells.append(cell_text)
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    blocks.append(f"## 表格 {tables}\n\n" + "\n".join(rows))
        text = _limit_extracted_text("\n\n".join(blocks))
        return ParsedDocument(
            name=name,
            text=text,
            source_format="docx",
            metadata={"paragraphs": paragraphs, "tables": tables},
        )


def _natural_slide_key(member: str) -> tuple[int, str]:
    match = re.search(r"slide(\d+)\.xml$", member)
    return (int(match.group(1)) if match else 10**9, member)


def _drawing_text(root: ET.Element) -> list[str]:
    return [
        node.text.strip()
        for node in root.iter(f"{{{_A_NS}}}t")
        if node.text and node.text.strip()
    ]


def _notes_member_for_slide(archive: zipfile.ZipFile, slide_member: str) -> str | None:
    slide_name = posixpath.basename(slide_member)
    relationships = f"ppt/slides/_rels/{slide_name}.rels"
    if relationships not in archive.namelist():
        return None
    root = _xml_root(archive, relationships)
    for relation in root:
        if relation.get("Type", "").endswith("/notesSlide"):
            target = relation.get("Target", "")
            if target:
                return posixpath.normpath(
                    posixpath.join(posixpath.dirname(slide_member), target)
                )
    return None


def parse_pptx(name: str, data: bytes) -> ParsedDocument:
    with _open_safe_zip(data) as archive:
        slide_members = sorted(
            (
                member
                for member in archive.namelist()
                if re.fullmatch(r"ppt/slides/slide\d+\.xml", member)
            ),
            key=_natural_slide_key,
        )
        if not slide_members:
            raise DocumentParseError("PowerPoint 文件中没有找到幻灯片。")
        blocks: list[str] = []
        notes_count = 0
        for slide_number, member in enumerate(slide_members, start=1):
            root = _xml_root(archive, member)
            texts = _drawing_text(root)
            slide_block = [f"## 第 {slide_number} 页"]
            if texts:
                slide_block.append("\n".join(texts))
            notes_member = _notes_member_for_slide(archive, member)
            if notes_member is None:
                fallback_notes = f"ppt/notesSlides/notesSlide{slide_number}.xml"
                notes_member = fallback_notes if fallback_notes in archive.namelist() else None
            if notes_member and notes_member in archive.namelist():
                notes = _drawing_text(_xml_root(archive, notes_member))
                cleaned_notes = [item for item in notes if item not in {str(slide_number)}]
                if cleaned_notes:
                    notes_count += 1
                    slide_block.append("### 演讲者备注\n\n" + "\n".join(cleaned_notes))
            blocks.append("\n\n".join(slide_block))
        text = _limit_extracted_text("\n\n".join(blocks))
        return ParsedDocument(
            name=name,
            text=text,
            source_format="pptx",
            metadata={"slides": len(slide_members), "slides_with_notes": notes_count},
        )


def parse_pdf(name: str, data: bytes) -> ParsedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - installation-specific
        raise DocumentParseError("PDF 解析依赖 pypdf 未安装。") from exc
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    raise DocumentParseError("PDF 已加密，暂不支持需要密码的文件。")
            except Exception as exc:
                if isinstance(exc, DocumentParseError):
                    raise
                raise DocumentParseError("PDF 已加密，暂不支持需要密码的文件。") from exc
        if len(reader.pages) > MAX_PDF_PAGES:
            raise DocumentParseError("PDF 超过 3,000 页安全限制。")
        blocks: list[str] = []
        pages_with_text = 0
        for page_number, page in enumerate(reader.pages, start=1):
            extracted = (page.extract_text() or "").strip()
            if extracted:
                pages_with_text += 1
                blocks.append(f"## 第 {page_number} 页\n\n{extracted}")
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(f"PDF 解析失败：{type(exc).__name__}") from exc
    if not blocks:
        raise DocumentParseError("PDF 没有可提取文字，可能是扫描件；请先执行 OCR。")
    text = _limit_extracted_text("\n\n".join(blocks))
    return ParsedDocument(
        name=name,
        text=text,
        source_format="pdf",
        metadata={"pages": len(reader.pages), "pages_with_text": pages_with_text},
    )


def parse_text(name: str, data: bytes) -> ParsedDocument:
    decoded = None
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            decoded = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise DocumentParseError("文本文件必须使用 UTF-8 或 UTF-16 编码。")
    return ParsedDocument(
        name=name,
        text=_limit_extracted_text(decoded),
        source_format=Path(name).suffix.lower().lstrip("."),
        metadata={},
    )


def _convert_legacy_office(name: str, data: bytes) -> ParsedDocument:
    suffix = Path(name).suffix.lower()
    target_extension = ".pptx" if suffix in {".ppt", ".odp"} else ".docx"
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise DocumentParseError(
            f"解析 {suffix} 需要 LibreOffice；建议另存为 {target_extension} 后上传。"
        )
    with tempfile.TemporaryDirectory(prefix="context-atlas-office-") as temporary:
        temporary_path = Path(temporary)
        source = temporary_path / f"source{suffix}"
        source.write_bytes(data)
        conversion = subprocess.run(
            [
                soffice,
                "--headless",
                "--convert-to",
                target_extension.lstrip("."),
                "--outdir",
                str(temporary_path),
                str(source),
            ],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        converted = temporary_path / f"source{target_extension}"
        if conversion.returncode != 0 or not converted.is_file():
            detail = (conversion.stderr or conversion.stdout).strip()[:240]
            raise DocumentParseError(f"旧版 Office 文件转换失败。{detail}")
        if target_extension == ".pptx":
            parsed = parse_pptx(name, converted.read_bytes())
        else:
            parsed = parse_docx(name, converted.read_bytes())
        return ParsedDocument(
            name=name,
            text=parsed.text,
            source_format=suffix.lstrip("."),
            metadata={**parsed.metadata, "converted_to": target_extension.lstrip(".")},
        )


def parse_uploaded_document(name: str, data: bytes) -> ParsedDocument:
    safe_name = Path(name).name or "document"
    extension = Path(safe_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = "、".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentParseError(f"不支持 {extension or '无扩展名'} 文件。支持：{supported}")
    _validate_size(data)
    if extension in TEXT_EXTENSIONS:
        return parse_text(safe_name, data)
    if extension == ".docx":
        return parse_docx(safe_name, data)
    if extension == ".pptx":
        return parse_pptx(safe_name, data)
    if extension == ".pdf":
        return parse_pdf(safe_name, data)
    return _convert_legacy_office(safe_name, data)
