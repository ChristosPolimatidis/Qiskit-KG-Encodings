from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap


SOURCE_PATH = Path("docs/manual.md")
OUTPUT_PATH = Path("docs/manual.pdf")

PAGE_WIDTH = 595
PAGE_HEIGHT = 842
LEFT_MARGIN = 54
RIGHT_MARGIN = 54
TOP_MARGIN = 64
BOTTOM_MARGIN = 56


@dataclass(slots=True)
class StyledLine:
    text: str
    font: str
    size: int
    indent: int = 0
    gap_before: int = 0
    gap_after: int = 0


def markdown_to_styled_lines(markdown_text: str) -> list[StyledLine]:
    """Convert a small markdown subset into styled lines."""

    max_chars = 78
    lines: list[StyledLine] = []

    for raw_line in markdown_text.splitlines():
        stripped = raw_line.strip()

        if not stripped:
            lines.append(StyledLine("", "F1", 11, gap_after=6))
            continue

        if stripped.startswith("# "):
            title = stripped[2:].strip()
            lines.append(StyledLine(title, "F2", 22, gap_before=4, gap_after=8))
            continue

        if stripped.startswith("## "):
            subtitle = stripped[3:].strip()
            lines.append(StyledLine(subtitle, "F2", 15, gap_before=8, gap_after=4))
            continue

        if stripped.startswith("- "):
            bullet_text = f"- {stripped[2:].strip()}"
            wrapped = textwrap.wrap(
                bullet_text,
                width=max_chars - 4,
                subsequent_indent="  ",
            )
            for item in wrapped:
                lines.append(StyledLine(item, "F1", 11, indent=14))
            lines.append(StyledLine("", "F1", 11, gap_after=2))
            continue

        if stripped.startswith("`") and stripped.endswith("`"):
            code_text = stripped.strip("`")
            wrapped = textwrap.wrap(code_text, width=max_chars - 2)
            for item in wrapped:
                lines.append(StyledLine(item, "F3", 10, indent=10))
            lines.append(StyledLine("", "F1", 11, gap_after=2))
            continue

        if stripped[:2].isdigit() and stripped[1:3] == ". ":
            wrapped = textwrap.wrap(
                stripped,
                width=max_chars - 2,
                subsequent_indent="   ",
            )
            for item in wrapped:
                lines.append(StyledLine(item, "F1", 11, indent=6))
            lines.append(StyledLine("", "F1", 11, gap_after=2))
            continue

        paragraph = stripped.replace("`", "")
        wrapped = textwrap.wrap(paragraph, width=max_chars)
        for item in wrapped:
            lines.append(StyledLine(item, "F1", 11))
        lines.append(StyledLine("", "F1", 11, gap_after=4))

    return lines


def estimate_line_height(styled_line: StyledLine) -> int:
    """Return a simple line height estimate."""

    if styled_line.text == "":
        return styled_line.gap_before + styled_line.gap_after
    return styled_line.size + 4 + styled_line.gap_before + styled_line.gap_after


def paginate_lines(styled_lines: list[StyledLine]) -> list[list[StyledLine]]:
    """Split styled lines into pages."""

    pages: list[list[StyledLine]] = []
    current_page: list[StyledLine] = []
    available_height = PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN
    used_height = 0

    for styled_line in styled_lines:
        line_height = estimate_line_height(styled_line)
        if current_page and used_height + line_height > available_height:
            pages.append(current_page)
            current_page = []
            used_height = 0

        current_page.append(styled_line)
        used_height += line_height

    if current_page:
        pages.append(current_page)

    return pages


def pdf_escape(text: str) -> str:
    """Escape text for use inside a PDF string."""

    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def build_page_stream(page_lines: list[StyledLine]) -> bytes:
    """Build the PDF content stream for one page."""

    y = PAGE_HEIGHT - TOP_MARGIN
    commands: list[str] = []

    for styled_line in page_lines:
        y -= styled_line.gap_before

        if styled_line.text:
            x = LEFT_MARGIN + styled_line.indent
            font_name = styled_line.font
            font_size = styled_line.size
            escaped_text = pdf_escape(styled_line.text)
            commands.append(
                f"BT /{font_name} {font_size} Tf 1 0 0 1 {x} {y} Tm ({escaped_text}) Tj ET"
            )
            y -= font_size + 4

        y -= styled_line.gap_after

    return "\n".join(commands).encode("latin-1", errors="replace")


def build_pdf(page_streams: list[bytes]) -> bytes:
    """Assemble a minimal PDF with built-in fonts."""

    objects: list[bytes] = []

    def add_object(content: bytes) -> int:
        objects.append(content)
        return len(objects)

    catalog_id = add_object(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = add_object(b"<< /Type /Pages /Kids [] /Count 0 >>")
    font_regular_id = add_object(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )
    font_bold_id = add_object(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"
    )
    font_mono_id = add_object(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"
    )

    page_ids: list[int] = []

    for page_stream in page_streams:
        content_id = add_object(
            f"<< /Length {len(page_stream)} >>\nstream\n".encode("latin-1")
            + page_stream
            + b"\nendstream"
        )
        page_id = add_object(
            (
                "<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Contents {content_id} 0 R "
                f"/Resources << /Font << /F1 {font_regular_id} 0 R "
                f"/F2 {font_bold_id} 0 R /F3 {font_mono_id} 0 R >> >> >>"
            ).encode("latin-1")
        )
        page_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("latin-1")
    )
    objects[catalog_id - 1] = b"<< /Type /Catalog /Pages 2 0 R >>"

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]

    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")

    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    pdf.extend(
        (
            "trailer\n"
            f"<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            "startxref\n"
            f"{xref_start}\n"
            "%%EOF\n"
        ).encode("latin-1")
    )

    return bytes(pdf)


def main() -> None:
    markdown_text = SOURCE_PATH.read_text(encoding="utf-8")
    styled_lines = markdown_to_styled_lines(markdown_text)
    pages = paginate_lines(styled_lines)
    page_streams = [build_page_stream(page_lines) for page_lines in pages]
    pdf_bytes = build_pdf(page_streams)
    OUTPUT_PATH.write_bytes(pdf_bytes)
    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
