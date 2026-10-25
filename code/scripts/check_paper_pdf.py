"""Check text geometry and create page contact sheets after Poppler rendering.

Uses the optional paper environment (requirements-paper.txt), not training deps.
Contact sheets support visual review; passing geometry does not replace it.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pdfplumber
from PIL import Image, ImageDraw
from pypdf import PdfReader


def check(pdf, renders, output):
    reader = PdfReader(pdf)
    pages = []
    text_parts = []
    with pdfplumber.open(pdf) as document:
        for number, page in enumerate(document.pages, 1):
            outside = [
                c
                for c in page.chars
                if c["x0"] < 42
                or c["x1"] > page.width - 42
                or c["top"] < 22
                or c["bottom"] > page.height - 22
            ]
            if outside:
                raise ValueError(f"Text outside safe margins on page {number}: {outside[:2]}")
            value = page.extract_text() or ""
            if not value.strip() or any(c in value for c in "\ufffd\u25a0"):
                raise ValueError(f"Blank or damaged text on page {number}")
            text_parts.append(value)
            pages.append(
                {
                    "page": number,
                    "characters": len(page.chars),
                    "images": len(page.images),
                    "text_within_safe_margins": True,
                }
            )
    fulltext = "\n".join(text_parts)
    for term in ("Аннотация", "0.02471", "0.11043", "0.21944", "Результаты", "Список литературы"):
        if term not in fulltext:
            raise ValueError(f"Expected manuscript text missing: {term}")
    pictures = sorted(renders.glob("research-*.png"))
    if len(pictures) != len(reader.pages):
        raise ValueError("Rendered page count differs from PDF page count")
    sheets = []
    for start in range(0, len(pictures), 6):
        selection = pictures[start : start + 6]
        sheet = Image.new("RGB", (1600, 2240), "#e8edf2")
        draw = ImageDraw.Draw(sheet)
        for position, path in enumerate(selection):
            with Image.open(path) as picture:
                picture.thumbnail((770, 1055))
                x, y = 15 + (position % 2) * 795, 26 + (position // 2) * 740
                # Three rows: each page thumbnail is 485x685, leaving a clear label.
                picture.thumbnail((750, 695))
                x += (770 - picture.width) // 2
                sheet.paste(picture.convert("RGB"), (x, y + 16))
                draw.text((x, y), f"Page {start + position + 1}", fill="#16263b")
        target = renders / f"contact-{start // 6 + 1}.jpg"
        sheet.save(target, quality=92)
        sheets.append(target.as_posix())
    result = {
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "pages": pages,
        "reference_links": sum(
            len(page["/Annots"].get_object()) if "/Annots" in page else 0 for page in reader.pages
        ),
        "outlines": len(reader.outline),
        "contact_sheets": sheets,
        "automatic_checks": "passed; visual inspection recorded separately",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(
        json.dumps(
            {
                "pages": len(pages),
                "links": result["reference_links"],
                "outlines": result["outlines"],
                "sheets": sheets,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=Path("output/pdf/research-paper.pdf"))
    parser.add_argument("--renders", type=Path, default=Path("tmp/pdfs/research-redesign/pages"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/quality.json"))
    args = parser.parse_args()
    check(args.pdf, args.renders, args.output)
