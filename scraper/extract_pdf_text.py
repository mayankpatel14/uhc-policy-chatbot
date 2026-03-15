import os
import re
import json
import pdfplumber
from tqdm import tqdm
from bs4 import BeautifulSoup
from dataclasses import dataclass, field, asdict
from typing import Optional

PDF_DIR = "data/pdfs"
OUTPUT_FILE = "data/processed/extracted_sections.json"

SKIP_FILES = {
    "TOU-UHCPROVIDER-COM-EN.pdf",
    "OSPP-UHCPROVIDER-COM-EN.pdf",
}

SECTION_HEADERS = [
    "Instructions for Use",
    "Coverage Rationale",
    "Coverage Summary",
    "Application",
    "Medical Records Documentation Used for Reviews",
    "Documentation Requirements",
    "Definitions",
    "Applicable Codes",
    "Description of Services",
    "Benefit Considerations",
    "Clinical Evidence",
    "Background",
    "U.S. Food and Drug Administration",
    "Centers for Medicare and Medicaid Services",
    "References",
    "Policy History/Revision Information",
    "Frequently Asked Questions",
]

SKIP_SECTIONS = {
    "Instructions for Use",
    "Policy History/Revision Information",
}

PAGE_HEADER_PATTERNS = [
    re.compile(r"^.{0,120}Page\s+\d+\s+of\s+\d+\s*$"),
    re.compile(r"^UnitedHealthcare.*(?:Medical|Drug)\s+(?:Policy|Benefit).*(?:Effective|Policy)"),
    re.compile(r"^Proprietary Information of UnitedHealthcare"),
    re.compile(r"^©\s*\d{4}"),
    re.compile(r"^Effective\s+\d{2}/\d{2}/\d{4}\s*$"),
]

SIDEBAR_PATTERNS = [
    re.compile(r"^(?:Related\s+)?(?:Commercial|Community\s+Plan|Medicare\s+Advantage)\s+(?:Policy|Policies)", re.IGNORECASE),
    re.compile(r"^Related\s+(?:Commercial|List)", re.IGNORECASE),
    re.compile(r"^Medicare\s+Advantage\s+Policy", re.IGNORECASE),
]

POLICY_NUMBER_RE = re.compile(r"Policy\s+Number:\s*(\S+)")
EFFECTIVE_DATE_RE = re.compile(r"Effective\s+Date:\s*(.+?)(?:\s{2,}|$)")
PLAN_TYPE_RE = re.compile(r"UnitedHealthcare®?\s+(Commercial.*?)$", re.MULTILINE)


@dataclass
class PolicySection:
    section: str
    content: str
    page_start: int
    page_end: int


@dataclass
class PolicyDocument:
    filename: str
    policy_name: str
    policy_number: str
    effective_date: str
    plan_type: str
    doc_type: str
    sections: list = field(default_factory=list)


def is_html_file(path):
    try:
        with open(path, "rb") as f:
            start = f.read(200).decode(errors="ignore").lower()
            return "<html" in start or "<!doctype html" in start
    except Exception:
        return False


def is_page_header(line):
    stripped = line.strip()
    if not stripped:
        return True
    for pat in PAGE_HEADER_PATTERNS:
        if pat.search(stripped):
            return True
    return False


def is_toc_line(line):
    stripped = line.strip()
    if re.match(r"^Table of Contents\s*Page?\s*$", stripped, re.IGNORECASE):
        return True
    if re.match(r"^.{3,80}\s*\.{3,}\s*\d+\s*$", stripped):
        return True
    return False


def is_sidebar_start(line):
    stripped = line.strip()
    for pat in SIDEBAR_PATTERNS:
        if pat.match(stripped):
            return True
    return False


def detect_section(line):
    stripped = line.strip()
    for header in SECTION_HEADERS:
        if stripped == header or stripped.startswith(header + "\n"):
            return header
        if re.match(re.escape(header) + r"\s*$", stripped):
            return header
    return None


def extract_metadata(full_text, filename):
    policy_name = os.path.basename(filename).replace(".pdf", "")

    policy_number = ""
    m = POLICY_NUMBER_RE.search(full_text[:2000])
    if m:
        policy_number = m.group(1).strip()

    effective_date = ""
    m = EFFECTIVE_DATE_RE.search(full_text[:2000])
    if m:
        effective_date = m.group(1).strip()

    plan_type = ""
    m = PLAN_TYPE_RE.search(full_text[:1000])
    if m:
        plan_type = m.group(1).strip()

    doc_type = "Medical Policy"
    if "Medical Benefit Drug Policy" in full_text[:1000]:
        doc_type = "Medical Benefit Drug Policy"
    elif "Medical Policy Update Bulletin" in full_text[:500]:
        doc_type = "Update Bulletin"

    return policy_name, policy_number, effective_date, plan_type, doc_type


def clean_page_text(text):
    lines = text.split("\n")
    cleaned = []
    in_sidebar = False
    in_toc = False

    for line in lines:
        if is_page_header(line):
            continue

        if is_toc_line(line):
            in_toc = True
            continue

        if in_toc:
            if re.match(r"^.{3,80}\s*\.{3,}\s*\d+\s*$", line.strip()):
                continue
            stripped = line.strip()
            if stripped and not re.search(r"\.{3,}", stripped):
                sec = detect_section(stripped)
                if not sec:
                    in_toc = False

            if in_toc:
                continue

        if is_sidebar_start(line):
            in_sidebar = True
            continue

        if in_sidebar:
            stripped = line.strip()
            if stripped.startswith("•") or stripped.startswith("–") or not stripped:
                continue
            sec = detect_section(stripped)
            if sec or (stripped and not stripped.startswith("•")):
                in_sidebar = False
                if sec:
                    cleaned.append(line)
                    continue
                cleaned.append(line)
                continue
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def extract_pages_pdf(pdf_path):
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    pages.append((page_num, text))

                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        table_text = format_table(table)
                        if table_text:
                            pages.append((page_num, f"[TABLE]\n{table_text}\n[/TABLE]"))
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")
    return pages


def format_table(table):
    if not table or len(table) < 2:
        return ""
    rows = []
    for row in table:
        if row:
            cells = [str(cell).strip() if cell else "" for cell in row]
            if any(cells):
                rows.append(" | ".join(cells))
    return "\n".join(rows)


def build_paragraphs(text):
    lines = text.split("\n")
    paragraphs = []
    current = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue

        is_bullet = bool(re.match(r"^[•\-–▪o]\s", stripped))
        is_numbered = bool(re.match(r"^\d+[\.\)]\s", stripped))
        is_lettered = bool(re.match(r"^[a-z][\.\)]\s", stripped))
        is_list_item = is_bullet or is_numbered or is_lettered

        if is_list_item:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            current.append(stripped)
        elif stripped.startswith("o\t") or stripped.startswith("o "):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            current.append(stripped)
        else:
            current.append(stripped)

    if current:
        paragraphs.append(" ".join(current))

    return paragraphs


def segment_into_sections(pages):
    all_text_by_section = []
    current_section = ""
    current_content = []
    current_page_start = 1

    for page_num, raw_text in pages:
        cleaned = clean_page_text(raw_text)
        lines = cleaned.split("\n")

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_content.append("")
                continue

            sec = detect_section(stripped)
            if sec:
                if current_content:
                    text = "\n".join(current_content).strip()
                    if text and current_section:
                        all_text_by_section.append(PolicySection(
                            section=current_section,
                            content=text,
                            page_start=current_page_start,
                            page_end=page_num
                        ))
                current_section = sec
                current_content = []
                current_page_start = page_num
                continue

            current_content.append(stripped)

    if current_content and current_section:
        text = "\n".join(current_content).strip()
        if text:
            last_page = pages[-1][0] if pages else 1
            all_text_by_section.append(PolicySection(
                section=current_section,
                content=text,
                page_start=current_page_start,
                page_end=last_page
            ))

    return all_text_by_section


def extract_html(path, filename):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")
    text = soup.get_text("\n")

    policy_name, policy_number, effective_date, plan_type, doc_type = extract_metadata(text, filename)

    pages = [(1, text)]
    sections = segment_into_sections(pages)

    return PolicyDocument(
        filename=filename,
        policy_name=policy_name,
        policy_number=policy_number,
        effective_date=effective_date,
        plan_type=plan_type,
        doc_type=doc_type,
        sections=[asdict(s) for s in sections if s.section not in SKIP_SECTIONS]
    )


def extract_policy(pdf_path, filename):
    pages = extract_pages_pdf(pdf_path)

    if not pages:
        return None

    full_text = "\n".join(text for _, text in pages[:3])
    policy_name, policy_number, effective_date, plan_type, doc_type = extract_metadata(full_text, filename)

    sections = segment_into_sections(pages)

    filtered_sections = []
    for sec in sections:
        if sec.section in SKIP_SECTIONS:
            continue

        paragraphs = build_paragraphs(sec.content)
        cleaned_content = "\n\n".join(p for p in paragraphs if len(p.strip()) > 10)

        if cleaned_content.strip():
            sec.content = cleaned_content
            filtered_sections.append(sec)

    return PolicyDocument(
        filename=filename,
        policy_name=policy_name,
        policy_number=policy_number,
        effective_date=effective_date,
        plan_type=plan_type,
        doc_type=doc_type,
        sections=[asdict(s) for s in filtered_sections]
    )


def main():
    all_policies = []

    pdfs = [f for f in os.listdir(PDF_DIR) if f not in SKIP_FILES]
    pdfs.sort()

    for filename in tqdm(pdfs, desc="Extracting policies"):
        path = os.path.join(PDF_DIR, filename)

        if is_html_file(path):
            doc = extract_html(path, filename)
        else:
            doc = extract_policy(path, filename)

        if doc and doc.sections:
            all_policies.append(asdict(doc))

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_policies, f, indent=2, ensure_ascii=False)

    total_sections = sum(len(p["sections"]) for p in all_policies)
    print(f"Extracted {len(all_policies)} policies with {total_sections} sections")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()