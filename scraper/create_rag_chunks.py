import json
import re
import hashlib
from tqdm import tqdm

INPUT_FILE = "data/processed/extracted_sections.json"
OUTPUT_FILE = "data/processed/rag_chunks.json"

TARGET_CHUNK_TOKENS = 400
MAX_CHUNK_TOKENS = 600
OVERLAP_SENTENCES = 2

LOW_VALUE_SECTIONS = {"References", "U.S. Food and Drug Administration"}
BOILERPLATE_SECTIONS = {"Instructions for Use", "Policy History/Revision Information"}


def estimate_tokens(text):
    return int(len(text.split()) * 1.3)


def split_sentences(text):
    parts = re.split(r'(?<=[.;])\s+(?=[A-Z])', text)
    sentences = []
    for part in parts:
        if estimate_tokens(part) > MAX_CHUNK_TOKENS:
            sub_parts = re.split(r'(?<=[:;])\s+', part)
            sentences.extend(sub_parts)
        else:
            sentences.append(part)
    return [s.strip() for s in sentences if s.strip()]


def chunk_id(policy, section, idx):
    raw = f"{policy}__{section}__{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def chunk_by_criteria(text, policy_name):
    blocks = re.split(
        r'\n\n(?=(?:The following|For (?:initial|continuation|subsequent|revision|replacement)|'
        r'(?:An?|The)\s+\w.*?is (?:proven|unproven|medically necessary|not medically)|'
        r'(?:Multiplex|Implantable|Removable|Emergency|Non-Surgical|Surgical)))',
        text
    )

    if len(blocks) <= 1:
        blocks = re.split(r'\n\n', text)

    result = []
    current = []
    current_tokens = 0

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        block_tokens = estimate_tokens(block)

        if block_tokens > MAX_CHUNK_TOKENS:
            if current:
                result.append("\n\n".join(current))
                current = []
                current_tokens = 0

            sents = split_sentences(block)
            sent_group = []
            sent_tokens = 0
            for sent in sents:
                st = estimate_tokens(sent)
                if sent_tokens + st > TARGET_CHUNK_TOKENS and sent_group:
                    result.append(" ".join(sent_group))
                    overlap = sent_group[-OVERLAP_SENTENCES:] if len(sent_group) > OVERLAP_SENTENCES else []
                    sent_group = overlap
                    sent_tokens = sum(estimate_tokens(s) for s in sent_group)
                sent_group.append(sent)
                sent_tokens += st
            if sent_group:
                result.append(" ".join(sent_group))

        elif current_tokens + block_tokens > TARGET_CHUNK_TOKENS and current:
            result.append("\n\n".join(current))
            current = [block]
            current_tokens = block_tokens
        else:
            current.append(block)
            current_tokens += block_tokens

    if current:
        result.append("\n\n".join(current))

    return result


def chunk_code_table(text):
    lines = text.split("\n")
    chunks = []
    current_lines = []
    current_tokens = 0

    header_line = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if re.match(r"^(?:CPT|HCPCS|Diagnosis|ICD-10)\s+(?:Code|Description)", stripped, re.IGNORECASE):
            header_line = stripped
            continue

        if re.match(r"^The following list\(s\)", stripped):
            continue
        if re.match(r"^CPT®?\s+is a registered", stripped):
            continue
        if re.match(r"^Listing of a code", stripped):
            continue

        line_tokens = estimate_tokens(stripped)

        if current_tokens + line_tokens > TARGET_CHUNK_TOKENS and current_lines:
            chunk_text = "\n".join(current_lines)
            if header_line:
                chunk_text = header_line + "\n" + chunk_text
            chunks.append(chunk_text)
            current_lines = []
            current_tokens = 0

        current_lines.append(stripped)
        current_tokens += line_tokens

    if current_lines:
        chunk_text = "\n".join(current_lines)
        if header_line:
            chunk_text = header_line + "\n" + chunk_text
        chunks.append(chunk_text)

    return chunks


def chunk_clinical_evidence(text):
    study_splits = re.split(
        r'\n\n(?=(?:[A-Z][a-z]+(?:\s+(?:et al\.|and|&))?.*?\(\d{4}\))|'
        r'(?:A\s+(?:phase|prospective|retrospective|randomized|multicenter|systematic|meta-analysis|Cochrane))|'
        r'(?:Professional Societies|American|European|National|International))',
        text
    )

    if len(study_splits) <= 1:
        study_splits = text.split("\n\n")

    chunks = []
    current = []
    current_tokens = 0

    for block in study_splits:
        block = block.strip()
        if not block:
            continue

        block_tokens = estimate_tokens(block)

        if block_tokens > MAX_CHUNK_TOKENS:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0

            sents = split_sentences(block)
            sent_group = []
            sent_tokens = 0
            for sent in sents:
                st = estimate_tokens(sent)
                if sent_tokens + st > TARGET_CHUNK_TOKENS and sent_group:
                    chunks.append(" ".join(sent_group))
                    overlap = sent_group[-OVERLAP_SENTENCES:] if len(sent_group) > OVERLAP_SENTENCES else []
                    sent_group = overlap
                    sent_tokens = sum(estimate_tokens(s) for s in sent_group)
                sent_group.append(sent)
                sent_tokens += st
            if sent_group:
                chunks.append(" ".join(sent_group))

        elif current_tokens + block_tokens > MAX_CHUNK_TOKENS and current:
            chunks.append("\n\n".join(current))
            current = [block]
            current_tokens = block_tokens
        else:
            current.append(block)
            current_tokens += block_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk_section(section_name, content, policy_name):
    if section_name in BOILERPLATE_SECTIONS:
        return []

    if section_name == "Applicable Codes" or section_name == "Coverage Summary":
        return chunk_code_table(content)

    if section_name == "Clinical Evidence":
        return chunk_clinical_evidence(content)

    if section_name in ("Coverage Rationale", "Application", "Definitions",
                        "Documentation Requirements", "Medical Records Documentation Used for Reviews"):
        return chunk_by_criteria(content, policy_name)

    tokens = estimate_tokens(content)
    if tokens <= TARGET_CHUNK_TOKENS:
        return [content]

    return chunk_by_criteria(content, policy_name)


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        policies = json.load(f)

    all_chunks = []

    for policy in tqdm(policies, desc="Creating chunks"):
        policy_name = policy["policy_name"]
        policy_number = policy.get("policy_number", "")
        effective_date = policy.get("effective_date", "")
        plan_type = policy.get("plan_type", "")
        doc_type = policy.get("doc_type", "")

        for section_data in policy["sections"]:
            section_name = section_data["section"]
            content = section_data["content"]
            page_start = section_data.get("page_start", 0)
            page_end = section_data.get("page_end", 0)

            if section_name in BOILERPLATE_SECTIONS:
                continue

            text_chunks = chunk_section(section_name, content, policy_name)

            for idx, chunk_text in enumerate(text_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text or len(chunk_text) < 20:
                    continue

                all_chunks.append({
                    "id": chunk_id(policy_name, section_name, idx),
                    "policy_name": policy_name,
                    "policy_number": policy_number,
                    "effective_date": effective_date,
                    "plan_type": plan_type,
                    "doc_type": doc_type,
                    "section": section_name,
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_index": idx,
                    "total_chunks_in_section": len(text_chunks),
                    "text": chunk_text,
                })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Total chunks: {len(all_chunks)}")
    print(f"Policies processed: {len(policies)}")
    print(f"Saved to: {OUTPUT_FILE}")

    section_counts = {}
    for c in all_chunks:
        section_counts[c["section"]] = section_counts.get(c["section"], 0) + 1
    print("\nChunks per section:")
    for sec, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"  {sec}: {count}")


if __name__ == "__main__":
    main()