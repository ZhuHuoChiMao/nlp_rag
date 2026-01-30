import re
import json
from pathlib import Path
from typing import List, Dict, Tuple


INPUT_PATH = Path.home() / "Desktop" / "nlp_doc" / "Boutique prêt-à-porter.txt"
OUT_CLEAN_TXT = Path.home() / "Desktop" / "nlp_doc" / "cleaned_boutique_prêt_à_porter.v2.txt"
OUT_JSONL = Path.home() / "Desktop" / "nlp_doc" / "cleaned_boutique_prêt_à_porter.v2.jsonl"


MAX_CHARS = 1200
OVERLAP_CHARS = 180



def normalize_whitespace(text: str) -> str:
    """Nettoie et normalise les espaces et retours à la ligne."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ").replace("\u200B", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def unify_bullets(text: str) -> str:
    """Uniformise les puces et la numérotation (ex: •, *, 1., (1) -> formats stables)."""
    lines = text.split("\n")
    out = []
    for line in lines:
        s = line.strip()


        if re.match(r"^(\*|•|·|–|—|-)\s+", s):
            s = re.sub(r"^(\*|•|·|–|—|-)\s+", "- ", s)
            out.append(s)
            continue


        if re.match(r"^\(?\d{1,2}\)?[.)]\s+", s):
            s = re.sub(r"^\(?(\d{1,2})\)?[.)]\s+", r"\1) ", s)
            out.append(s)
            continue

        out.append(line)
    return "\n".join(out)



def normalize_note_tip_line(line: str) -> Tuple[str, str]:
    """Détecte 'Remarque/Conseil' et les normalise en 'NOTE:' / 'TIP:'."""
    s = line.strip()
    m = re.match(r"^(Remarque|Conseil)\s*:?\s*(.*)$", s, flags=re.IGNORECASE)
    if not m:
        return "text", line

    kind = m.group(1).lower()
    rest = m.group(2).strip()

    if kind.startswith("rem"):
        return "note", f"NOTE: {rest}".strip()
    else:
        return "tip", f"TIP: {rest}".strip()



FR_BODY_START = {
    "vous", "si", "lorsque", "pour", "afin", "le", "la", "les", "des", "de", "du",
    "une", "un", "ce", "cette", "ces", "dans", "sur", "en", "au", "aux", "il", "elle",
    "nous", "ils", "elles", "votre", "vos", "notre", "nos"
}

def looks_like_title(line: str) -> bool:
    """Heuristique : décide si une ligne ressemble à un titre de section."""
    s = line.strip()
    if not s:
        return False
    if len(s) > 90:
        return False
    if re.match(r"^-\s+", s):
        return False
    if re.search(r"[.!?…]$", s):
        return False


    first_word = re.split(r"\s+", s)[0].lower().strip("’'\"“”")
    if first_word in FR_BODY_START:
        return False


    words = re.findall(r"\w+", s, flags=re.UNICODE)
    if len(words) > 12:
        return False


    if s.endswith(":"):
        return True
    if s[0].isupper():
        return True
    if s.isupper():
        return True
    return False



STEP_LEAD_RE = re.compile(
    r"^(Pour|Comment|Étapes|Etapes|Procédure|Procedure|Afin de|Pour .*?)\b.*:\s*$",
    flags=re.IGNORECASE
)

def sentence_split_to_bullets(text: str) -> List[str]:
    """Découpe un texte en phrases courtes pour en faire des items (puces)."""
    s = re.sub(r"\s+", " ", text.strip())
    if not s:
        return []
    parts = re.split(r"(?<=[.;:])\s+(?=[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ])", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        p = p.rstrip()
        out.append(p)
    return out

def post_fix_tip_note_and_enumerations(text: str) -> str:
    """Post-traitement : fusion NOTE/TIP, transforme certaines énumérations en puces."""
    import re

    lines = text.split("\n")
    out = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()
        if line in ("NOTE:", "TIP:"):
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines):
                nxt = lines[j].strip()

                if nxt and not re.match(r"^-\s+", nxt) and nxt not in ("NOTE:", "TIP:"):
                    out.append(f"{line} {nxt}".strip())
                    i = j + 1
                    continue
        out.append(line)
        i += 1

    lines = out
    out = []

    enum_re = re.compile(r"^(.*\b(incluent|inclut|comprennent|comprend)\s*:)\s*(.+)$", re.IGNORECASE)
    for line in lines:
        m = enum_re.match(line.strip())
        if m:
            prefix = m.group(1).strip()
            rest = m.group(3).strip()
            out.append(prefix)
            items = [x.strip() for x in rest.split(";") if x.strip()]
            for it in items:
                out.append(f"- {it}")
        else:
            out.append(line)

    lines = out
    out = []

    lead_re = re.compile(r"^-\s+.*\b(telles?\s+que|tels?\s+que|comme)\s*:\s*$", re.IGNORECASE)

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        out.append(line)

        if lead_re.match(line.strip()):
            j = i + 1
            while j < len(lines):
                nxt_raw = lines[j]
                nxt = nxt_raw.strip()

                if nxt == "":
                    j += 1
                    continue
                if nxt in ("NOTE:", "TIP:") or looks_like_title(nxt) or re.match(r"^-\s+", nxt):
                    break

                if len(nxt) > 120:
                    break

                out.append(f"- {nxt}")
                j += 1

            i = j
            continue

        i += 1

    fixed = "\n".join(out)
    fixed = re.sub(r"\n{3,}", "\n\n", fixed).strip() + "\n"
    return fixed



def merge_wrapped_lines_smart(text: str) -> List[Dict]:
    """Fusionne les lignes coupées et produit des blocs typés (titre, paragraphe, puce, note...)."""
    lines = text.split("\n")
    blocks: List[Dict] = []

    buf = ""
    in_step_block = False
    step_buf_lines: List[str] = []

    def flush_paragraph():
        nonlocal buf
        if buf.strip():
            blocks.append({"type": "paragraph", "text": buf.strip()})
        buf = ""

    def flush_step_buf():
        nonlocal step_buf_lines
        if not step_buf_lines:
            return
        combined = " ".join(x.strip() for x in step_buf_lines if x.strip())
        bullets = sentence_split_to_bullets(combined)
        for b in bullets:
            blocks.append({"type": "bullet", "text": f"- {b}" if not b.startswith("- ") else b})
        step_buf_lines = []

    for raw in lines:
        line = raw.strip()

        if not line:
            if in_step_block:
                flush_step_buf()
                in_step_block = False
            flush_paragraph()
            blocks.append({"type": "blank", "text": ""})
            continue

        btype, norm = normalize_note_tip_line(line)
        if btype in ("note", "tip"):
            if in_step_block:
                flush_step_buf()
                in_step_block = False
            flush_paragraph()
            blocks.append({"type": btype, "text": norm})
            continue

        if looks_like_title(line):
            if in_step_block:
                flush_step_buf()
                in_step_block = False
            flush_paragraph()
            blocks.append({"type": "title", "text": line.rstrip(":")})
            continue

        if re.match(r"^-\s+\S+", line) or re.match(r"^\d{1,2}\)\s+\S+", line):
            if in_step_block:
                flush_step_buf()
                in_step_block = False
            flush_paragraph()
            if re.match(r"^\d{1,2}\)\s+", line):
                blocks.append({"type": "bullet", "text": f"- {line}"})
            else:
                blocks.append({"type": "bullet", "text": line})
            continue

        if STEP_LEAD_RE.match(line):
            if in_step_block:
                flush_step_buf()
            flush_paragraph()
            blocks.append({"type": "paragraph", "text": line})  # 保留引导句
            in_step_block = True
            step_buf_lines = []
            continue

        if in_step_block:
            step_buf_lines.append(line)
        else:
            if not buf:
                buf = line
            else:
                if buf.endswith("-") and len(buf) > 2 and buf[-2].isalpha():
                    buf = buf[:-1] + line
                else:
                    buf += " " + line

    if in_step_block:
        flush_step_buf()
    flush_paragraph()

    compact: List[Dict] = []
    for b in blocks:
        if b["type"] == "blank":
            if not compact or compact[-1]["type"] == "blank":
                continue
        compact.append(b)
    while compact and compact[0]["type"] == "blank":
        compact.pop(0)
    while compact and compact[-1]["type"] == "blank":
        compact.pop()
    return compact


def blocks_to_sections(blocks: List[Dict]) -> List[Dict]:
    """Regroupe les blocs en sections, séparées par les titres."""
    sections: List[Dict] = []
    current = {"title": "UNTITLED", "blocks": []}

    def flush():
        nonlocal current
        if current["blocks"]:
            sections.append(current)
        current = {"title": "UNTITLED", "blocks": []}

    for b in blocks:
        if b["type"] == "title":
            flush()
            current["title"] = b["text"]
        else:
            current["blocks"].append(b)

    flush()
    return sections


def blocks_to_text(blocks: List[Dict]) -> str:
    """Reconstruit un texte à partir d'une liste de blocs."""
    parts = []
    for b in blocks:
        if b["type"] == "blank":
            parts.append("")
        else:
            parts.append(b["text"])
    txt = "\n".join(parts)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    return txt

def chunk_blocks(blocks: List[Dict], max_chars: int, overlap_chars: int) -> List[List[Dict]]:
    """Découpe les blocs en chunks de taille max avec chevauchement (overlap)."""
    chunks: List[List[Dict]] = []
    cur: List[Dict] = []
    cur_len = 0

    def blocks_len(bs: List[Dict]) -> int:
        return len(blocks_to_text(bs))

    def take_overlap(prev: List[Dict]) -> List[Dict]:
        if overlap_chars <= 0 or not prev:
            return []
        tail: List[Dict] = []
        for b in reversed(prev):
            tail.insert(0, b)
            if blocks_len(tail) >= overlap_chars:
                break
        return tail

    for b in blocks:
        b_len = len(b.get("text", "")) + 2

        if not cur:
            cur = [b]
            cur_len = b_len
            continue

        if cur_len + b_len <= max_chars:
            cur.append(b)
            cur_len += b_len
        else:
            chunks.append(cur)
            ov = take_overlap(cur)
            cur = ov + [b]
            cur_len = blocks_len(cur)

    if cur:
        chunks.append(cur)
    return chunks


def main():
    """Lit, nettoie, structure le texte, puis exporte en .txt propre et en .jsonl chunké."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    raw = INPUT_PATH.read_text(encoding="utf-8", errors="replace")
    t = normalize_whitespace(raw)
    t = unify_bullets(t)
    t = normalize_whitespace(t)

    blocks = merge_wrapped_lines_smart(t)
    sections = blocks_to_sections(blocks)

    cleaned_lines = []
    for sec in sections:
        if sec["title"] != "UNTITLED":
            cleaned_lines.append(sec["title"])
        body = blocks_to_text(sec["blocks"])
        if body:
            cleaned_lines.append(body)
        cleaned_lines.append("")  # section 间空行

    cleaned_text = "\n".join(cleaned_lines).strip() + "\n"
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    cleaned_text = post_fix_tip_note_and_enumerations(cleaned_text)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip() + "\n"
    OUT_CLEAN_TXT.write_text(cleaned_text, encoding="utf-8")
    print(f"[OK] cleaned txt -> {OUT_CLEAN_TXT}")

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        global_id = 0
        for s_idx, sec in enumerate(sections, start=1):
            sec_chunks = chunk_blocks(sec["blocks"], MAX_CHARS, OVERLAP_CHARS)
            for c_idx, ch_blocks in enumerate(sec_chunks, start=1):
                global_id += 1
                txt = post_fix_tip_note_and_enumerations(blocks_to_text(ch_blocks))
                txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
                types = {b["type"] for b in ch_blocks if b["type"] != "blank"}
                block_type = "mixed"
                if types == {"note"}:
                    block_type = "note"
                elif types == {"tip"}:
                    block_type = "tip"
                elif "note" in types or "tip" in types:
                    block_type = "has_note_or_tip"

                obj = {
                    "id": global_id,
                    "section_id": s_idx,
                    "chunk_id": c_idx,
                    "title": sec["title"],
                    "block_type": block_type,
                    "text": txt,
                    "char_len": len(txt),
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] sections+chunks jsonl -> {OUT_JSONL}")
    print(f"[INFO] sections: {len(sections)}  |  max_chars={MAX_CHARS} overlap={OVERLAP_CHARS}")


if __name__ == "__main__":
    main()
