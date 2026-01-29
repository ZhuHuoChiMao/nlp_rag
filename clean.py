import re
import json
from pathlib import Path
from typing import List, Dict, Tuple

# ========= 路径（改这里） =========
INPUT_PATH = Path.home() / "Desktop" / "nlp_doc" / "Boutique prêt-à-porter.txt"
OUT_CLEAN_TXT = Path.home() / "Desktop" / "nlp_doc" / "cleaned_boutique_prêt_à_porter.v2.txt"
OUT_JSONL = Path.home() / "Desktop" / "nlp_doc" / "cleaned_boutique_prêt_à_porter.v2.jsonl"

# ========= chunk 参数（可调） =========
MAX_CHARS = 1200
OVERLAP_CHARS = 180


# ---------- 基础清洗 ----------
def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00A0", " ").replace("\u200B", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def unify_bullets(text: str) -> str:
    lines = text.split("\n")
    out = []
    for line in lines:
        s = line.strip()

        # 统一 bullet 符号
        if re.match(r"^(\*|•|·|–|—|-)\s+", s):
            s = re.sub(r"^(\*|•|·|–|—|-)\s+", "- ", s)
            out.append(s)
            continue

        # 统一编号列表 1) / 1. / (1)
        if re.match(r"^\(?\d{1,2}\)?[.)]\s+", s):
            s = re.sub(r"^\(?(\d{1,2})\)?[.)]\s+", r"\1) ", s)
            out.append(s)
            continue

        out.append(line)
    return "\n".join(out)


# ---------- NOTE / TIP 规范化 ----------
def normalize_note_tip_line(line: str) -> Tuple[str, str]:
    """
    return (block_type, normalized_line)
    block_type: "note" | "tip" | "text"
    """
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


# ---------- 更保守的标题识别 ----------
FR_BODY_START = {
    "vous", "si", "lorsque", "pour", "afin", "le", "la", "les", "des", "de", "du",
    "une", "un", "ce", "cette", "ces", "dans", "sur", "en", "au", "aux", "il", "elle",
    "nous", "ils", "elles", "votre", "vos", "notre", "nos"
}

def looks_like_title(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if len(s) > 90:
        return False
    if re.match(r"^-\s+", s):
        return False
    if re.search(r"[.!?…]$", s):
        return False

    # 过滤：常见正文开头
    first_word = re.split(r"\s+", s)[0].lower().strip("’'\"“”")
    if first_word in FR_BODY_START:
        return False

    # 标题一般词数不多
    words = re.findall(r"\w+", s, flags=re.UNICODE)
    if len(words) > 12:
        return False

    # 大写开头 / 或者全大写 / 或者以冒号结尾
    if s.endswith(":"):
        return True
    if s[0].isupper():
        return True
    if s.isupper():
        return True
    return False


# ---------- “步骤段”识别与拆分 ----------
STEP_LEAD_RE = re.compile(
    r"^(Pour|Comment|Étapes|Etapes|Procédure|Procedure|Afin de|Pour .*?)\b.*:\s*$",
    flags=re.IGNORECASE
)

def sentence_split_to_bullets(text: str) -> List[str]:
    """
    简单句子拆分：按 . ; 以及 “. ” 这种边界拆成 bullets
    """
    s = re.sub(r"\s+", " ", text.strip())
    if not s:
        return []
    parts = re.split(r"(?<=[.;:])\s+(?=[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŸ])", s)
    # 再处理没有大写开头但有分号的情况
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 去掉末尾孤立分号
        p = p.rstrip()
        out.append(p)
    return out

def post_fix_tip_note_and_enumerations(text: str) -> str:
    import re

    lines = text.split("\n")
    out = []
    i = 0

    # 1) 修复空 NOTE/TIP：把下一段第一条内容并进来
    while i < len(lines):
        line = lines[i].rstrip()
        if line in ("NOTE:", "TIP:"):
            # 跳过后面的空行
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines):
                nxt = lines[j].strip()
                # 只合并“看起来是正文”的下一行
                if nxt and not re.match(r"^-\s+", nxt) and nxt not in ("NOTE:", "TIP:"):
                    out.append(f"{line} {nxt}".strip())
                    i = j + 1
                    continue
        out.append(line)
        i += 1

    lines = out
    out = []

    # 2) 把 “incluent : ... ; ...” 拆成 bullets（你之前那条保留）
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

    # 3) 修复 “telles que : / comme :” 后的子列表：把后续短行变成 bullets
    lead_re = re.compile(r"^-\s+.*\b(telles?\s+que|tels?\s+que|comme)\s*:\s*$", re.IGNORECASE)

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        out.append(line)

        if lead_re.match(line.strip()):
            # 把后续连续的“短正文行”变成 bullets，直到遇到空行/标题/已经是 bullet
            j = i + 1
            while j < len(lines):
                nxt_raw = lines[j]
                nxt = nxt_raw.strip()

                if nxt == "":
                    # 允许冒号后空一行：跳过空行继续
                    j += 1
                    continue
                if nxt in ("NOTE:", "TIP:") or looks_like_title(nxt) or re.match(r"^-\s+", nxt):
                    break

                # 过滤极长段落（防止把正文误当子项）
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



# ---------- 合并断行（保留步骤结构 + NOTE/TIP 块） ----------
def merge_wrapped_lines_smart(text: str) -> List[Dict]:
    """
    返回 blocks: [{"type": "...", "text": "..."}]
    type: title | paragraph | bullet | note | tip | blank
    """
    lines = text.split("\n")
    blocks: List[Dict] = []

    buf = ""  # 普通段落缓冲
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

        # 空行：结束段落/步骤段
        if not line:
            if in_step_block:
                flush_step_buf()
                in_step_block = False
            flush_paragraph()
            blocks.append({"type": "blank", "text": ""})
            continue

        # NOTE/TIP 行：独立成块
        btype, norm = normalize_note_tip_line(line)
        if btype in ("note", "tip"):
            if in_step_block:
                flush_step_buf()
                in_step_block = False
            flush_paragraph()
            blocks.append({"type": btype, "text": norm})
            continue

        # 标题行：独立成块
        if looks_like_title(line):
            if in_step_block:
                flush_step_buf()
                in_step_block = False
            flush_paragraph()
            blocks.append({"type": "title", "text": line.rstrip(":")})
            continue

        # 已经是 bullet / 编号：独立成块
        if re.match(r"^-\s+\S+", line) or re.match(r"^\d{1,2}\)\s+\S+", line):
            if in_step_block:
                flush_step_buf()
                in_step_block = False
            flush_paragraph()
            # 编号统一为 bullet 也可以；这里保留编号文本
            if re.match(r"^\d{1,2}\)\s+", line):
                blocks.append({"type": "bullet", "text": f"- {line}"})
            else:
                blocks.append({"type": "bullet", "text": line})
            continue

        # 进入“步骤段”（冒号引出）
        if STEP_LEAD_RE.match(line):
            if in_step_block:
                flush_step_buf()
            flush_paragraph()
            blocks.append({"type": "paragraph", "text": line})  # 保留引导句
            in_step_block = True
            step_buf_lines = []
            continue

        # 普通内容：如果处于步骤段，就收集进 step_buf；否则合并为段落
        if in_step_block:
            step_buf_lines.append(line)
        else:
            if not buf:
                buf = line
            else:
                # 断词连字符拼接
                if buf.endswith("-") and len(buf) > 2 and buf[-2].isalpha():
                    buf = buf[:-1] + line
                else:
                    buf += " " + line

    # 结尾 flush
    if in_step_block:
        flush_step_buf()
    flush_paragraph()

    # 去掉多余 blank（首尾、连续）
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


# ---------- 按标题切 section ----------
def blocks_to_sections(blocks: List[Dict]) -> List[Dict]:
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


# ---------- section 内 chunk ----------
def blocks_to_text(blocks: List[Dict]) -> str:
    parts = []
    for b in blocks:
        if b["type"] == "blank":
            parts.append("")
        else:
            parts.append(b["text"])
    # 用双换行保持段落/列表结构
    txt = "\n".join(parts)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    return txt

def chunk_blocks(blocks: List[Dict], max_chars: int, overlap_chars: int) -> List[List[Dict]]:
    """
    以 block 为单位切 chunk，尽量不把 bullet/NOTE/TIP 拆开。
    overlap 用“尾部 block 累积到 overlap_chars”为准。
    """
    chunks: List[List[Dict]] = []
    cur: List[Dict] = []
    cur_len = 0

    def blocks_len(bs: List[Dict]) -> int:
        return len(blocks_to_text(bs))

    def take_overlap(prev: List[Dict]) -> List[Dict]:
        if overlap_chars <= 0 or not prev:
            return []
        tail: List[Dict] = []
        # 从后往前拿 blocks，直到长度达到 overlap_chars
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
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    raw = INPUT_PATH.read_text(encoding="utf-8", errors="replace")
    t = normalize_whitespace(raw)
    t = unify_bullets(t)
    t = normalize_whitespace(t)

    blocks = merge_wrapped_lines_smart(t)
    sections = blocks_to_sections(blocks)

    # 输出 cleaned txt（保持结构：标题、段落、bullet、NOTE/TIP）
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

    # 输出 jsonl（按 section + chunk）
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        global_id = 0
        for s_idx, sec in enumerate(sections, start=1):
            sec_chunks = chunk_blocks(sec["blocks"], MAX_CHARS, OVERLAP_CHARS)
            for c_idx, ch_blocks in enumerate(sec_chunks, start=1):
                global_id += 1
                txt = post_fix_tip_note_and_enumerations(blocks_to_text(ch_blocks))
                txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
                # chunk 的主要 block_type（用于筛选 NOTE/TIP chunk）
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
