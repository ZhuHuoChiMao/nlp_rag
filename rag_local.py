import os
import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import List

# -----------------------------
# Utils: load jsonl
# -----------------------------
def load_jsonl_dir(data_dir: Path) -> List[Dict[str, Any]]:
    records = []
    for fp in sorted(data_dir.glob("*.jsonl")):
        with fp.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] JSON parse fail: {fp.name}:{line_no} -> {e}")
                    continue
                # Minimal sanity checks
                if "text" not in obj or not obj["text"]:
                    continue
                if "doc_id" not in obj:
                    # fallback: derive from file stem (but you already added doc_id)
                    obj["doc_id"] = fp.stem
                if "title" not in obj:
                    obj["title"] = ""
                records.append(obj)
    return records


def build_text_for_embedding(r: Dict[str, Any]) -> str:
    # Put identifiers + title before body to improve retrieval hits
    doc_id = r.get("doc_id", "")
    title = r.get("title", "")
    section_id = r.get("section_id", "")
    chunk_id = r.get("chunk_id", "")
    header = f"[doc_id={doc_id} | section_id={section_id} | chunk_id={chunk_id} | title={title}]"
    return header + "\n" + r.get("text", "")


# -----------------------------
# Index build / load
# -----------------------------
def normalize_rows(x: np.ndarray) -> np.ndarray:
    # for cosine similarity via inner product
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    # Cosine similarity = inner product on normalized vectors
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    return index


def save_index(index_dir: Path, index: faiss.Index, records: List[Dict[str, Any]], embed_model_name: str):
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    with (index_dir / "meta.pkl").open("wb") as f:
        pickle.dump(
            {
                "records": records,
                "embed_model_name": embed_model_name,
            },
            f,
        )


def load_index(index_dir: Path) -> Tuple[faiss.Index, List[Dict[str, Any]], str]:
    index_path = index_dir / "faiss.index"
    meta_path = index_dir / "meta.pkl"
    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index not found. Run with --rebuild first.")
    index = faiss.read_index(str(index_path))
    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    return index, meta["records"], meta.get("embed_model_name", "")


# -----------------------------
# Retrieval
# -----------------------------
def retrieve(
    model: SentenceTransformer,
    index: faiss.Index,
    records: List[Dict[str, Any]],
    query: str,
    top_k: int = 5,
) -> List[Tuple[float, Dict[str, Any]]]:
    q_emb = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q_emb = normalize_rows(q_emb).astype(np.float32)
    scores, idxs = index.search(q_emb, top_k)
    hits = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if idx < 0 or idx >= len(records):
            continue
        hits.append((float(score), records[idx]))
    return hits


def format_hit(score: float, r: Dict[str, Any]) -> str:
    doc_id = r.get("doc_id", "")
    title = r.get("title", "")
    section_id = r.get("section_id", "")
    chunk_id = r.get("chunk_id", "")
    text = r.get("text", "").strip()
    return (
        f"[score={score:.4f}] [doc_id={doc_id} | section_id={section_id} | chunk_id={chunk_id} | title={title}]\n"
        f"{text}"
    )
def format_context(r: Dict[str, Any]) -> str:
    return (
        f"({r.get('doc_id')}:{r.get('section_id')}:{r.get('chunk_id')}) "
        f"{r.get('title','')}\n"
        f"{r.get('text','').strip()}"
    )



# -----------------------------
# Generation (two modes)
# -----------------------------


def pick_dtype():
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 8:
            return torch.bfloat16
    return torch.float16

def load_local_llama(model_id="meta-llama/Llama-3.2-1B-Instruct"):
    dtype = pick_dtype()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=dtype,
    )
    return tok, model


from typing import List

def generate_with_local_llama(tokenizer, model, question: str, context_blocks: List[str]) -> str:
    # 给 LLM 的 context 建议不要带 score，但你暂时先不改也能跑
    context = "\n\n---\n\n".join(context_blocks)

    prompt = (
        "Tu es un assistant service client.\n"
        "Réponds UNIQUEMENT avec les preuves ci-dessous.\n"
        "Si la réponse n'est pas dans les preuves, dis \"Je ne sais pas\".\n"
        "Ajoute des citations au format (doc_id:section_id:chunk_id).\n\n"
        f"Preuves:\n{context}\n\nQuestion:\n{question}\n\nRéponse:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,                 # 先用贪心，最稳定
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][input_len:]  # 只取新生成部分
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text if text else "[WARN] LLM produced empty output."



def generate_fallback(question: str, hits: List[Tuple[float, Dict[str, Any]]]) -> str:
    """
    No-LLM fallback: produce an evidence-first answer.
    This is NOT as fluent, but it proves the RAG pipeline works end-to-end.
    """
    lines = []
    lines.append("Réponse (mode sans LLM) — preuves les plus pertinentes :")
    lines.append(f"Question: {question}\n")
    for score, r in hits:
        doc_id = r.get("doc_id", "")
        section_id = r.get("section_id", "")
        chunk_id = r.get("chunk_id", "")
        title = r.get("title", "")
        text = r.get("text", "").strip().replace("\n", " ")
        lines.append(f"- ({doc_id}:{section_id}:{chunk_id}) {title} — {text}")
    lines.append("\nSi tu veux une réponse rédigée automatiquement, configure OPENAI_API_KEY.")
    return "\n".join(lines)



# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data_with_doc_id", help="Directory containing jsonl files")
    ap.add_argument("--index_dir", type=str, default="rag_index", help="Directory to save/load FAISS index")
    ap.add_argument("--rebuild", action="store_true", help="Rebuild index from data_dir")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--use_llama", action="store_true", help="Use local Llama for generation")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    index_dir = Path(args.index_dir)

    # Load embedding model
    model = SentenceTransformer(args.embed_model)
    tokenizer, llm = load_local_llama("meta-llama/Llama-3.2-1B-Instruct")

    if args.rebuild:
        records = load_jsonl_dir(data_dir)
        if not records:
            raise RuntimeError(f"No records found in {data_dir.resolve()}")

        texts = [build_text_for_embedding(r) for r in records]
        emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        emb = normalize_rows(emb).astype(np.float32)

        index = build_faiss_index(emb)
        save_index(index_dir, index, records, args.embed_model)
        print(f"[OK] Index built. records={len(records)} -> {index_dir.resolve()}")
    else:
        index, records, embed_model_name = load_index(index_dir)
        if embed_model_name and embed_model_name != args.embed_model:
            print(f"[WARN] Index was built with embed_model={embed_model_name}, now using {args.embed_model}")

    # Interactive loop
    index, records, _ = load_index(index_dir)
    print("\nType your question (empty to quit):")
    while True:
        q = input("> ").strip()
        if not q:
            break

        hits = retrieve(model, index, records, q, top_k=args.top_k)

        print("\n=== TOP HITS ===")
        context_blocks = []
        for score, r in hits:
            # 给你看
            print(format_hit(score, r))
            print()
            # 给 LLM 用
            context_blocks.append(format_context(r))

        print("\n=== ANSWER ===")
        ans = generate_with_local_llama(tokenizer, llm, q, context_blocks)
        print(ans)
        print("\n-----------------------------\n")


if __name__ == "__main__":
    main()
