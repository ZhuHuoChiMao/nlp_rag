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

                if "text" not in obj or not obj["text"]:
                    continue
                if "doc_id" not in obj:
                    obj["doc_id"] = fp.stem
                if "title" not in obj:
                    obj["title"] = ""
                records.append(obj)
    return records


def build_text_for_embedding(r: Dict[str, Any]) -> str:
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
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine via inner product on normalized vecs
    index.add(embeddings.astype(np.float32))
    return index


def save_index(index_dir: Path, index: faiss.Index, records: List[Dict[str, Any]], embed_model_name: str):
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.index"))
    with (index_dir / "meta.pkl").open("wb") as f:
        pickle.dump({"records": records, "embed_model_name": embed_model_name}, f)


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
    embedder: SentenceTransformer,
    index: faiss.Index,
    records: List[Dict[str, Any]],
    query: str,
    top_k: int = 5,
) -> List[Tuple[float, Dict[str, Any]]]:
    q_emb = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q_emb = normalize_rows(q_emb).astype(np.float32)
    scores, idxs = index.search(q_emb, top_k)
    hits = []
    for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
        if 0 <= idx < len(records):
            hits.append((float(score), records[idx]))
    return hits


def format_context(r: Dict[str, Any]) -> str:
    return (
        f"({r.get('doc_id')}:{r.get('section_id')}:{r.get('chunk_id')}) "
        f"{r.get('title','')}\n"
        f"{r.get('text','').strip()}"
    )


def format_source(r: Dict[str, Any]) -> str:
    return f"({r.get('doc_id')}:{r.get('section_id')}:{r.get('chunk_id')})"


# -----------------------------
# Llama load + generation
# -----------------------------
def pick_dtype():
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability(0)
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
    # 避免 pad token 报 warning/问题
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.eval()
    return tok, model


def generate_baseline_llama(tokenizer, model, question: str) -> str:
    prompt = (
        "Tu es un assistant service client.\n"
        "Réponds de manière concise.\n\n"
        f"Question:\n{question}\n\nRéponse:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_rag_llama(tokenizer, model, question: str, context_blocks: List[str], sources_line: str) -> str:
    context = "\n\n---\n\n".join(context_blocks)

    prompt = (
        "Tu es un assistant service client.\n"
        "Réponds UNIQUEMENT avec les preuves ci-dessous.\n"
        "Si la réponse n'est pas dans les preuves, dis \"Je ne sais pas\".\n"
        "Ne recopie pas les preuves.\n"
        "Termine par une ligne EXACTE: Sources: (doc_id:section_id:chunk_id), ...\n\n"
        f"Preuves:\n{context}\n\nQuestion:\n{question}\n\nRéponse:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    # 1) 如果模型自己输出了 Sources:，把它和后面全部砍掉（防止把证据原文贴出来）
    if "Sources:" in text:
        text = text.split("Sources:", 1)[0].rstrip()

    # 2) 无论如何都由代码统一追加 Sources（保证就是 top_k 的 sources_line）
    if not text:
        text = "Je ne sais pas"
    text = text + f"\nSources: {sources_line}"
    return text



TEST_QUESTIONS = [
    "Qu’est-ce qu’une facture (standard) ?",
    "Qu’est-ce qu’une facture commerciale (commercial invoice) ?",
    "À quoi sert une facture commerciale pour la douane / l’export ?",
    "Quels documents la boutique peut-elle fournir : facture, facture commerciale, bon cadeau ?",
    "Comment demander une facture PDF au service client ? Quelles informations faut-il envoyer ?",
    "Quelles informations sont nécessaires pour une facture (particulier) vs une facture (professionnel) ?",
    "Puis-je obtenir un bon cadeau / reçu sans prix ? Quand faut-il le demander ?",
]


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
    ap.add_argument("--llama_model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    index_dir = Path(args.index_dir)

    # embedding model
    embedder = SentenceTransformer(args.embed_model)

    # llama model
    tokenizer, llm = load_local_llama(args.llama_model)
    llm.generation_config.temperature = 1.0
    llm.generation_config.top_p = 1.0

    # build/load index
    if args.rebuild:
        records = load_jsonl_dir(data_dir)
        if not records:
            raise RuntimeError(f"No records found in {data_dir.resolve()}")

        texts = [build_text_for_embedding(r) for r in records]
        emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        emb = normalize_rows(emb).astype(np.float32)

        index = build_faiss_index(emb)
        save_index(index_dir, index, records, args.embed_model)
        print(f"[OK] Index built. records={len(records)} -> {index_dir.resolve()}")

    index, records, built_embed_model = load_index(index_dir)
    if built_embed_model and built_embed_model != args.embed_model:
        print(f"[WARN] Index was built with embed_model={built_embed_model}, now using {args.embed_model}")

    # run 7 questions
    for i, q in enumerate(TEST_QUESTIONS, start=1):
        print("\n" + "=" * 80)
        print(f"[Q{i}] {q}")

        print("\n--- BASELINE (No RAG) ---")
        base = generate_baseline_llama(tokenizer, llm, q)
        print(base)

        hits = retrieve(embedder, index, records, q, top_k=args.top_k)

        context_blocks = [format_context(r) for _, r in hits]
        sources_line = ", ".join(format_source(r) for _, r in hits)

        print("\n--- RAG (With sources) ---")
        rag_ans = generate_rag_llama(tokenizer, llm, q, context_blocks, sources_line)
        print(rag_ans)

    print("\nDone.")


if __name__ == "__main__":
    main()

