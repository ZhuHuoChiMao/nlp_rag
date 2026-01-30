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

from rank_bm25 import BM25Okapi
import re

from typing import List, Dict, Any, Tuple


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

def simple_tokenize(text: str):
    text = text.lower()
    # 保留字母数字，其他变空格
    text = re.sub(r"[^a-z0-9À-ÿ]+", " ", text)
    return text.split()

def build_bm25(records):
    corpus_tokens = [simple_tokenize(r.get("text","")) for r in records]
    return BM25Okapi(corpus_tokens)

def minmax_norm(arr, eps=1e-9):
    arr = np.asarray(arr, dtype=np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn < eps:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return (arr - mn) / (mx - mn)


FR_STOP = {
    "le","la","les","un","une","des","du","de","d","l","au","aux",
    "et","ou","mais","donc","or","ni","car",
    "que","qui","quoi","dont","où",
    "est","suis","es","sommes","êtes","sont","été",
    "ce","cet","cette","ces","ça",
    "je","tu","il","elle","on","nous","vous","ils","elles",
    "mon","ma","mes","ton","ta","tes","son","sa","ses","notre","votre","leur","leurs",
    "pour","par","avec","sans","sur","sous","dans","en","à","a",
    "plus","moins","très","trop","pas","ne",
}

def token_set(text: str):
    toks = re.findall(r"[A-Za-zÀ-ÿ0-9]+", text.lower())
    toks = [t for t in toks if len(t) >= 3 and t not in FR_STOP]
    return set(toks)


def overlap_ratio(q: str, ctx: str) -> float:
    qtok = token_set(q)
    if not qtok:
        return 0.0
    ctok = token_set(ctx)
    inter = qtok & ctok
    if len(inter) == 0:
        return 0.0
    return len(inter) / len(qtok)


def has_enough_overlap(q: str, contexts, min_overlap: float = 0.15, check_top: int = 1) -> bool:
    for ctx in contexts[:check_top]:
        if overlap_ratio(q, ctx) >= min_overlap:
            return True
    return False


def should_answer_with_rag(fused_scores, min_best=0.25, min_gap=0.03):
    if fused_scores is None or len(fused_scores) == 0:
        return False
    best = float(fused_scores[0])
    second = float(fused_scores[1]) if len(fused_scores) > 1 else 0.0
    if best < min_best:
        return False
    if (best - second) < min_gap:
        return False
    return True

def size_question_gate(question: str, hits) -> bool:
    q = question.lower()
    if "taille" not in q and "cm" not in q and "kg" not in q:
        return True

    evidence_text = " ".join((r.get("title","") + " " + r.get("text","")) for _, r in hits).lower()

    if "taille" not in evidence_text:
        return False
    if ("cm" not in evidence_text) and ("kg" not in evidence_text) and (not re.search(r"\b\d{2,3}\b", evidence_text)):
        return False
    return True


def retrieve_hybrid(embedder, index, records, bm25, query: str, top_k: int = 5, alpha: float = 0.6):
    cand_k = max(top_k * 5, top_k)
    dense_hits = retrieve(embedder, index, records, query, top_k=cand_k)
    if not dense_hits:
        return []

    dense_scores = np.array([s for s, _ in dense_hits], dtype=np.float32)
    bm25_scores_all = np.array(bm25.get_scores(simple_tokenize(query)), dtype=np.float32)
    bm25_scores = np.array([bm25_scores_all[r["_rid"]] for _, r in dense_hits], dtype=np.float32)

    fused = alpha * minmax_norm(dense_scores) + (1 - alpha) * minmax_norm(bm25_scores)

    order = np.argsort(-fused)[:top_k]
    top_hits = [(float(fused[i]), dense_hits[i][1]) for i in order]
    top_fused = [float(fused[i]) for i in order]

    contexts = [(r.get("title","") + " " + r.get("text","")) for _, r in top_hits]

    if not should_answer_with_rag(top_fused, min_best=0.25, min_gap=0.03):
        return []
    if not has_enough_overlap(query, contexts, min_overlap=0.15, check_top=2):
        return []

    if not size_question_gate(query, top_hits):
        return []

    if not evidence_keyword_gate(query, top_hits):
        return []

    return top_hits




def evidence_keyword_gate(question: str, hits) -> bool:
    q = question.lower()

    if any(k in q for k in ["laver", "lavage", "nettoyer", "entretien", "laine", "pull"]):
        evidence_text = " ".join((r.get("title","") + " " + r.get("text","")) for _, r in hits).lower()

        must = ["laine", "lavage", "laver", "entretien", "nettoyer", "programme", "délicat", "main", "température", "sécher", "eau"]
        hit_cnt = sum(1 for m in must if m in evidence_text)

        if hit_cnt < 2:
            return False
        if ("laine" not in evidence_text) and ("laver" not in evidence_text) and ("lavage" not in evidence_text):
            return False
        return True

    return True




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
            max_new_tokens=250,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_rag_llama(tokenizer, model, question: str, context_blocks: List[str], sources_line: str) -> str:
    context = "\n\n---\n\n".join(context_blocks)
    r'''
    prompt = (
        "Tu es un assistant service client.\n"
        "Réponds UNIQUEMENT avec les preuves ci-dessous.\n"
        "Si la réponse n'est pas dans les preuves, dis \"Je ne sais pas\".\n"
        "Ne recopie pas les preuves.\n"
        "Ne réponds qu'à la question posée. N'ajoute aucune autre question/réponse.\n"
        "Interdiction d'écrire \"Q:\", \"A:\" ou \"R:\".\n"
        "Interdiction d’ajouter des exemples, des pays ou des détails non présents dans les preuves.\n"
        "Termine par une ligne EXACTE: Sources: (doc_id:section_id:chunk_id), ...\n\n"
        f"Preuves:\n{context}\n\nQuestion:\n{question}\n\nRéponse:\n"
    )
    '''
    prompt = (
        "Tu es un assistant service client.\n"
        "Réponds UNIQUEMENT avec les preuves ci-dessous.\n"
        "Si la réponse n'est pas dans les preuves, dis \"Je ne sais pas\".\n"
        "Ne recopie pas les preuves.\n"
        "Réponds en 2 phrases maximum.\n"
        "Sans liste numérotée, sans puces.\n"
        f"Preuves:\n{context}\n\nQuestion:\n{question}\n\nRéponse:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    text = re.sub(r"(?im)^\s*(Q|A)\s*:\s*", "", text)
    text = re.sub(r"(?im)^\s*R\s*:\s*", "", text)
    text = re.split(r"\n\s*(Q\s*:|R\s*:|J'ai besoin d')", text, maxsplit=1)[0].strip()
    text = re.sub(r"\n\s*\d+\.\s*$", "", text).strip()

    def clip_chars(t: str, max_chars: int = 320) -> str:
        t = t.strip()
        if len(t) <= max_chars:
            return t
        cut_text = t[:max_chars]
        cut_pos = max(
            cut_text.rfind("\n"),
            cut_text.rfind("."),
            cut_text.rfind("!"),
            cut_text.rfind("?"),
            cut_text.rfind(";"),
            cut_text.rfind(":"),
        )

        if cut_pos >= 80:
            cut_text = cut_text[:cut_pos + 1]
        else:
            cut_text = cut_text.rsplit(" ", 1)[0]

        return cut_text.strip()

    text = clip_chars(text, 360)

    if "Sources:" in text:
        text = text.split("Sources:", 1)[0].rstrip()

    if not text:
        text = "Je ne sais pas"
    text = text
    return text



TEST_QUESTIONS = [
    "Qu’est-ce qu’une facture standard ?",
    "Qu’est-ce qu’une facture commerciale ?",
    "Puis-je obtenir un reçu sans prix ou bon cadeau ? Quand dois-je le demander ?",
    "Je fais 170 cm et 60 kg, je prends quelle taille ?",
    "Comment laver mon nouveau pull en laine ?",
    "Je souhaite retourner l'article.",
    "Qu'est-ce que l'amour?",
]



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

    embedder = SentenceTransformer(args.embed_model)

    tokenizer, llm = load_local_llama(args.llama_model)
    llm.generation_config.temperature = 1.0
    llm.generation_config.top_p = 1.0

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
    for i, r in enumerate(records):
        r["_rid"] = i
    bm25 = build_bm25(records)

    if built_embed_model and built_embed_model != args.embed_model:
        print(f"[WARN] Index was built with embed_model={built_embed_model}, now using {args.embed_model}")


    for i, q in enumerate(TEST_QUESTIONS, start=1):
        print("\n" + "=" * 80)
        print(f"[Q{i}] {q}")

        print("\n--- BASELINE (No RAG) ---")
        base = generate_baseline_llama(tokenizer, llm, q)
        print(base)


        hits = retrieve_hybrid(embedder, index, records, bm25, q, top_k=args.top_k,alpha=0.6)

        print("\n--- RETRIEVAL (Top hits) ---")
        if not hits:
            print("(no reliable evidence)")
            print("\n--- RAG (With sources) ---")
            print("Il s'agit d'un problème complexe qui est en cours de transfert au service client.")
            continue

        for score, r in hits:
            print(
                f"- score={score:.4f} doc_id={r.get('doc_id')} section={r.get('section_id')} chunk={r.get('chunk_id')} title={r.get('title', '')}")


        context_blocks = [format_context(r) for _, r in hits]
        sources_line = ", ".join(format_source(r) for _, r in hits)

        print("\n--- RAG (With sources) ---")
        rag_ans = generate_rag_llama(tokenizer, llm, q, context_blocks, sources_line)
        print(rag_ans)



if __name__ == "__main__":
    main()

