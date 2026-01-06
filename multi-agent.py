#!/usr/bin/env python3
"""
multi-agent.py (amendment-aware, pre-aggregation) — MODIFIED to re-use stored chunk embeddings in Neo4j.

Key changes vs original:
- GraphRAG Step 5 chunk rerank no longer embeds chunk text each run.
- GraphRAG chunk collection no longer relies on LangChain pickle ChunkStore by default.
  It fetches TextChunk.content + TextChunk.embedding from Neo4j using (document_id, chunk_id) with fallbacks.
- NaiveRAG already used Neo4j vector index on TextChunk.embedding; left as-is (it was already efficient for chunks).

Still embeds (per run):
- The user query (needed for vector search / similarity).
- Query triples + entities (needed for triple/entity retrieval).

Neo4j expectations (unchanged):
- (:Triple {embedding}) vector index: 'triple_vec'
- Optional entity vector indexes: 'document_vec', 'content_vec', 'expression_vec'
- (:TextChunk {embedding, uu_number, content, document_id, chunk_id, pages}) vector index: 'chunk_embedding_index'
"""

import os, time, json, math, pickle, re, random, hashlib, threading, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set
from threading import Lock, Semaphore, Thread

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from neo4j import GraphDatabase

# ----------------- Load .env (two common locations: repo root and local) -----------------
_here = Path(__file__).resolve().parent
_default_env1 = _here / ".env"
_default_env2 = _here / ".env"
for _envp in (_default_env1, _default_env2):
    if _envp.exists():
        load_dotenv(dotenv_path=_envp)
        break

# ----------------- Config -----------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is required in environment")

NEO4J_URI  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")

# Neo4j retry/timeout controls
NEO4J_TX_TIMEOUT_S = float(os.getenv("NEO4J_TX_TIMEOUT_S", "60"))
NEO4J_MAX_ATTEMPTS = int(os.getenv("NEO4J_MAX_ATTEMPTS", "100000000"))
NEO4J_MAX_CONCURRENCY = int(os.getenv("NEO4J_MAX_CONCURRENCY", "0"))  # 0=unlimited

# Models
GEN_MODEL   = os.getenv("GEN_MODEL", "models/gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

# Dataset folder for original chunk pickles (GraphRAG ChunkStore) — now optional for fallback only
DEFAULT_LANGCHAIN_DIR = (_here / "../../../dataset/3_indexing/3a_langchain_results/").resolve()
LANGCHAIN_DIR = Path(os.getenv("LANGCHAIN_DIR") or str(DEFAULT_LANGCHAIN_DIR))
SKIP_FILES = {"all_langchain_documents.pkl"}

# GraphRAG parameters
ENTITY_MATCH_TOP_K = int(os.getenv("ENTITY_MATCH_TOP_K", "10"))
ENTITY_SUBGRAPH_HOPS = int(os.getenv("ENTITY_SUBGRAPH_HOPS", "3"))
ENTITY_SUBGRAPH_PER_HOP_LIMIT = int(os.getenv("ENTITY_SUBGRAPH_PER_HOP_LIMIT", "2000"))
SUBGRAPH_TRIPLES_TOP_K = int(os.getenv("SUBGRAPH_TRIPLES_TOP_K", "30"))
QUERY_TRIPLE_MATCH_TOP_K_PER = int(os.getenv("QUERY_TRIPLE_MATCH_TOP_K_PER", "20"))
MAX_TRIPLES_FINAL = int(os.getenv("MAX_TRIPLES_FINAL", "60"))
MAX_CHUNKS_FINAL = int(os.getenv("MAX_CHUNKS_FINAL", "40"))
CHUNK_RERANK_CAND_LIMIT = int(os.getenv("CHUNK_RERANK_CAND_LIMIT", "10000000"))
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "4096"))
MAX_ANSWER_JUDGE_ITERS = int(os.getenv("MAX_ANSWER_JUDGE_ITERS", "3"))
AJ_ANSWER_MAX_CHARS = int(os.getenv("AJ_ANSWER_MAX_CHARS", "400000000"))

# NaiveRAG parameters
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "40"))
NAIVE_MAX_CHUNKS_FINAL = int(os.getenv("NAIVE_MAX_CHUNKS_FINAL", str(MAX_CHUNKS_FINAL)))
CHUNK_TEXT_CLAMP = int(os.getenv("CHUNK_TEXT_CLAMP", "200000"))

# Global LLM throttling (concurrency + QPS)
LLM_EMBED_MAX_CONCURRENCY = max(1, int(os.getenv("LLM_EMBED_MAX_CONCURRENCY", "165")))
LLM_EMBED_QPS = float(os.getenv("LLM_EMBED_QPS", "165.0"))
LLM_GEN_MAX_CONCURRENCY   = max(1, int(os.getenv("LLM_GEN_MAX_CONCURRENCY", "100")))
LLM_GEN_QPS   = float(os.getenv("LLM_GEN_QPS", "1.0"))

# Embedding cache cap
CACHE_EMBED_MAX_ITEMS = int(os.getenv("CACHE_EMBED_MAX_ITEMS", "200000"))

# Aggregator
AGG_TEMPERATURE = float(os.getenv("AGG_TEMPERATURE", "0.2"))

# ----------------- Initialize SDKs -----------------
genai.configure(api_key=GOOGLE_API_KEY)
gen_model = genai.GenerativeModel(GEN_MODEL)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ----------------- Logger -----------------
_LOGGER = None
_LOG_TL = threading.local()  # thread-local for pipeline/iteration tags

def set_log_context(pipeline_tag: Optional[str] = None, iter_tag: Optional[str] = None):
    setattr(_LOG_TL, "pipeline_tag", pipeline_tag or None)
    setattr(_LOG_TL, "iter_tag", iter_tag or None)

def _now_ts() -> str:
    t = time.time()
    base = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    ms = int((t % 1) * 1000)
    return f"{base}.{ms:03d}"

def _pid() -> int:
    try:
        return os.getpid()
    except Exception:
        return -1

def _prefix(level: str = "INFO") -> str:
    ptag = getattr(_LOG_TL, "pipeline_tag", None)
    itag = getattr(_LOG_TL, "iter_tag", None)
    p_part = f" [pipe={ptag}]" if ptag else ""
    i_part = f" [iter={itag}]" if itag else ""
    return f"[{_now_ts()}] [{level}]{p_part}{i_part} [pid={_pid()}]"

class FileLogger:
    def __init__(self, file_path: Path, also_console: bool = True):
        self.file_path = file_path
        self.also_console = also_console
        self._fh = open(file_path, "w", encoding="utf-8")
        self._lock = Lock()

    def log(self, msg: Any = ""):
        if not isinstance(msg, str):
            try:
                msg = json.dumps(msg, ensure_ascii=False, default=str)
            except Exception:
                msg = str(msg)
        out = msg + "\n"
        with self._lock:
            self._fh.write(out)
            self._fh.flush()
            if self.also_console:
                try:
                    sys.stdout.write(out)
                    sys.stdout.flush()
                except Exception:
                    pass

    def close(self):
        try:
            with self._lock:
                self._fh.flush()
                self._fh.close()
        except Exception:
            pass

def log(msg: Any = "", level: str = "INFO"):
    global _LOGGER
    if not isinstance(msg, str):
        try:
            msg = json.dumps(msg, ensure_ascii=False, default=str)
        except Exception:
            msg = str(msg)
    lines = str(msg).splitlines() or [str(msg)]
    out = "\n".join([f"{_prefix(level)} {ln}" for ln in lines])
    if _LOGGER is not None:
        _LOGGER.log(out)
    else:
        print(out, flush=True)

def make_timestamp_name() -> str:
    t = time.time()
    base = time.strftime("%Y%m%d-%H%M%S", time.localtime(t))
    ms = int((t % 1) * 1000)
    return f"{base}-{ms:03d}"

# ----------------- Utilities -----------------
def now_ms() -> float:
    return time.time()

def dur_ms(start: float) -> float:
    return (time.time() - start) * 1000.0

def _norm_id(x) -> str:
    return str(x).strip() if x is not None else ""

def estimate_tokens_for_text(text: str) -> int:
    return max(1, int(len(text) / 4))  # naive heuristic

def _as_float_list(vec) -> List[float]:
    if vec is None:
        return []
    try:
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
    except Exception:
        pass
    try:
        return [float(x) for x in list(vec)]
    except Exception:
        try:
            return [float(vec)]
        except Exception:
            return []

def _rand_wait_seconds(low=5.0, high=10.0) -> float:
    return random.uniform(low, high)

# ----------------- Global rate limiters -----------------
class QpsLimiter:
    def __init__(self, qps: float):
        self.qps = max(0.0, float(qps))
        self._min_interval = 1.0 / self.qps if self.qps > 0 else 0.0
        self._lock = Lock()
        self._next_time = 0.0

    def acquire(self):
        if self.qps <= 0:
            return
        with self._lock:
            now = time.time()
            if now < self._next_time:
                time.sleep(self._next_time - now)
                now = time.time()
            self._next_time = now + self._min_interval

_EMBED_SEM = Semaphore(LLM_EMBED_MAX_CONCURRENCY)
_GEN_SEM   = Semaphore(LLM_GEN_MAX_CONCURRENCY)
_EMBED_QPS = QpsLimiter(LLM_EMBED_QPS)
_GEN_QPS   = QpsLimiter(LLM_GEN_QPS)
_NEO4J_SEM = Semaphore(NEO4J_MAX_CONCURRENCY) if NEO4J_MAX_CONCURRENCY > 0 else None

# Embedding cache (for texts we still embed: query, query triples, entities)
_EMB_CACHE: Dict[str, List[float]] = {}
_EMB_CACHE_LOCK = Lock()

def _cache_key_for_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

# ----------------- Language detection (ID vs EN) -----------------
def detect_user_language(text: str) -> str:
    t = (text or "").lower()
    if re.search(r"\b(pasal|undang[- ]?undang|uu\s*\d|peraturan|menteri|ayat|bab|bagian|paragraf|ketentuan|sebagaimana|dimaksud)\b", t):
        return "id"
    if re.search(r"\b(article|act|law|regulation|minister|section|paragraph|chapter|pursuant|provided that)\b", t):
        return "en"
    id_tokens = {"yang","dan","atau","tidak","adalah","berdasarkan","sebagaimana","pada","dalam","dapat","harus","wajib",
                 "pasal","undang","peraturan","menteri","ayat","bab","bagian","paragraf","ketentuan","pengundangan","apabila","jika"}
    en_tokens = {"the","and","or","not","is","based","as","provided","pursuant","in","may","must","shall",
                 "article","act","law","regulation","minister","section","paragraph","chapter","whereas"}
    words = re.findall(r"[a-z]+", t)
    score_id = sum(1 for w in words if w in id_tokens)
    score_en = sum(1 for w in words if w in en_tokens)
    if score_id > score_en:
        return "id"
    if score_en > score_id:
        return "en"
    return "en"

# ----------------- Safe LLM helpers -----------------
def get_finish_info(resp) -> Dict[str, Any]:
    info = {}
    try:
        cand = resp.candidates[0] if resp.candidates else None
        if cand:
            info["finish_reason"] = getattr(cand, "finish_reason", None)
            safety = []
            try:
                for sr in getattr(cand, "safety_ratings", []) or []:
                    safety.append({"category": getattr(sr, "category", None), "prob": getattr(sr, "probability", None)})
            except Exception:
                pass
            info["safety_ratings"] = safety
    except Exception:
        pass
    return info

def extract_text_from_response(resp) -> Optional[str]:
    try:
        text = resp.text
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        pass
    try:
        for cand in (resp.candidates or []):
            parts = getattr(cand, "content", None)
            if parts and getattr(parts, "parts", None):
                buf = []
                for p in parts.parts:
                    t = getattr(p, "text", None)
                    if isinstance(t, str):
                        buf.append(t)
                if buf:
                    return "\n".join(buf).strip()
    except Exception:
        pass
    return None

def _api_call_with_retry(func, *args, **kwargs):
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait_s = _rand_wait_seconds()
            log(f"[Retry] API call failed: {e}. Retrying in {wait_s:.1f}s.", level="WARN")
            time.sleep(wait_s)

def embed_text(text: str) -> List[float]:
    """Used only for query / query-triple / entity embeddings (not chunk text reranking)."""
    key = _cache_key_for_text(text)
    with _EMB_CACHE_LOCK:
        if key in _EMB_CACHE:
            return list(_EMB_CACHE[key])
    with _EMBED_SEM:
        _EMBED_QPS.acquire()
        t0 = now_ms()
        res = _api_call_with_retry(genai.embed_content, model=EMBED_MODEL, content=text)
    vec = None
    if isinstance(res, dict):
        emb = res.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            vec = emb["values"]
        elif isinstance(emb, list):
            vec = emb
    if vec is None:
        try:
            vec = res.embedding.values  # type: ignore[attr-defined]
        except Exception:
            pass
    if vec is None:
        raise RuntimeError("Unexpected embedding response shape for embeddings")
    out = _as_float_list(vec)
    log(f"[Embed] text_len={len(text)} -> vec_len={len(out)} | {dur_ms(t0):.0f} ms", level="DEBUG")
    with _EMB_CACHE_LOCK:
        if len(_EMB_CACHE) >= CACHE_EMBED_MAX_ITEMS:
            try:
                _EMB_CACHE.pop(next(iter(_EMB_CACHE)))
            except Exception:
                _EMB_CACHE.clear()
        _EMB_CACHE[key] = list(out)
    return out

def safe_generate_json(prompt: str, schema: Dict[str, Any], temp: float = 0.0) -> Dict[str, Any]:
    cfg = GenerationConfig(
        temperature=temp,
        response_mime_type="application/json",
        response_schema=schema,
    )
    with _GEN_SEM:
        _GEN_QPS.acquire()
        t0 = now_ms()
        resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    log(f"[LLM JSON] call completed in {dur_ms(t0):.0f} ms", level="DEBUG")
    try:
        if isinstance(resp.text, str) and resp.text.strip():
            return json.loads(resp.text)
    except Exception:
        pass
    try:
        raw = resp.candidates[0].content.parts[0].text
        return json.loads(raw)
    except Exception as e:
        info = get_finish_info(resp)
        log(f"[LLM JSON parse warning] No JSON content returned. Diagnostics: {info}. Error: {e}", level="WARN")
        return {}

def safe_generate_text(prompt: str, max_tokens: int, temperature: float = 0.2) -> str:
    cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
    with _GEN_SEM:
        _GEN_QPS.acquire()
        t0 = now_ms()
        resp = _api_call_with_retry(gen_model.generate_content, prompt, generation_config=cfg)
    took = dur_ms(t0)
    text = extract_text_from_response(resp)
    if text is not None and text.strip():
        log(f"[LLM TEXT] call completed in {took:.0f} ms, len={len(text)}", level="DEBUG")
        return text.strip()
    info = get_finish_info(resp)
    log(f"[LLM text warning] No text returned. Diagnostics: {info}. Took={took:.0f} ms", level="WARN")
    return f"(Model returned no text. finish_info={info})"

# ----------------- Neo4j helpers -----------------
_NEO4J_QUERY_SEQ = 0
_NEO4J_QUERY_LOCK = Lock()

def _next_query_id() -> int:
    global _NEO4J_QUERY_SEQ
    with _NEO4J_QUERY_LOCK:
        _NEO4J_QUERY_SEQ += 1
        return _NEO4J_QUERY_SEQ

def _summarize_params(params: Dict[str, Any]) -> str:
    parts = []
    for k, v in (params or {}).items():
        try:
            if k.lower() in ("q_emb", "q", "embedding", "emb"):
                if isinstance(v, (list, tuple)):
                    parts.append(f"{k}=list(len={len(v)})")
                else:
                    parts.append(f"{k}=vector")
            elif isinstance(v, list):
                parts.append(f"{k}=list(len={len(v)})")
            elif isinstance(v, str) and len(v) > 60:
                parts.append(f"{k}=str(len={len(v)})")
            else:
                parts.append(f"{k}={v}")
        except Exception:
            parts.append(f"{k}=<?>")
    return ", ".join(parts)

def run_cypher_with_retry(cypher: str, params: Dict[str, Any]) -> List[Any]:
    attempts = 0
    last_e: Optional[Exception] = None
    qid = _next_query_id()
    preview = " ".join((cypher or "").split())
    if len(preview) > 220:
        preview = preview[:220] + "..."
    param_summary = _summarize_params(params)
    while attempts < max(1, NEO4J_MAX_ATTEMPTS):
        attempts += 1
        t0 = now_ms()
        log(f"[Neo4j] Attempt {attempts}/{NEO4J_MAX_ATTEMPTS} | qid={qid} | timeout={NEO4J_TX_TIMEOUT_S:.1f}s | Cypher=\"{preview}\" | Params: {param_summary}")
        try:
            if _NEO4J_SEM is not None:
                _NEO4J_SEM.acquire()
            try:
                with driver.session() as session:
                    res = session.run(cypher, **params, timeout=NEO4J_TX_TIMEOUT_S)
                    records = list(res)
            finally:
                if _NEO4J_SEM is not None:
                    _NEO4J_SEM.release()
            took = dur_ms(t0)
            log(f"[Neo4j] Success | qid={qid} | rows={len(records)} | {took:.0f} ms")
            return records
        except Exception as e:
            took = dur_ms(t0)
            last_e = e
            if attempts >= NEO4J_MAX_ATTEMPTS:
                break
            wait_s = random.uniform(5, 10)
            log(f"[Neo4j] Failure | qid={qid} | attempt={attempts}/{NEO4J_MAX_ATTEMPTS} | {took:.0f} ms | error={e}. Retrying in {wait_s:.1f}s.", level="WARN")
            time.sleep(wait_s)
    raise RuntimeError(f"Neo4j query failed after {NEO4J_MAX_ATTEMPTS} attempts (qid={qid}): {last_e}")

def cos_sim(a: List[float], b: List[float]) -> float:
    a = _as_float_list(a); b = _as_float_list(b)
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

# ----------------- TextChunk fetch (NEW): retrieve stored chunk embeddings/content from Neo4j -----------------
def fetch_textchunk_by_doc_chunk(document_id: Any, chunk_id: Any) -> Optional[Dict[str, Any]]:
    """
    Fetch TextChunk node by (document_id, chunk_id) with fallbacks:
    1) exact (document_id, chunk_id)
    2) base chunk_id if chunk_id contains '::...'
    3) chunk_id only (doc mismatch rescue)
    Returns dict: {content, embedding, key, document_id?, match}
    """
    doc = _norm_id(document_id)
    ch = _norm_id(chunk_id)
    if not doc or not ch:
        return None

    cypher1 = """
    MATCH (c:TextChunk {document_id:$doc, chunk_id:$chunk})
    RETURN c.content AS content, c.embedding AS embedding, c.key AS key
    LIMIT 1
    """
    res = run_cypher_with_retry(cypher1, {"doc": doc, "chunk": ch})
    if res:
        r = res[0]
        return {"content": r["content"], "embedding": r["embedding"], "key": r["key"], "match": "exact"}

    if "::" in ch:
        base = ch.split("::", 1)[0]
        res = run_cypher_with_retry(cypher1, {"doc": doc, "chunk": base})
        if res:
            r = res[0]
            return {"content": r["content"], "embedding": r["embedding"], "key": r["key"], "match": "base_id"}

    cypher3 = """
    MATCH (c:TextChunk {chunk_id:$chunk})
    RETURN c.content AS content, c.embedding AS embedding, c.key AS key, c.document_id AS document_id
    LIMIT 1
    """
    res = run_cypher_with_retry(cypher3, {"chunk": ch})
    if res:
        r = res[0]
        return {
            "content": r["content"],
            "embedding": r["embedding"],
            "key": r["key"],
            "document_id": r["document_id"],
            "match": "chunk_id_only"
        }

    return None

# ----------------- ChunkStore (GraphRAG, now optional fallback) -----------------
class ChunkStore:
    def __init__(self, root: Path, skip: Set[str]):
        self.root = root
        self.skip = skip
        self._index: Dict[Tuple[str, str], str] = {}
        self._by_chunk: Dict[str, List[Tuple[str, str]]] = {}
        self._loaded_files: Set[Path] = set()
        self._built = False
        self._build_lock = Lock()

    def _build_index(self):
        if self._built:
            return
        with self._build_lock:
            if self._built:
                return
            start = time.monotonic()
            log(f"[ChunkStore] Building index from {self.root}...")
            pkls = [p for p in self.root.glob("*.pkl") if p.name not in self.skip]
            total_chunks_indexed = 0
            for pkl in pkls:
                try:
                    with open(pkl, "rb") as f:
                        chunks = pickle.load(f)
                    loaded_count = 0
                    for ch in chunks:
                        meta = getattr(ch, "metadata", {}) if hasattr(ch, "metadata") else {}
                        doc_id = _norm_id(meta.get("document_id"))
                        chunk_id = _norm_id(meta.get("chunk_id"))
                        text = getattr(ch, "page_content", None)
                        if doc_id and chunk_id and isinstance(text, str):
                            self._index[(doc_id, chunk_id)] = text
                            self._by_chunk.setdefault(chunk_id, []).append((doc_id, chunk_id))
                            loaded_count += 1
                    self._loaded_files.add(pkl)
                    total_chunks_indexed += loaded_count
                    log(f"[ChunkStore] Loaded {loaded_count} chunks from {pkl.name}")
                except Exception as e:
                    log(f"[ChunkStore] Failed to load or process {pkl.name}: {e}", level="WARN")
                    continue
            elapsed = time.monotonic() - start
            log(f"[ChunkStore] Index built. Total chunks indexed: {total_chunks_indexed} from {len(self._loaded_files)} files in {elapsed:.3f}s.")
            self._built = True

    def get_chunk(self, document_id: Any, chunk_id: Any) -> Optional[str]:
        if not self._built:
            self._build_index()
        doc_id_s = _norm_id(document_id)
        chunk_id_s = _norm_id(chunk_id)
        val = self._index.get((doc_id_s, chunk_id_s))
        if val is not None:
            log(f"[ChunkStore] HIT exact: doc={doc_id_s} chunk={chunk_id_s} len={len(val)}", level="DEBUG")
            return val
        if "::" in chunk_id_s:
            base_id = chunk_id_s.split("::", 1)[0]
            val = self._index.get((doc_id_s, base_id))
            if val is not None:
                log(f"[ChunkStore] HIT base-id: doc={doc_id_s} chunk={chunk_id_s} -> base={base_id} len={len(val)}", level="DEBUG")
                return val
        matches = self._by_chunk.get(chunk_id_s)
        if matches:
            chosen_doc, chosen_chunk = matches[0]
            val = self._index.get((chosen_doc, chosen_chunk))
            if val is not None:
                note = "" if len(matches) == 1 else f" (warn: chunk_id occurs in {len(matches)} docs; chose doc={chosen_doc})"
                log(f"[ChunkStore] HIT by chunk_id only: requested doc={doc_id_s} chunk={chunk_id_s}; using doc={chosen_doc}{note}. len={len(val)}", level="WARN")
                return val
        log(f"[ChunkStore] MISS: doc={doc_id_s} chunk={chunk_id_s}", level="DEBUG")
        return None

# ----------------- Agents (shared) -----------------
LEGAL_ENTITY_TYPES = [
    "UU", "PASAL", "AYAT", "INSTANSI", "ORANG", "ISTILAH", "SANKSI", "NOMINAL", "TANGGAL"
]
LEGAL_PREDICATES = [
    "mendefinisikan", "mengubah", "mencabut", "mulai_berlaku", "mewajibkan",
    "melarang", "memberikan_sanksi", "berlaku_untuk", "termuat_dalam",
    "mendelegasikan_kepada", "berjumlah", "berdurasi"
]

PIPELINE_BRIEF = """
GraphRAG pipeline summary:
1) Agent 1 extracts legal entities/predicates from the query (e.g., UU numbers, Pasal/Ayat).
2) Agent 1b extracts query triples: subject [predicate] object.
3) Triple-centric retrieval: embed "s [p] o" and query a triple_vec index to find similar KG triples.
4) Entity-centric retrieval: embed key entities, match similar KG entities via vector indexes, expand a subgraph to collect triples.
5) Merge triples, rerank by similarity to the query/triples.
6) Collect candidate document chunks for those triples (doc_id/chunk_id), then score chunks by cosine similarity
   to the whole query using stored TextChunk.embedding (no per-run chunk embedding).
7) Answerer answers strictly from selected chunks.
""".strip()

def _truncate(s: str, max_chars: int) -> str:
    if not isinstance(s, str):
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20] + " ...[truncated]"

# --- Agent 1: entities/predicates ---
QUERY_SCHEMA = {
  "type": "object",
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {"text": {"type":"string"}, "type": {"type":"string"}},
        "required": ["text"]
      }
    },
    "predicates": {"type": "array", "items": {"type": "string"}}
  },
  "required": ["entities","predicates"]
}

def agent1_extract_entities_predicates(query: str) -> Dict[str, Any]:
    prompt = f"""
You are Agent 1. Extract the legal entities and predicates mentioned or implied by the user's question.

Output JSON:
- "entities": array of {{text, type(optional in {LEGAL_ENTITY_TYPES})}}
- "predicates": array of strings (Indonesian; snake_case if multiword)

User question:
\"\"\"{query}\"\"\"
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent 1] Prompt:"); log(prompt)
    log(f"[Agent 1] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")
    out = safe_generate_json(prompt, QUERY_SCHEMA, temp=0.0) or {}
    if "entities" not in out: out["entities"] = []
    if "predicates" not in out: out["predicates"] = []
    log(f"[Agent 1] Output: entities={ [e.get('text') for e in out['entities']] }, predicates={ out['predicates'] }")
    return out

# --- Agent 1b: triples from query ---
QUERY_TRIPLES_SCHEMA = {
    "type": "object",
    "properties": {
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "object",
                        "properties": {"text":{"type":"string"}, "type":{"type":"string"}},
                        "required": ["text"]
                    },
                    "predicate": {"type": "string"},
                    "object": {
                        "type": "object",
                        "properties": {"text":{"type":"string"}, "type":{"type":"string"}},
                        "required": ["text"]
                    }
                },
                "required": ["subject","predicate","object"]
            }
        }
    },
    "required": ["triples"]
}

def agent1b_extract_query_triples(query: str) -> List[Dict[str, Any]]:
    prompt = f"""
You are Agent 1b. Extract explicit or implied triples from the user's question in the form:
subject — predicate — object.

Rules:
- Use short, literal subject/object texts as they appear in the question.
- Predicates should be concise (lowercase, snake_case if multiword).
- If type is unknown, leave it blank.
- Do not invent or speculate.

Return JSON with key "triples".
User question:
\"\"\"{query}\"\"\"
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent 1b] Prompt:"); log(prompt)
    log(f"[Agent 1b] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")
    out = safe_generate_json(prompt, QUERY_TRIPLES_SCHEMA, temp=0.0) or {}
    triples = out.get("triples", []) if isinstance(out, dict) else []
    clean: List[Dict[str, Any]] = []
    for t in triples:
        try:
            s = (t.get("subject") or {}).get("text","").strip()
            p = (t.get("predicate") or "").strip()
            o = (t.get("object") or {}).get("text","").strip()
            if s and p and o:
                clean.append({
                    "subject": {"text": s, "type": (t.get("subject") or {}).get("type","").strip()},
                    "predicate": p,
                    "object":  {"text": o, "type": (t.get("object")  or {}).get("type","").strip()},
                })
        except Exception:
            pass
    log(f"[Agent 1b] Extracted query triples: {['{} [{}] {}'.format(x['subject']['text'], x['predicate'], x['object']['text']) for x in clean]}")
    return clean

def query_triple_to_text(t: Dict[str, Any]) -> str:
    s = ((t.get("subject") or {}).get("text") or "").strip()
    p = (t.get("predicate") or "").strip()
    o = ((t.get("object") or {}).get("text") or "").strip() if isinstance(t.get("object"), dict) else ((t.get("object") or "").strip())
    return f"{s} [{p}] {o}"

# ----------------- GraphRAG retrieval -----------------
def _vector_query_nodes(index_name: str, q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    q_emb = _as_float_list(q_emb)
    cypher = """
    CALL db.index.vector.queryNodes($index_name, $k, $q_emb) YIELD node AS n, score
    RETURN n, score, elementId(n) AS elem_id
    ORDER BY score DESC
    LIMIT $k
    """
    res = run_cypher_with_retry(cypher, {"index_name": index_name, "k": k, "q_emb": q_emb})
    rows = []
    for r in res:
        n = r["n"]
        rows.append({
            "key": n.get("key"),
            "elem_id": r["elem_id"],
            "name": n.get("name"),
            "type": n.get("type"),
            "score": r["score"],
        })
    return rows

def search_similar_entities_by_embedding(q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    try:
        candidates.extend(_vector_query_nodes("document_vec", q_emb, k))
    except Exception as e:
        log(f"[Warn] document_vec query failed: {e}", level="WARN")
    try:
        candidates.extend(_vector_query_nodes("content_vec", q_emb, k))
    except Exception as e:
        log(f"[Warn] content_vec query failed: {e}", level="WARN")
    try:
        candidates.extend(_vector_query_nodes("expression_vec", q_emb, k))
    except Exception as e:
        log(f"[Warn] expression_vec query failed: {e}", level="WARN")

    best: Dict[str, Dict[str, Any]] = {}
    for row in candidates:
        dedup_key = row.get("elem_id") or f"{row.get('key')}|{row.get('type')}"
        if dedup_key not in best or (row.get("score", -1) > best[dedup_key].get("score", -1)):
            best[dedup_key] = row

    merged = list(best.values())
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return merged[:k]

def search_similar_triples_by_embedding(q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    q_emb = _as_float_list(q_emb)
    cypher = """
    CALL db.index.vector.queryNodes('triple_vec', $k, $q_emb) YIELD node AS tr, score
    OPTIONAL MATCH (tr)-[:SUBJECT]->(s)
    OPTIONAL MATCH (tr)-[:OBJECT]->(o)
    RETURN tr, s, o, score
    ORDER BY score DESC
    LIMIT $k
    """
    res = run_cypher_with_retry(cypher, {"k": k, "q_emb": q_emb})
    rows = []
    for r in res:
        tr = r["tr"]; s = r["s"]; o = r["o"]
        rows.append({
            "triple_uid": tr.get("triple_uid"),
            "predicate": tr.get("predicate"),
            "uu_number": tr.get("uu_number"),
            "evidence_quote": tr.get("evidence_quote"),
            "subject": s.get("name") if s else None,
            "subject_key": s.get("key") if s else None,
            "subject_type": s.get("type") if s else None,
            "object": o.get("name") if o else None,
            "object_key": o.get("key") if o else None,
            "object_type": o.get("type") if o else None,
            "score": r["score"],
            "embedding": tr.get("embedding"),
            "document_id": tr.get("document_id"),
            "chunk_id": tr.get("chunk_id"),
            "evidence_article_ref": tr.get("evidence_article_ref"),
        })
    return rows

def expand_from_entities(entity_keys: List[str], hops: int, per_hop_limit: int, entity_elem_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    triples: Dict[str, Dict[str, Any]] = {}
    current_ids: Set[str] = set(x for x in (entity_elem_ids or []) if x)
    current_keys: Set[str] = set(x for x in (entity_keys or []) if x)

    for _ in range(hops):
        if not current_ids and not current_keys:
            break

        if current_ids:
            cypher = """
            UNWIND $ids AS eid
            MATCH (e) WHERE elementId(e) = eid
            MATCH (e)-[r:PREDICATE]->()
            WITH DISTINCT r.triple_uid AS uid LIMIT $limit
            MATCH (tr:Triple {triple_uid: uid})
            OPTIONAL MATCH (tr)-[:SUBJECT]->(s)
            OPTIONAL MATCH (tr)-[:OBJECT]->(o)
            RETURN tr, s, o, elementId(s) AS s_id, elementId(o) AS o_id
            """
            params = {"ids": list(current_ids), "limit": per_hop_limit}
        else:
            cypher = """
            UNWIND $keys AS k
            MATCH (e:Entity {key:k})-[r:PREDICATE]->()
            WITH DISTINCT r.triple_uid AS uid LIMIT $limit
            MATCH (tr:Triple {triple_uid: uid})
            OPTIONAL MATCH (tr)-[:SUBJECT]->(s)
            OPTIONAL MATCH (tr)-[:OBJECT]->(o)
            RETURN tr, s, o, elementId(s) AS s_id, elementId(o) AS o_id
            """
            params = {"keys": list(current_keys), "limit": per_hop_limit}

        res = run_cypher_with_retry(cypher, params)
        next_ids: Set[str] = set()
        next_keys: Set[str] = set()
        for r in res:
            tr = r["tr"]; s = r["s"]; o = r["o"]
            uid = tr.get("triple_uid")
            if uid not in triples:
                triples[uid] = {
                    "triple_uid": uid,
                    "predicate": tr.get("predicate"),
                    "uu_number": tr.get("uu_number"),
                    "evidence_quote": tr.get("evidence_quote"),
                    "embedding": tr.get("embedding"),
                    "document_id": tr.get("document_id"),
                    "chunk_id": tr.get("chunk_id"),
                    "evidence_article_ref": tr.get("evidence_article_ref"),
                    "subject": s.get("name") if s else None,
                    "subject_key": s.get("key") if s else None,
                    "subject_type": s.get("type") if s else None,
                    "object": o.get("name") if o else None,
                    "object_key": o.get("key") if o else None,
                    "object_type": o.get("type") if o else None,
                }
            if s and s.get("key"): next_keys.add(s.get("key"))
            if o and o.get("key"): next_keys.add(o.get("key"))
            s_id = r.get("s_id"); o_id = r.get("o_id")
            if s_id: next_ids.add(s_id)
            if o_id: next_ids.add(o_id)

        current_ids = next_ids if next_ids else set()
        current_keys = set() if next_ids else next_keys

    return list(triples.values())

def mean_similarity_to_query_triples(triple_emb: Optional[List[float]], q_trip_embs: List[List[float]]) -> float:
    if not isinstance(triple_emb, list) and triple_emb is not None:
        triple_emb = _as_float_list(triple_emb)
    if not isinstance(triple_emb, list) or not q_trip_embs:
        return 0.0
    sims = [cos_sim(triple_emb, q) for q in q_trip_embs]
    if not sims:
        return 0.0
    return sum(sims) / len(sims)

def triple_centric_retrieval(query_triples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    triples_map: Dict[str, Dict[str, Any]] = {}
    q_trip_embs: List[List[float]] = []
    for qt in query_triples:
        try:
            txt = query_triple_to_text(qt)
            emb = embed_text(txt)
            q_trip_embs.append(emb)
        except Exception as ex:
            log(f"[TripleRetrieval] Embedding failed for query triple '{qt}': {ex}", level="WARN")
            continue
        matches = search_similar_triples_by_embedding(emb, k=QUERY_TRIPLE_MATCH_TOP_K_PER)
        for m in matches:
            uid = m.get("triple_uid")
            if uid:
                if uid not in triples_map:
                    triples_map[uid] = m
                else:
                    if m.get("score", 0.0) > triples_map[uid].get("score", 0.0):
                        triples_map[uid] = m
    merged = list(triples_map.values())
    log(f"[TripleRetrieval] Collected {len(merged)} unique KG triples across {len(q_trip_embs)} query triple(s)")
    return merged, q_trip_embs

def entity_centric_retrieval(query_entities: List[Dict[str, Any]], q_trip_embs: List[List[float]], q_emb_fallback: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    all_matched_keys: Set[str] = set()
    all_matched_ids: Set[str] = set()
    for e in query_entities:
        text = (e.get("text") or "").strip()
        if not text:
            continue
        try:
            e_emb = embed_text(text)
        except Exception as ex:
            log(f"[EntityRetrieval] Embedding failed for entity '{text}': {ex}", level="WARN")
            continue
        matches = search_similar_entities_by_embedding(e_emb, k=ENTITY_MATCH_TOP_K)
        keys = [m.get("key") for m in matches if m.get("key")]
        ids  = [m.get("elem_id") for m in matches if m.get("elem_id")]
        all_matched_keys.update(keys)
        all_matched_ids.update(ids)

    if not (all_matched_keys or all_matched_ids):
        log("[EntityRetrieval] No KG entity matches found from query entities.")
        return []

    t0 = now_ms()
    expanded_triples = expand_from_entities(
        list(all_matched_keys),
        hops=ENTITY_SUBGRAPH_HOPS,
        per_hop_limit=ENTITY_SUBGRAPH_PER_HOP_LIMIT,
        entity_elem_ids=list(all_matched_ids) if all_matched_ids else None
    )
    log(f"[EntityRetrieval] Expanded subgraph triples: {len(expanded_triples)} | {dur_ms(t0):.0f} ms")

    if not expanded_triples:
        return []

    def score(t: Dict[str, Any]) -> float:
        emb = t.get("embedding")
        if q_trip_embs:
            return mean_similarity_to_query_triples(emb, q_trip_embs)
        if q_emb_fallback and isinstance(emb, list):
            return cos_sim(q_emb_fallback, emb)
        return 0.0

    ranked = sorted(expanded_triples, key=score, reverse=True)
    top = ranked[:SUBGRAPH_TRIPLES_TOP_K]
    log(f"[EntityRetrieval] Selected top-{len(top)} triples from subgraph")
    return top

# ----------------- GraphRAG chunk collection + rerank (MODIFIED) -----------------
def collect_chunks_for_triples_from_neo4j(
    triples: List[Dict[str, Any]],
    chunk_store_fallback: Optional[ChunkStore] = None
) -> List[Tuple[Tuple[Any, Any], str, Optional[List[float]], Dict[str, Any]]]:
    """
    Collect chunk candidates for triples, but pull content+embedding from Neo4j (:TextChunk).
    Falls back to ChunkStore or evidence_quote when needed.

    Returns list of:
      (key_pair, text, embedding_or_none, triple_dict)
    """
    seen_pairs: Set[Tuple[Any, Any]] = set()
    out: List[Tuple[Tuple[Any, Any], str, Optional[List[float]], Dict[str, Any]]] = []

    for t in triples:
        doc_id = t.get("document_id")
        chunk_id = t.get("chunk_id")

        # If no chunk pointers, use quote fallback
        if doc_id is None or chunk_id is None:
            quote = t.get("evidence_quote")
            if quote:
                key = (t.get("triple_uid"), "quote")
                if key not in seen_pairs:
                    t["_is_quote_fallback"] = True
                    out.append((key, quote, None, t))
                    seen_pairs.add(key)
            continue

        norm_key = (_norm_id(doc_id), _norm_id(chunk_id))
        if norm_key in seen_pairs:
            continue

        row = fetch_textchunk_by_doc_chunk(doc_id, chunk_id)
        if row and isinstance(row.get("content"), str) and row.get("content"):
            emb = row.get("embedding")
            emb_f = _as_float_list(emb) if emb is not None else None
            t["_is_quote_fallback"] = False
            out.append((norm_key, row["content"], emb_f, t))
            seen_pairs.add(norm_key)
            log(f"[TextChunk] HIT Neo4j ({row.get('match')}): doc={_norm_id(doc_id)} chunk={_norm_id(chunk_id)} emb={'yes' if emb_f else 'no'}", level="DEBUG")
            continue

        # Optional fallback to pickle ChunkStore (content only; embedding still missing)
        if chunk_store_fallback is not None:
            text = chunk_store_fallback.get_chunk(doc_id, chunk_id)
            if text:
                t["_is_quote_fallback"] = False
                out.append((norm_key, text, None, t))
                seen_pairs.add(norm_key)
                log(f"[TextChunk] MISS Neo4j but HIT ChunkStore fallback: doc={_norm_id(doc_id)} chunk={_norm_id(chunk_id)}", level="WARN")
                continue

        # Final fallback: use quote
        quote = t.get("evidence_quote")
        if quote:
            key2 = (t.get("triple_uid"), "quote")
            if key2 not in seen_pairs:
                t["_is_quote_fallback"] = True
                log(f"[TextChunk] MISS Neo4j: doc={_norm_id(doc_id)} chunk={_norm_id(chunk_id)}; using quote fallback", level="WARN")
                out.append((key2, quote, None, t))
                seen_pairs.add(key2)

    return out

def rerank_chunks_by_query_using_stored_embeddings(
    chunk_records: List[Tuple[Tuple[Any, Any], str, Optional[List[float]], Dict[str, Any]]],
    q_emb_query: List[float],
    top_k: int,
    cand_limit: Optional[int] = None,
) -> List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]]:
    """
    Score chunks by cosine similarity between query embedding and stored chunk embedding.
    Does NOT embed chunk text at runtime.
    If a chunk has no embedding, it is kept with score 0.0 (so it can still appear if everything is missing).
    """
    limit = cand_limit if isinstance(cand_limit, int) and cand_limit > 0 else CHUNK_RERANK_CAND_LIMIT
    cand = chunk_records[:limit]
    t0 = now_ms()

    scored: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]] = []
    used = 0
    missing = 0
    for key, text, emb, t in cand:
        if isinstance(emb, list) and emb:
            s = cos_sim(q_emb_query, emb)
            scored.append((key, text, t, s))
            used += 1
        else:
            scored.append((key, text, t, 0.0))
            missing += 1

    scored.sort(key=lambda x: x[3], reverse=True)
    took = dur_ms(t0)
    log(f"[ChunkRerank] Scored {len(scored)} candidates using stored embeddings (used={used}, missing={missing}) | picked top {min(top_k, len(scored))} | {took:.0f} ms")
    return scored[:top_k]

def rerank_triples_by_query_triples(triples: List[Dict[str, Any]], q_trip_embs: List[List[float]], q_emb_fallback: Optional[List[float]], top_k: int) -> List[Dict[str, Any]]:
    t0 = now_ms()
    def score(t: Dict[str, Any]) -> float:
        emb = t.get("embedding")
        if q_trip_embs:
            return mean_similarity_to_query_triples(emb, q_trip_embs)
        if q_emb_fallback and isinstance(emb, list):
            return cos_sim(q_emb_fallback, emb)
        return 0.0
    ranked = sorted(triples, key=score, reverse=True)
    took = dur_ms(t0)
    log(f"[TripleRerank] Input={len(triples)} | Output={min(top_k, len(ranked))} | {took:.0f} ms")
    return ranked[:top_k]

def build_combined_context_text(
    triples_ranked: List[Dict[str, Any]],
    chunks_ranked: List[Tuple[Tuple[Any, Any], str, Dict[str, Any], float]]
) -> Tuple[str, str, List[Dict[str, Any]]]:
    summary_lines = []
    summary_lines.append("Ringkasan triple yang relevan:")
    for t in triples_ranked[:min(50, len(triples_ranked))]:
        s = t.get("subject"); p = t.get("predicate"); o = t.get("object")
        uu = t.get("uu_number") or ""
        art = t.get("evidence_article_ref") or ""
        quote = (t.get("evidence_quote") or "")[:300]
        summary_lines.append(f"- {s} [{p}] {o} | {uu} | {art} | bukti: {quote}")
    summary_text = "\n".join(summary_lines)

    lines = [summary_text, "\nPotongan teks terkait (chunk):"]
    for idx, (key, text, t, score) in enumerate(chunks_ranked, start=1):
        doc_id = t.get("document_id")
        chunk_id = t.get("chunk_id")
        uu = t.get("uu_number") or ""
        fb = " | quote-fallback" if t.get("_is_quote_fallback") else ""
        lines.append(f"[Chunk {idx}] doc={doc_id} chunk={chunk_id} | {uu} | score={score:.3f}{fb}\n{text}")
    context = "\n".join(lines)
    chunk_records = [{"key": key, "text": text, "triple": t, "score": score} for key, text, t, score in chunks_ranked]
    return context, summary_text, chunk_records

def agent2_answer(query_original: str, context: str, guidance: Optional[str], output_lang: str = "id") -> str:
    instructions = (
        "Answer concisely and accurately based strictly on the provided context. "
        "Cite UU/Article references when they are clear. "
        "Respond in the same language as the user's question."
    )
    guidance_text = guidance.strip() if isinstance(guidance, str) and guidance.strip() else "(No additional guidance.)"
    prompt = f"""
You are Agent 2 (Answerer). Task: provide an answer based on the context only.

Core instructions:
{instructions}

Additional guidance (if any):
\"\"\"{guidance_text}\"\"\"

Original question:
\"\"\"{query_original}\"\"\"

Context:
\"\"\"{context}\"\"\"
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent 2] Prompt:"); log(prompt)
    log(f"[Agent 2] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")
    answer = safe_generate_text(prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2)
    log(f"[Agent 2] Answer length={len(answer)}")
    return answer

# --- Answer Judge (GraphRAG) ---
AJ_SCHEMA = {
    "type": "object",
    "properties": {
        "acceptable": {"type": "boolean"},
        "problems": {"type": "string"},
        "suggestion": {"type": "string"},
        "notes": {"type": "string"}
    },
    "required": ["acceptable", "problems", "suggestion"]
}

def agent_aj_answer_judge(current_query: str, current_answer: str, prior_qaf_history: List[Dict[str, Any]], user_lang: str) -> Dict[str, Any]:
    history_str = json.dumps(prior_qaf_history, ensure_ascii=False, indent=2) if prior_qaf_history else "[]"
    prompt = f"""
You are Agent AJ (Answer Judge).
Task: Evaluate whether the generated answer sufficiently and accurately addresses the current query, given how the GraphRAG pipeline works.

Pipeline brief:
{PIPELINE_BRIEF}

Guidelines:
- Mark "acceptable": true only if the answer is relevant, sufficiently specific, and plausibly grounded (e.g., cites UU/Article or key text).
- If unacceptable, diagnose concrete issues in "problems" (e.g., vague, missing statute identifiers, wrong scope/jurisdiction, incomplete, lacks references).
- Provide an actionable "suggestion" for modifying the next query so the pipeline can retrieve better context and produce a stronger answer.
- Keep outputs in the same language as the user's query ("{user_lang}").
- Do not include your own knowledge in the evaluation; focus on the answer's quality relative to the query.

Current query:
\"\"\"{current_query}\"\"\"

Generated answer (truncated if long):
\"\"\"{_truncate(current_answer, AJ_ANSWER_MAX_CHARS)}\"\"\"

Prior query–answer–feedback history (latest last):
{history_str}

Return JSON:
- "acceptable": boolean
- "problems": short diagnosis of answer issues (if any)
- "suggestion": concrete improvements to the next query
- "notes": optional
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent AJ] Prompt:"); log(prompt)
    log(f"[Agent AJ] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")
    out = safe_generate_json(prompt, AJ_SCHEMA, temp=0.0)
    acceptable = bool(out.get("acceptable")) if isinstance(out, dict) else False
    problems = (out.get("problems") or "").strip() if isinstance(out, dict) else ""
    suggestion = (out.get("suggestion") or "").strip() if isinstance(out, dict) else ""
    notes = (out.get("notes") or "").strip() if isinstance(out, dict) else ""
    log(f"[Agent AJ] Verdict: acceptable={acceptable} | problems='{problems[:120]}' | suggestion='{suggestion[:120]}'")
    return {"acceptable": acceptable, "problems": problems, "suggestion": suggestion, "notes": notes}

# --- Query Modifier (GraphRAG) ---
QM_SCHEMA = {
    "type": "object",
    "properties": {
        "modified_query": {"type": "string"},
        "rationale": {"type": "string"}
    },
    "required": ["modified_query"]
}

def agent_qm_modify_query(current_query: str, current_answer: str, judge_problems: str, judge_suggestion: str, prior_qaf_history: List[Dict[str, Any]], user_lang: str) -> Dict[str, Any]:
    history_str = json.dumps(prior_qaf_history, ensure_ascii=False, indent=2) if prior_qaf_history else "[]"
    prompt = f"""
You are Agent QM (Query Modifier).
Task: Rewrite the current query to improve retrieval quality and, consequently, the final answer. Use the Answer Judge's feedback and prior history.

Pipeline brief:
{PIPELINE_BRIEF}

Guidelines:
- Preserve the user's original intent and language ("{user_lang}").
- Apply the judge's suggestion(s) concretely to the query text (e.g., add UU number/year, Pasal/Ayat, scope, relation verbs, timeframe, aliases).
- Produce ONE improved query string; do not include explanations in the query text.

Current query:
\"\"\"{current_query}\"\"\"

Current answer (truncated if long):
\"\"\"{_truncate(current_answer, AJ_ANSWER_MAX_CHARS)}\"\"\"

Answer Judge feedback:
- Problems: {judge_problems}
- Suggestion: {judge_suggestion}

Prior query–answer–feedback history (latest last):
{history_str}

Return JSON:
- "modified_query": string (only the improved query, same language and intent)
- "rationale": (optional) short explanation for logs
"""
    est_tokens = estimate_tokens_for_text(prompt)
    log("\n[Agent QM] Prompt:"); log(prompt)
    log(f"[Agent QM] Prompt size: {len(prompt)} chars, est_tokens≈{est_tokens}")
    out = safe_generate_json(prompt, QM_SCHEMA, temp=0.0)
    modified_query = (out.get("modified_query") or "").strip() if isinstance(out, dict) else ""
    rationale = (out.get("rationale") or "").strip() if isinstance(out, dict) else ""
    if not modified_query:
        modified_query = current_query
        rationale = (rationale + " (fallback to current query)").strip()
    log(f"[Agent QM] Modified query: '{modified_query}'")
    return {"modified_query": modified_query, "rationale": rationale}

# ----------------- GraphRAG orchestrator (single-pass retrieval + answer) -----------------
def run_retrieval_for_query_graph(
    query_original: str,
    chunk_store: Optional[ChunkStore],
    user_lang: Optional[str] = None,
    cand_limit_override: Optional[int] = None
) -> Dict[str, Any]:
    user_lang = user_lang or detect_user_language(query_original)

    # Step 0: Embed whole query
    t0 = now_ms()
    q_emb_query = embed_text(query_original)
    t_embed = dur_ms(t0)
    log(f"[G:Retrieval] Step 0 (embed) in {t_embed:.0f} ms")

    # Step 1: Agent 1 – extract entities/predicates
    t1 = now_ms()
    extraction = agent1_extract_entities_predicates(query_original)
    ents = extraction.get("entities", [])
    t_extract = dur_ms(t1)
    log(f"[G:Retrieval] Step 1 (entities/predicates) in {t_extract:.0f} ms")

    # Step 1b: Agent 1b – extract triples from query
    t1b = now_ms()
    query_triples = agent1b_extract_query_triples(query_original)
    t_extract_tr = dur_ms(t1b)
    log(f"[G:Retrieval] Step 1b (query triples) in {t_extract_tr:.0f} ms")

    # Step 2: Triple-centric retrieval
    t2 = now_ms()
    ctx2_triples, q_trip_embs = triple_centric_retrieval(query_triples)
    t_triple = dur_ms(t2)
    log(f"[G:Retrieval] Step 2 (triple-centric) in {t_triple:.0f} ms; triples={len(ctx2_triples)}")

    # Step 3: Entity-centric retrieval
    t3 = now_ms()
    ctx1_triples = entity_centric_retrieval(ents, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query)
    t_entity = dur_ms(t3)
    log(f"[G:Retrieval] Step 3 (entity-centric) in {t_entity:.0f} ms; triples={len(ctx1_triples)}")

    # Step 4: Merge contexts, dedupe triples
    t4 = now_ms()
    triple_map: Dict[str, Dict[str, Any]] = {}
    for t in ctx1_triples + ctx2_triples:
        uid = t.get("triple_uid")
        if uid:
            prev = triple_map.get(uid)
            if prev is None or (t.get("score", 0.0) > prev.get("score", 0.0)):
                triple_map[uid] = t
    merged_triples = list(triple_map.values())
    t_merge = dur_ms(t4)
    log(f"[G:Retrieval] Step 4 (merge triples) in {t_merge:.0f} ms; merged={len(merged_triples)}")

    # Step 5: Gather chunks and rerank (MODIFIED: use stored TextChunk.embedding)
    t5 = now_ms()
    chunk_records = collect_chunks_for_triples_from_neo4j(merged_triples, chunk_store_fallback=chunk_store)
    log(f"[G:Retrieval] Step 5a (collect chunks from Neo4j) candidates={len(chunk_records)}")
    chunks_ranked = rerank_chunks_by_query_using_stored_embeddings(
        chunk_records, q_emb_query, top_k=MAX_CHUNKS_FINAL, cand_limit=cand_limit_override
    )
    t_chunks = dur_ms(t5)
    log(f"[G:Retrieval] Step 5b (rerank chunks) in {t_chunks:.0f} ms; selected={len(chunks_ranked)}")

    # Step 6: Rerank triples
    t6 = now_ms()
    triples_ranked = rerank_triples_by_query_triples(
        merged_triples, q_trip_embs=q_trip_embs, q_emb_fallback=q_emb_query, top_k=MAX_TRIPLES_FINAL
    )
    t_rerank = dur_ms(t6)
    log(f"[G:Retrieval] Step 6 (rerank triples) in {t_rerank:.0f} ms; selected={len(triples_ranked)}")

    # Build combined context
    t_ctx = now_ms()
    context_text, context_summary, _ = build_combined_context_text(triples_ranked, chunks_ranked)
    t_ctx_build = dur_ms(t_ctx)
    log(f"[G:Retrieval] Context built in {t_ctx_build:.0f} ms")

    diagnostics = {
        "timings_ms": {
            "embed": int(t_embed),
            "extract_entities": int(t_extract),
            "extract_triples": int(t_extract_tr),
            "triple_retrieval": int(t_triple),
            "entity_retrieval": int(t_entity),
            "merge_triples": int(t_merge),
            "chunks_rerank": int(t_chunks),
            "triples_rerank": int(t_rerank),
            "context_build": int(t_ctx_build),
        },
        "counts": {
            "ctx2_triples": len(ctx2_triples),
            "ctx1_triples": len(ctx1_triples),
            "merged_triples": len(merged_triples),
            "chunk_candidates": len(chunk_records),
            "chunks_selected": len(chunks_ranked),
            "triples_selected": len(triples_ranked),
        }
    }

    return {
        "context_text": context_text,
        "context_summary": context_summary,
        "diagnostics": diagnostics
    }

def run_single_pass_graph(
    query_original: str,
    chunk_store: Optional[ChunkStore],
    user_lang: Optional[str] = None,
    cand_limit_override: Optional[int] = None,
    guidance: Optional[str] = None
) -> Dict[str, Any]:
    user_lang = user_lang or detect_user_language(query_original)
    r = run_retrieval_for_query_graph(query_original, chunk_store, user_lang, cand_limit_override)
    context_text = r["context_text"]
    context_summary = r["context_summary"]
    t_ans = now_ms()
    intermediate_answer = agent2_answer(query_original, context_text, guidance=guidance, output_lang=user_lang)
    t_ans_ms = dur_ms(t_ans)
    log(f"[G:SinglePass] Answered in {t_ans_ms:.0f} ms")
    return {
        "question": query_original,
        "answer": intermediate_answer,
        "context_summary": context_summary,
        "counts": r["diagnostics"]["counts"],
        "timings_ms": r["diagnostics"]["timings_ms"]
    }

def run_graphrag_loop(query_original: str, chunk_store: Optional[ChunkStore], user_lang: str) -> Dict[str, Any]:
    qaf_history: List[Dict[str, Any]] = []
    per_iteration: List[Dict[str, Any]] = []
    current_query = query_original
    final_answer = ""
    for i in range(1, MAX_ANSWER_JUDGE_ITERS + 1):
        set_log_context("G", f"{i}/{MAX_ANSWER_JUDGE_ITERS}")
        log(f"--- GraphRAG Iteration {i}/{MAX_ANSWER_JUDGE_ITERS} START ---")
        retr = run_retrieval_for_query_graph(current_query, chunk_store, user_lang, cand_limit_override=None)
        context_text = retr["context_text"]; context_summary = retr["context_summary"]; diagnostics = retr["diagnostics"]
        answer = agent2_answer(current_query, context_text, guidance=None, output_lang=user_lang)
        aj = agent_aj_answer_judge(current_query, answer, qaf_history, user_lang)
        acceptable = bool(aj.get("acceptable"))
        problems = (aj.get("problems") or "").strip(); suggestion = (aj.get("suggestion") or "").strip(); notes = (aj.get("notes") or "").strip()
        if acceptable:
            log("[G:Loop] Answer acceptable; stopping.")
            final_answer = answer
            per_iteration.append({
                "query": current_query,
                "answer": answer,
                "judge": {"acceptable": True, "problems": problems, "suggestion": suggestion, "notes": notes},
                "modified_query": None,
                "context_summary": context_summary,
                "retrieval_diagnostics": diagnostics
            })
            break
        if i >= MAX_ANSWER_JUDGE_ITERS:
            log("[G:Loop] Iteration cap reached; returning latest answer despite judge rejection.", level="WARN")
            final_answer = answer
            per_iteration.append({
                "query": current_query,
                "answer": answer,
                "judge": {"acceptable": False, "problems": problems, "suggestion": suggestion, "notes": notes, "stopped_at_cap": True},
                "modified_query": None,
                "context_summary": context_summary,
                "retrieval_diagnostics": diagnostics
            })
            break
        qm = agent_qm_modify_query(current_query, answer, problems, suggestion, qaf_history, user_lang)
        modified_query = (qm.get("modified_query") or current_query).strip()
        rationale = (qm.get("rationale") or "").strip()
        qaf_history.append({
            "query": current_query,
            "answer": answer,
            "feedback": {"problems": problems, "suggestion": suggestion},
            "modified_query": modified_query
        })
        per_iteration.append({
            "query": current_query,
            "answer": answer,
            "judge": {"acceptable": False, "problems": problems, "suggestion": suggestion, "notes": notes},
            "modified_query": modified_query,
            "modifier_rationale": rationale,
            "context_summary": context_summary,
            "retrieval_diagnostics": diagnostics
        })
        if modified_query == current_query:
            log("[G:Loop] Modified query identical to current; continuing due to cap protection.", level="WARN")
        current_query = modified_query
        log(f"--- GraphRAG Iteration {i}/{MAX_ANSWER_JUDGE_ITERS} DONE ---")
    if not final_answer and per_iteration:
        final_answer = per_iteration[-1].get("answer", "") or "(No answer produced)"
    return {
        "final_answer": final_answer,
        "iterations_used": len(per_iteration),
        "per_iteration": per_iteration
    }

# ----------------- NaiveRAG components -----------------
def vector_query_chunks(q_emb: List[float], k: int) -> List[Dict[str, Any]]:
    cypher = """
    WITH $q AS q
    CALL db.index.vector.queryNodes('chunk_embedding_index', $k, q)
    YIELD node, score
    RETURN node AS c, score
    ORDER BY score DESC
    LIMIT $k
    """
    rows = run_cypher_with_retry(cypher, {"q": q_emb, "k": k})
    out = []
    for r in rows:
        c = r["c"]
        out.append({
            "key": c.get("key"),
            "chunk_id": c.get("chunk_id"),
            "document_id": c.get("document_id"),
            "uu_number": c.get("uu_number"),
            "pages": c.get("pages"),
            "content": c.get("content"),
            "score": r["score"],
        })
    return out

# Amendment-aware filtered retrieval (two-stage: filter then rerank)
def convert_to_db_formats(uu_identifier: str) -> List[str]:
    """
    Convert any UU identifier to TextChunk.uu_number database formats:
    - "Undang-undang (UU) Nomor X Tahun Y"
    - "Undang-undang (UU) No. X Tahun Y"
    Accepts: "AMD_16_2025", "16_2025", "UU No. 16 Tahun 2025", etc.
    """
    if not uu_identifier:
        return []
    s = uu_identifier.strip()
    if s.startswith("AMD_"):
        s = s.replace("AMD_", "", 1)
    m = re.match(r"^(\d+)_(\d{4})$", s)
    if m:
        num, year = int(m.group(1)), int(m.group(2))
        return [f"Undang-undang (UU) Nomor {num} Tahun {year}", f"Undang-undang (UU) No. {num} Tahun {year}"]
    pats = [
        r'(?:UU|Undang-undang|Undang-Undang)\s*(?:\(UU\))?\s*(?:Nomor|No\.?)\s*(\d+)\s*Tahun\s*(\d{4})',
        r'(?:UU|Undang-undang|Undang-Undang)\s*(\d+)/(\d{4})',
    ]
    for p in pats:
        mm = re.search(p, uu_identifier, re.IGNORECASE)
        if mm:
            num, year = int(mm.group(1)), int(mm.group(2))
            return [f"Undang-undang (UU) Nomor {num} Tahun {year}", f"Undang-undang (UU) No. {num} Tahun {year}"]
    return []

def vector_query_chunks_filtered(q_emb: List[float], k: int, uu_filters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Two-stage retrieval restricted to specified UUs:
    1) Filter nodes by uu_number IN converted formats
    2) Manually compute cosine similarity score with q_emb
    3) Return top-k by score
    """
    if not uu_filters:
        return vector_query_chunks(q_emb, k)
    db_formats: List[str] = []
    for uf in uu_filters:
        db_formats.extend(convert_to_db_formats(uf))
    db_formats = [x for x in db_formats if x]
    log(f"[Vector Filter] Converted filters -> DB formats: {db_formats}")

    cypher = """
    MATCH (c:TextChunk)
    WHERE c.uu_number IN $uu_filters
    RETURN c AS node, c.embedding AS embedding
    """
    rows = run_cypher_with_retry(cypher, {"uu_filters": db_formats})

    if not rows:
        log("[Vector Filter] No chunks matched provided uu_filters.")
        return []

    scored: List[Tuple[Dict[str, Any], float]] = []
    for r in rows:
        node = r["node"]
        emb = r.get("embedding")
        if not isinstance(emb, list) or not emb:
            continue
        score = cos_sim(q_emb, emb)
        scored.append((node, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]

    out: List[Dict[str, Any]] = []
    for node, score in top:
        out.append({
            "key": node.get("key"),
            "chunk_id": node.get("chunk_id"),
            "document_id": node.get("document_id"),
            "uu_number": node.get("uu_number"),
            "pages": node.get("pages"),
            "content": node.get("content"),
            "score": score,
        })
    log(f"[Vector Filter] Returning {len(out)} filtered chunks")
    return out

def clamp(s: Optional[str], n: int) -> str:
    t = (s or "").strip()
    return t[:n]

def build_context_from_chunks(chunks: List[Dict[str, Any]], max_chunks: int) -> str:
    chosen = chunks[:max_chunks]
    lines = ["Potongan teks terkait (chunk):"]
    for i, c in enumerate(chosen, 1):
        doc = c.get("document_id")
        cid = c.get("chunk_id")
        uu = c.get("uu_number") or ""
        pg = c.get("pages")
        txt = clamp(c.get("content") or "", CHUNK_TEXT_CLAMP)
        lines.append(f"[Chunk {i}] doc={doc} chunk={cid} | {uu} | pages={pg} | score={c.get('score'):.3f}\n{txt}\n")
    return "".join(lines)

ANSWER_JUDGE_SCHEMA_N = {
  "type": "object",
  "properties": {
    "decision": {"type": "string", "enum": ["acceptable", "insufficient"]},
    "reasoning": {"type": "string"},
    "problem": {"type": "string"},
    "suggested_solution": {"type": "string"}
  },
  "required": ["decision"]
}

def answer_judge_naive(current_query: str, current_answer: str, qaf_history: List[Dict[str, Any]], output_lang: str = "en") -> Dict[str, Any]:
    hist_lines = []
    for i, item in enumerate(qaf_history, 1):
        qq = (item.get("query") or "").strip()
        qa = (item.get("answer") or "").strip()
        fb = (item.get("feedback") or {}) or {}
        pr = (fb.get("problem","") or "").strip()
        ss = (fb.get("suggested_solution","") or "").strip()
        hist_lines.append(
            f"[Iter {i}] Query: {qq}\n[Iter {i}] Answer: {qa}\n[Iter {i}] Feedback.problem: {pr}\n[Iter {i}] Feedback.suggested_solution: {ss}"
        )
    history_block = "\n".join(hist_lines) if hist_lines else "(no prior history)"
    instructions = (
        "You are the Answer Judge for a GraphRAG pipeline using naive chunk retrieval.\n"
        "- Judge whether the current answer adequately addresses the current query (relevance, completeness, specificity, legal grounding).\n"
        "- If acceptable: set decision='acceptable'. If insufficient: set decision='insufficient' and provide problem + suggested_solution.\n"
        "- Respond in the same language as the user's question."
        "- Do not include your own knowledge; focus on the answer's quality relative to the query."
    )
    prompt = f"""
You are the Answer Judge.

Current query:
\"\"\"{current_query}\"\"\"

Generated answer:
\"\"\"{current_answer}\"\"\"

Prior query–answer–feedback history:
\"\"\"{history_block}\"\"\"

Instructions:
{instructions}

Return JSON with: decision ('acceptable'|'insufficient'), and if insufficient also problem and suggested_solution.
"""
    est = estimate_tokens_for_text(prompt)
    log("=== [N] Answer Judge ===")
    log("[N:Judge] Prompt:"); log(prompt)
    log(f"[N:Judge] Prompt size: {len(prompt)} chars, est_tokens≈{est}")

    out = safe_generate_json(prompt, ANSWER_JUDGE_SCHEMA_N, temp=0.0) or {}
    decision = (out.get("decision") or "").strip().lower()
    reasoning = (out.get("reasoning") or "").strip()
    problem = (out.get("problem") or "").strip()
    solution = (out.get("suggested_solution") or "").strip()

    if decision not in ("acceptable","insufficient"):
        decision = "insufficient"
    log(f"[N:Judge] Decision: {decision.upper()} | Reasoning: {reasoning}")
    if decision == "insufficient":
        log(f"[N:Judge] Problem: {problem}")
        log(f"[N:Judge] Suggested solution: {solution}")

    return {"decision": decision, "reasoning": reasoning, "problem": problem, "suggested_solution": solution}

QUERY_MODIFIER_SCHEMA_N = {
  "type": "object",
  "properties": {
    "modified_query": {"type": "string"},
    "notes": {"type": "string"}
  },
  "required": ["modified_query"]
}

def query_modifier_naive(current_query: str, current_answer: str, judge_feedback: Dict[str, Any], qaf_history: List[Dict[str, Any]], output_lang: str = "en") -> Dict[str, Any]:
    hist_lines = []
    for i, item in enumerate(qaf_history, 1):
        qq = (item.get("query") or "").strip()
        qa = (item.get("answer") or "").strip()
        fb = (item.get("feedback") or {}) or {}
        pr = (fb.get("problem","") or "").strip()
        ss = (fb.get("suggested_solution","") or "").strip()
        hist_lines.append(
            f"[Iter {i}] Query: {qq}\n[Iter {i}] Answer: {qa}\n[Iter {i}] Feedback.problem: {pr}\n[Iter {i}] Feedback.suggested_solution: {ss}"
        )
    history_block = "\n".join(hist_lines) if hist_lines else "(no prior history)"

    problem = (judge_feedback.get("problem") or "").strip()
    solution = (judge_feedback.get("suggested_solution") or "").strip()

    instructions = (
        "You are the Query Modifier for a naive GraphRAG pipeline.\n"
        "- Use the judge's diagnosis and suggestion to rewrite the query so retrieval improves.\n"
        "- Keep the user's intent and language.\n"
        "- Prefer adding exact identifiers, synonyms, constraints (jurisdiction/timeframe)."
    )

    prompt = f"""
You are the Query Modifier.

Current query:
\"\"\"{current_query}\"\"\"

Current answer:
\"\"\"{current_answer}\"\"\"

Answer judge feedback:
- problem: {problem}
- suggested_solution: {solution}

Prior query–answer–feedback history:
\"\"\"{history_block}\"\"\"

Instructions:
{instructions}

Return JSON with:
- modified_query: improved query (same language)
- notes: brief rationale (optional)
"""
    est = estimate_tokens_for_text(prompt)
    log("=== [N] Query Modifier ===")
    log("[N:Modifier] Prompt:"); log(prompt)
    log(f"[N:Modifier] Prompt size: {len(prompt)} chars, est_tokens≈{est}")

    out = safe_generate_json(prompt, QUERY_MODIFIER_SCHEMA_N, temp=0.0) or {}
    modified_query = (out.get("modified_query") or "").strip()
    notes = (out.get("notes") or "").strip()
    if not modified_query:
        modified_query = current_query
        notes = notes or "(No modification; using current query.)"
    log(f"[N:Modifier] Modified query: {modified_query}")
    if notes:
        log(f"[N:Modifier] Notes: {notes}")
    return {"modified_query": modified_query, "notes": notes}

def run_naiverag_loop(query_original: str, user_lang: str) -> Dict[str, Any]:
    qaf_history: List[Dict[str, Any]] = []
    iteration_runs: List[Dict[str, Any]] = []
    final_answer = ""
    current_query = query_original.strip()
    for it in range(1, MAX_ANSWER_JUDGE_ITERS + 1):
        set_log_context("N", f"{it}/{MAX_ANSWER_JUDGE_ITERS}")
        log(f"--- NaiveRAG Iteration {it}/{MAX_ANSWER_JUDGE_ITERS} ---")
        t_e = time.time()
        q_emb = embed_text(current_query)
        log(f"[N:Iter {it}] Embedded query in {(time.time()-t_e)*1000:.0f} ms")

        t_v = time.time()
        candidates = vector_query_chunks(q_emb, k=TOP_K_CHUNKS)
        log(f"[N:Iter {it}] Vector search returned {len(candidates)} candidates in {(time.time()-t_v)*1000:.0f} ms")

        if not candidates:
            context_text = "Tidak ada potongan teks yang ditemukan." if user_lang == "id" else "No relevant chunks found."
        else:
            context_text = build_context_from_chunks(candidates, max_chunks=NAIVE_MAX_CHUNKS_FINAL)
            preview = "\n".join(context_text.splitlines()[:20])
            log(f"[N:Iter {it}] Context preview:\n{preview}")

        t_ans = time.time()
        answer = agent2_answer(current_query, context_text, guidance=None, output_lang=user_lang)
        log(f"[N:Iter {it}] Answer generated in {(time.time()-t_ans)*1000:.0f} ms")

        if it >= MAX_ANSWER_JUDGE_ITERS:
            log(f"[N:Iter {it}] Iteration limit reached. Finalizing current answer.")
            final_answer = answer
            meta_top = []
            for c in (candidates[:5] if candidates else []):
                meta_top.append({
                    "document_id": c.get("document_id"),
                    "chunk_id": c.get("chunk_id"),
                    "uu_number": c.get("uu_number"),
                    "pages": c.get("pages"),
                    "score": c.get("score"),
                })
            iteration_runs.append({
                "iteration": it,
                "query": current_query,
                "answer": answer,
                "judge": {"decision": "skipped_limit"},
                "modified_query": None,
                "retrieval_meta": {
                    "num_candidates": len(candidates) if candidates else 0,
                    "top_chunks": meta_top
                }
            })
            break

        judge = answer_judge_naive(current_query, answer, qaf_history, output_lang=user_lang)
        decision = judge.get("decision","insufficient")

        if decision == "acceptable":
            log(f"[N:Iter {it}] Judge deemed answer ACCEPTABLE. Finalizing.")
            final_answer = answer
            meta_top = []
            for c in (candidates[:5] if candidates else []):
                meta_top.append({
                    "document_id": c.get("document_id"),
                    "chunk_id": c.get("chunk_id"),
                    "uu_number": c.get("uu_number"),
                    "pages": c.get("pages"),
                    "score": c.get("score"),
                })
            iteration_runs.append({
                "iteration": it,
                "query": current_query,
                "answer": answer,
                "judge": judge,
                "modified_query": None,
                "retrieval_meta": {
                    "num_candidates": len(candidates) if candidates else 0,
                    "top_chunks": meta_top
                }
            })
            break
        else:
            fb = {"problem": judge.get("problem",""), "suggested_solution": judge.get("suggested_solution","")}
            qaf_history.append({"iteration": it, "query": current_query, "answer": answer, "feedback": fb})
            log(f"[N:Iter {it}] Judge indicates insufficiency; invoking Query Modifier.")
            mod = query_modifier_naive(current_query, answer, fb, qaf_history, output_lang=user_lang)
            new_query = mod.get("modified_query", "").strip() or current_query
            notes = mod.get("notes","")
            log(f"[N:Iter {it}] Modified query -> {new_query}")
            if notes:
                log(f"[N:Iter {it}] Modifier notes: {notes}")
            iteration_runs.append({
                "iteration": it,
                "query": current_query,
                "answer": answer,
                "judge": judge,
                "modified_query": new_query,
                "retrieval_meta": {
                    "num_candidates": len(candidates) if candidates else 0
                }
            })
            current_query = new_query
            continue

    if not final_answer and iteration_runs:
        final_answer = iteration_runs[-1].get("answer", "") or "(No answer produced)"
    return {
        "final_answer": final_answer,
        "iterations_used": len(iteration_runs),
        "iteration_runs": iteration_runs
    }

# ----------------- Aggregator Agent -----------------
AGG_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["choose_graphrag", "choose_naiverag", "merge"]},
        "final_answer": {"type": "string"},
        "rationale": {"type": "string"}
    },
    "required": ["decision", "final_answer"]
}

def aggregator_agent(query: str, graphrag_answer: str, naiverag_answer: str, graphrag_meta: Dict[str, Any], naiverag_meta: Dict[str, Any], user_lang: str) -> Dict[str, Any]:
    g_meta = json.dumps(graphrag_meta, ensure_ascii=False, indent=2) if graphrag_meta else "{}"
    n_meta = json.dumps(naiverag_meta, ensure_ascii=False, indent=2) if naiverag_meta else "{}"
    prompt = f"""
You are the Aggregator Agent. Your task: produce the best possible final answer to the user's question.

Inputs:
- Question (language='{user_lang}'):
\"\"\"{query}\"\"\"

- GraphRAG answer:
\"\"\"{_truncate(graphrag_answer, 12000)}\"\"\"

- NaiveRAG answer:
\"\"\"{_truncate(naiverag_answer, 12000)}\"\"\"

- GraphRAG meta (for context only; do not invent facts beyond the answers):
{g_meta}

- NaiveRAG meta (for context only; do not invent facts beyond the answers):
{n_meta}

Guidelines:
- Choose the better answer OR synthesize a concise, coherent answer that combines non-conflicting strengths of both.
- Prefer specificity and legal grounding (e.g., citing UU number / Pasal / Article) if present in the answers.
- Do not add information that does not appear in either answer.
- Keep the final answer in the same language as the user's question.
- If merging, ensure consistency and avoid contradictions.

Return JSON:
- decision: 'choose_graphrag' | 'choose_naiverag' | 'merge'
- final_answer: the chosen or synthesized answer
- rationale: short explanation (1-2 sentences)
"""
    est = estimate_tokens_for_text(prompt)
    log("\n[Aggregator] Prompt:"); log(prompt)
    log(f"[Aggregator] Prompt size: {len(prompt)} chars, est_tokens≈{est}")
    out = safe_generate_json(prompt, AGG_SCHEMA, temp=AGG_TEMPERATURE) or {}
    decision = (out.get("decision") or "").strip()
    final_answer = (out.get("final_answer") or "").strip()
    rationale = (out.get("rationale") or "").strip()
    if decision not in ("choose_graphrag","choose_naiverag","merge") or not final_answer:
        def score(a: str) -> int:
            t = a.lower()
            cues = 0
            cues += len(re.findall(r"\bpasal\b", t))
            cues += len(re.findall(r"\buu\b", t))
            cues += len(re.findall(r"\barticle\b", t))
            cues += len(re.findall(r"\bsection\b", t))
            cues += len(re.findall(r"\bperaturan\b", t))
            cues += len(re.findall(r"\b\d{4}\b", t))
            return cues
        sg = score(graphrag_answer); sn = score(naiverag_answer)
        if sn > sg and naiverag_answer.strip():
            decision = "choose_naiverag"; final_answer = naiverag_answer.strip(); rationale = rationale or "Fallback: naive answer appears more specific."
        else:
            decision = "choose_graphrag"; final_answer = graphrag_answer.strip() or naiverag_answer.strip(); rationale = rationale or "Fallback: GraphRAG chosen by default or better specificity."
        log("[Aggregator] Fallback decision used due to invalid or empty aggregator output.", level="WARN")
    log(f"[Aggregator] Decision={decision} | Rationale: {rationale[:160]}")
    return {"decision": decision, "final_answer": final_answer, "rationale": rationale}

# ----------------- Amendment-aware utilities (shared) -----------------
def normalize_uu_identifier(number: int, year: int) -> str:
    return f"AMD_{number}_{year}"

def get_outgoing_amendments(uu_key: str) -> List[Dict[str, Any]]:
    cypher = """
    MATCH (source:AMD_UndangUndang {key: $uu_key})-[r]->(target:AMD_UndangUndang)
    WHERE type(r) IN ['AMD_DIUBAH_DENGAN', 'AMD_DIUBAH_SEBAGIAN_DENGAN',
                      'AMD_DICABUT_DENGAN', 'AMD_DICABUT_SEBAGIAN_DENGAN']
    RETURN target.key AS target_key,
           target.number AS target_number,
           target.year AS target_year,
           type(r) AS relationship_type
    """
    rows = run_cypher_with_retry(cypher, {"uu_key": uu_key})
    out = []
    for row in rows:
        out.append({
            "target_key": row["target_key"],
            "target_number": row["target_number"],
            "target_year": row["target_year"],
            "relationship_type": row["relationship_type"]
        })
    return out

def traverse_amendment_chain_with_reset(start_number: int, start_year: int) -> Dict[str, Any]:
    start_key = normalize_uu_identifier(start_number, start_year)
    check_cypher = "MATCH (u:AMD_UndangUndang {key: $key}) RETURN u"
    exists = run_cypher_with_retry(check_cypher, {"key": start_key})
    if not exists:
        return {
            "original_uu": start_key,
            "relevant_uus": [start_key],
            "amendment_info": [],
            "has_amendments": False
        }
    relevant_uus = [start_key]
    amendment_info = []
    visited = set()
    queue = [start_key]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        outgoing = get_outgoing_amendments(current)
        for rel in outgoing:
            target = rel["target_key"]; rel_type = rel["relationship_type"]
            amendment_info.append({
                "from": current, "to": target, "type": rel_type,
                "target_number": rel["target_number"], "target_year": rel["target_year"]
            })
            if rel_type == "AMD_DICABUT_DENGAN":
                relevant_uus = [target]
            else:
                if target not in relevant_uus:
                    relevant_uus.append(target)
            queue.append(target)
    has_amendments = len(relevant_uus) > 1 or len(amendment_info) > 0
    return {
        "original_uu": start_key,
        "relevant_uus": relevant_uus,
        "amendment_info": amendment_info,
        "has_amendments": has_amendments
    }

AMENDMENT_TYPE_NAMES = {
    "AMD_DIUBAH_DENGAN": "diubah sepenuhnya dengan",
    "AMD_DIUBAH_SEBAGIAN_DENGAN": "diubah sebagian dengan",
    "AMD_DICABUT_DENGAN": "dicabut dengan",
    "AMD_DICABUT_SEBAGIAN_DENGAN": "dicabut sebagian dengan"
}

def format_uu_display(uu_key: str) -> str:
    parts = uu_key.replace("AMD_", "").split("_")
    return f"UU No. {parts[0]} Tahun {parts[1]}" if len(parts) == 2 else uu_key

def generate_currency_warning(chain_results: List[Dict[str, Any]]) -> str:
    warnings = []
    for chain in chain_results:
        if not chain.get("has_amendments"):
            continue
        original = format_uu_display(chain["original_uu"])
        amendments_by_type: Dict[str, List[str]] = {}
        for amd in chain["amendment_info"]:
            rel_type = amd["type"]
            target_display = f"UU No. {amd['target_number']} Tahun {amd['target_year']}"
            amendments_by_type.setdefault(rel_type, []).append(target_display)
        descs = []
        for rel_type, targets in amendments_by_type.items():
            type_name = AMENDMENT_TYPE_NAMES.get(rel_type, rel_type)
            descs.append(f"{type_name} {', '.join(targets)}")
        warnings.append(f"⚠️ {original} telah {'; '.join(descs)}. Memeriksa ketentuan terbaru...")
    return "\n".join(warnings)

UU_REFERENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "year": {"type": "integer"},
                    "context": {"type": "string"}
                },
                "required": ["number", "year"]
            }
        }
    },
    "required": ["references"]
}

def extract_uu_references_llm(answer_text: str) -> List[Dict[str, Any]]:
    prompt = f"""
Extract all Indonesian law (Undang-undang/UU) references from the following text.

For each reference, identify:
- number: UU number (integer)
- year: year (integer)
- context: brief context (optional)

Common formats include:
- "UU No. 16 Tahun 2025"
- "Undang-undang Nomor 16 Tahun 2025"
- "UU 16/2025"
- "Undang-Undang (UU) No. 16 Tahun 2025"

Return JSON with "references" array containing objects with number, year, and optional context.

Text:
\"\"\"
{answer_text}
\"\"\""""
    log("[Amendment] Extracting UU references from answer...")
    result = safe_generate_json(prompt, UU_REFERENCE_SCHEMA, temp=0.0) or {}
    refs = result.get("references", []) if isinstance(result, dict) else []
    log(f"[Amendment] Found {len(refs)} UU references: {[(r.get('number'), r.get('year')) for r in refs]}")
    return refs

RELEVANCE_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_relevant": {"type": "boolean"},
        "affected_aspects": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"}
    },
    "required": ["is_relevant", "affected_aspects", "reasoning"]
}

def judge_amendment_relevance(query_original: str, initial_answer: str, new_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    new_context = build_context_from_chunks(new_chunks, max_chunks=MAX_CHUNKS_FINAL)
    prompt = f"""
You are a legal relevance judge. Determine if the new legal provisions from amending documents
contain information that addresses or modifies the answer to the user's original question.

Original user question:
\"\"\"{query_original}\"\"\"

Initial answer (based on older law):
\"\"\"{initial_answer}\"\"\"

New provisions from amending documents:
\"\"\"{new_context}\"\"\"

Task:
1) Determine if the new provisions are relevant to the original question.
2) If relevant, identify which aspects of the initial answer are affected.
3) Provide brief reasoning.

Return JSON with:
- is_relevant: boolean
- affected_aspects: array of strings
- reasoning: string
"""
    log("[Amendment] Judging relevance of amending provisions...")
    result = safe_generate_json(prompt, RELEVANCE_JUDGE_SCHEMA, temp=0.0) or {}
    return {
        "is_relevant": bool(result.get("is_relevant")),
        "affected_aspects": result.get("affected_aspects", []) or [],
        "reasoning": result.get("reasoning", "") or ""
    }

def integrate_amendments(
    query_original: str,
    initial_answer: str,
    new_context: str,
    amendment_info: List[Dict[str, Any]],
    relevance_result: Dict[str, Any],
    output_lang: str = "id"
) -> str:
    amendment_descriptions = []
    for amd in amendment_info:
        from_uu = format_uu_display(amd["from"])
        to_uu = format_uu_display(amd["to"])
        rel_type_display = AMENDMENT_TYPE_NAMES.get(amd["type"], amd["type"])
        amendment_descriptions.append(f"- {from_uu} {rel_type_display} {to_uu}")
    amendments_text = "\n".join(amendment_descriptions) if amendment_descriptions else "Tidak ada amandemen"
    affected_aspects_text = ", ".join(relevance_result.get("affected_aspects", [])) if relevance_result.get("affected_aspects") else "Semua aspek"
    prompt = f"""
You are a legal amendment integration agent. Update the initial answer with new information from amending documents.

Original user question:
\"\"\"{query_original}\"\"\"

Initial answer (from older law):
\"\"\"{initial_answer}\"\"\"

Amendment information:
{amendments_text}

Affected aspects identified:
{affected_aspects_text}

New provisions from amending documents:
\"\"\"{new_context}\"\"\"

Task:
1) Integrate the new information into the initial answer.
2) Clearly indicate which parts come from which document.
3) Specify which provisions are currently operative.
4) If only some aspects are affected, indicate which parts remain valid.
5) Use clear labeling, e.g. "[Aspek X (UU A/B)]: ... [DIPERBARUI oleh UU C/D]: ..."

Respond in {output_lang}.
"""
    log("[Amendment] Integrating amendments into final answer...")
    return safe_generate_text(prompt, max_tokens=ANSWER_MAX_TOKENS, temperature=0.2)

_AMENDMENT_CHAIN_CACHE: Dict[str, Dict[str, Any]] = {}
_AMENDMENT_CHAIN_LOCK = Lock()

def traverse_amendment_chain_with_reset_cached(number: int, year: int) -> Dict[str, Any]:
    key = f"{number}_{year}"
    with _AMENDMENT_CHAIN_LOCK:
        if key in _AMENDMENT_CHAIN_CACHE:
            return _AMENDMENT_CHAIN_CACHE[key]
    res = traverse_amendment_chain_with_reset(number, year)
    with _AMENDMENT_CHAIN_LOCK:
        _AMENDMENT_CHAIN_CACHE[key] = res
    return res

def apply_amendments_pre_aggregation(query_original: str, initial_answer: str, user_lang: str) -> Dict[str, Any]:
    refs = extract_uu_references_llm(initial_answer)
    if not refs:
        log("[Amendment] No UU references detected in answer. Skipping amendment handling.")
        return {
            "final_answer": initial_answer,
            "has_amendments": False,
            "currency_warnings": "",
            "amendment_info": [],
            "amending_chunks_used": 0
        }

    chain_results = []
    amending_set: Set[str] = set()
    all_amendment_info: List[Dict[str, Any]] = []
    for r in refs:
        num = r.get("number"); yr = r.get("year")
        if num is None or yr is None:
            continue
        chain = traverse_amendment_chain_with_reset_cached(int(num), int(yr))
        chain_results.append(chain)
        if chain.get("has_amendments"):
            for uu in chain.get("relevant_uus", []):
                if uu != chain.get("original_uu"):
                    amending_set.add(uu)
            all_amendment_info.extend(chain.get("amendment_info", []))

    currency_warning = generate_currency_warning(chain_results) if chain_results else ""
    if not amending_set:
        log("[Amendment] No amending UUs found. Returning original answer (with warning if any).")
        final = (currency_warning + "\n" if currency_warning else "") + initial_answer
        return {
            "final_answer": final,
            "has_amendments": False,
            "currency_warnings": currency_warning,
            "amendment_info": [],
            "amending_chunks_used": 0
        }

    # Query embedding (needed for vector retrieval); chunks are retrieved using stored embeddings
    q_emb = embed_text(query_original)
    amending_chunks = vector_query_chunks_filtered(q_emb, k=TOP_K_CHUNKS, uu_filters=list(amending_set))

    if not amending_chunks:
        log("[Amendment] No chunks from amending documents were retrieved. Appending warning note.")
        final = initial_answer
        if currency_warning:
            final = currency_warning + "\n" + final
        final += "\nCatatan: UU yang dirujuk mengalami amandemen, namun tidak ditemukan potongan relevan yang mengubah jawaban di atas."
        return {
            "final_answer": final,
            "has_amendments": True,
            "currency_warnings": currency_warning,
            "amendment_info": all_amendment_info,
            "amending_chunks_used": 0
        }

    rel = judge_amendment_relevance(query_original, initial_answer, amending_chunks)
    if not rel.get("is_relevant"):
        final = initial_answer
        if currency_warning:
            final = currency_warning + "\n" + final
        final += "\nCatatan: Amandemen tidak mempengaruhi aspek yang ditanyakan; jawaban tetap berlaku."
        return {
            "final_answer": final,
            "has_amendments": True,
            "currency_warnings": currency_warning,
            "amendment_info": all_amendment_info,
            "amending_chunks_used": 0
        }

    new_context = build_context_from_chunks(amending_chunks, max_chunks=MAX_CHUNKS_FINAL)
    integrated = integrate_amendments(query_original, initial_answer, new_context, all_amendment_info, rel, output_lang=user_lang)
    final_answer = (currency_warning + "\n" if currency_warning else "") + integrated
    return {
        "final_answer": final_answer,
        "has_amendments": True,
        "currency_warnings": currency_warning,
        "amendment_info": all_amendment_info,
        "amending_chunks_used": len(amending_chunks)
    }

# ----------------- Top-level Multi-Agent orchestrator (amendment-aware, pre-aggregation) -----------------
def run_multi_agent(query_original: str) -> Dict[str, Any]:
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.multi-agent.txt"
    _LOGGER = FileLogger(log_file, also_console=True)
    set_log_context(None, None)
    t_all = now_ms()
    try:
        log("=== Multi-Agent RAG run started ===")
        log(f"Process info: pid={_pid()}")
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log(f"Parameters: MAX_ANSWER_JUDGE_ITERS={MAX_ANSWER_JUDGE_ITERS}, MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}, TOP_K_CHUNKS={TOP_K_CHUNKS}")
        log(f"LLM limits: EMBED(max_conc={LLM_EMBED_MAX_CONCURRENCY}, qps={LLM_EMBED_QPS}), GEN(max_conc={LLM_GEN_MAX_CONCURRENCY}, qps={LLM_GEN_QPS})")
        log(f"Neo4j: attempts={NEO4J_MAX_ATTEMPTS}, tx_timeout_s={NEO4J_TX_TIMEOUT_S:.1f}, max_concurrency={NEO4J_MAX_CONCURRENCY or 'unlimited'}")
        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")

        # ChunkStore now optional; kept for fallback only
        chunk_store = ChunkStore(LANGCHAIN_DIR, set(SKIP_FILES))

        G_res: Dict[str, Any] = {}
        N_res: Dict[str, Any] = {}
        exc: Dict[str, str] = {}

        def run_G():
            try:
                set_log_context("G", None)
                base_out = run_graphrag_loop(query_original, chunk_store, user_lang)
                base_answer = (base_out.get("final_answer") or "").strip()
                amend = apply_amendments_pre_aggregation(query_original, base_answer, user_lang)
                G_res.update(base_out)
                G_res["final_answer"] = amend["final_answer"]
                G_res["amendment"] = {
                    "has_amendments": amend["has_amendments"],
                    "currency_warnings": amend["currency_warnings"],
                    "amendment_info": amend["amendment_info"],
                    "amending_chunks_used": amend["amending_chunks_used"]
                }
            except Exception as e:
                exc["G"] = str(e)
                log(f"[G] Error: {e}", level="ERROR")

        def run_N():
            try:
                set_log_context("N", None)
                base_out = run_naiverag_loop(query_original, user_lang)
                base_answer = (base_out.get("final_answer") or "").strip()
                amend = apply_amendments_pre_aggregation(query_original, base_answer, user_lang)
                N_res.update(base_out)
                N_res["final_answer"] = amend["final_answer"]
                N_res["amendment"] = {
                    "has_amendments": amend["has_amendments"],
                    "currency_warnings": amend["currency_warnings"],
                    "amendment_info": amend["amendment_info"],
                    "amending_chunks_used": amend["amending_chunks_used"]
                }
            except Exception as e:
                exc["N"] = str(e)
                log(f"[N] Error: {e}", level="ERROR")

        t0 = now_ms()
        tG = Thread(target=run_G, name="GraphRAGThread", daemon=True)
        tN = Thread(target=run_N, name="NaiveRAGThread", daemon=True)
        tG.start(); tN.start()
        tG.join(); tN.join()
        log(f"[MultiAgent] Pipelines completed in {dur_ms(t0):.0f} ms")

        g_answer = (G_res.get("final_answer") or "").strip()
        n_answer = (N_res.get("final_answer") or "").strip()

        if "G" in exc and not g_answer and n_answer:
            final = {"decision": "choose_naiverag", "final_answer": n_answer, "rationale": "GraphRAG failed. Using NaiveRAG answer."}
        elif "N" in exc and not n_answer and g_answer:
            final = {"decision": "choose_graphrag", "final_answer": g_answer, "rationale": "NaiveRAG failed. Using GraphRAG answer."}
        else:
            g_meta = {
                "iterations_used": G_res.get("iterations_used"),
                "context_hint": "Graph triples + chunks",
                "per_iteration_count": len(G_res.get("per_iteration", []) or []),
                "amendment": G_res.get("amendment", {})
            }
            n_meta = {
                "iterations_used": N_res.get("iterations_used"),
                "context_hint": "Top chunk content",
                "iteration_runs_count": len(N_res.get("iteration_runs", []) or []),
                "amendment": N_res.get("amendment", {})
            }
            final = aggregator_agent(query_original, g_answer, n_answer, g_meta, n_meta, user_lang)

        total_ms = dur_ms(t_all)
        set_log_context(None, None)
        log("\n=== Multi-Agent RAG summary ===")
        log(f"- GraphRAG iters: {G_res.get('iterations_used')}")
        log(f"- NaiveRAG iters: {N_res.get('iterations_used')}")
        log(f"- Aggregator decision: {final.get('decision')}")
        log(f"- Total runtime: {total_ms:.0f} ms")
        log("\n=== Final Answer (Aggregated) ===")
        log(final.get("final_answer",""))
        log(f"\nLogs saved to: {log_file}")

        return {
            "final_answer": final.get("final_answer",""),
            "aggregator": {
                "decision": final.get("decision",""),
                "rationale": final.get("rationale","")
            },
            "graphrag": G_res,
            "naiverag": N_res,
            "log_file": str(log_file)
        }
    finally:
        if _LOGGER is not None:
            _LOGGER.close()
            _LOGGER = None

def run_multi_agent_after_aggregation(query_original: str) -> Dict[str, Any]:
    """
    Option B: Same as run_multi_agent, but the amendment-aware pipeline runs
    AFTER aggregation (once on the aggregated final answer), rather than per-pipeline before aggregation.
    """
    global _LOGGER
    ts_name = make_timestamp_name()
    log_file = Path.cwd() / f"{ts_name}.multi-agent.txt"
    _LOGGER = FileLogger(log_file, also_console=True)
    set_log_context(None, None)
    t_all = now_ms()
    try:
        log("=== Multi-Agent RAG run started ===")
        log(f"Process info: pid={_pid()}")
        log(f"Log file: {log_file}")
        log(f"Original Query: {query_original}")
        log(f"Parameters: MAX_ANSWER_JUDGE_ITERS={MAX_ANSWER_JUDGE_ITERS}, MAX_CHUNKS_FINAL={MAX_CHUNKS_FINAL}, TOP_K_CHUNKS={TOP_K_CHUNKS}")
        log(f"LLM limits: EMBED(max_conc={LLM_EMBED_MAX_CONCURRENCY}, qps={LLM_EMBED_QPS}), GEN(max_conc={LLM_GEN_MAX_CONCURRENCY}, qps={LLM_GEN_QPS})")
        log(f"Neo4j: attempts={NEO4J_MAX_ATTEMPTS}, tx_timeout_s={NEO4J_TX_TIMEOUT_S:.1f}, max_concurrency={NEO4J_MAX_CONCURRENCY or 'unlimited'}")
        user_lang = detect_user_language(query_original)
        log(f"[Language] Detected user language: {user_lang}")

        # ChunkStore now optional; kept for fallback only
        chunk_store = ChunkStore(LANGCHAIN_DIR, set(SKIP_FILES))

        G_res: Dict[str, Any] = {}
        N_res: Dict[str, Any] = {}
        exc: Dict[str, str] = {}

        def run_G():
            try:
                set_log_context("G", None)
                base_out = run_graphrag_loop(query_original, chunk_store, user_lang)
                base_answer = (base_out.get("final_answer") or "").strip()
                G_res.update(base_out)
                G_res["final_answer"] = base_answer
                G_res["amendment"] = {"has_amendments": False, "currency_warnings": "", "amendment_info": [], "amending_chunks_used": 0}
            except Exception as e:
                exc["G"] = str(e)
                log(f"[G] Error: {e}", level="ERROR")

        def run_N():
            try:
                set_log_context("N", None)
                base_out = run_naiverag_loop(query_original, user_lang)
                base_answer = (base_out.get("final_answer") or "").strip()
                N_res.update(base_out)
                N_res["final_answer"] = base_answer
                N_res["amendment"] = {"has_amendments": False, "currency_warnings": "", "amendment_info": [], "amending_chunks_used": 0}
            except Exception as e:
                exc["N"] = str(e)
                log(f"[N] Error: {e}", level="ERROR")

        t0 = now_ms()
        tG = Thread(target=run_G, name="GraphRAGThread", daemon=True)
        tN = Thread(target=run_N, name="NaiveRAGThread", daemon=True)
        tG.start(); tN.start()
        tG.join(); tN.join()
        log(f"[MultiAgent] Pipelines completed in {dur_ms(t0):.0f} ms")

        g_answer = (G_res.get("final_answer") or "").strip()
        n_answer = (N_res.get("final_answer") or "").strip()

        if "G" in exc and not g_answer and n_answer:
            final = {"decision": "choose_naiverag", "final_answer": n_answer, "rationale": "GraphRAG failed. Using NaiveRAG answer."}
        elif "N" in exc and not n_answer and g_answer:
            final = {"decision": "choose_graphrag", "final_answer": g_answer, "rationale": "NaiveRAG failed. Using GraphRAG answer."}
        else:
            g_meta = {
                "iterations_used": G_res.get("iterations_used"),
                "context_hint": "Graph triples + chunks",
                "per_iteration_count": len(G_res.get("per_iteration", []) or []),
                "amendment": G_res.get("amendment", {})
            }
            n_meta = {
                "iterations_used": N_res.get("iterations_used"),
                "context_hint": "Top chunk content",
                "iteration_runs_count": len(N_res.get("iteration_runs", []) or []),
                "amendment": N_res.get("amendment", {})
            }
            final = aggregator_agent(query_original, g_answer, n_answer, g_meta, n_meta, user_lang)

        amended = apply_amendments_pre_aggregation(query_original, final.get("final_answer", ""), user_lang)
        final["final_answer"] = amended["final_answer"]

        total_ms = dur_ms(t_all)
        set_log_context(None, None)
        log("\n=== Multi-Agent RAG summary ===")
        log(f"- GraphRAG iters: {G_res.get('iterations_used')}")
        log(f"- NaiveRAG iters: {N_res.get('iterations_used')}")
        log(f"- Aggregator decision: {final.get('decision')}")
        log(f"- Total runtime: {total_ms:.0f} ms")
        log("\n=== Final Answer (Aggregated) ===")
        log(final.get("final_answer",""))
        log(f"\nLogs saved to: {log_file}")

        return {
            "final_answer": final.get("final_answer",""),
            "aggregator": {
                "decision": final.get("decision",""),
                "rationale": final.get("rationale","")
            },
            "graphrag": G_res,
            "naiverag": N_res,
            "log_file": str(log_file)
        }
    finally:
        if _LOGGER is not None:
            _LOGGER.close()
            _LOGGER = None

# ----------------- Main -----------------
if __name__ == "__main__":
    try:
        user_query = input("Enter your query: ").strip()
        if not user_query:
            print("Empty query. Exiting.")
        else:
            res = run_multi_agent_after_aggregation(user_query)
            print("\n===== Aggregated Final Answer =====\n")
            print(res.get("final_answer",""))
    finally:
        try:
            driver.close()
        except Exception:
            pass