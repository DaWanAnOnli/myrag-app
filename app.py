#!/usr/bin/env python3
"""
Flask web application for Amendment-aware Agentic Legal Question Answering.
Wraps the multi-agent.py script and provides real-time process visualization.
"""

import os
import sys
import json
import time
import threading
import queue
from pathlib import Path
from typing import Any, Dict, List, Optional
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import importlib.util

# Import the multi-agent module dynamically
_here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("multi_agent", _here / "multi-agent.py")
multi_agent = importlib.util.module_from_spec(spec)
sys.modules["multi_agent"] = multi_agent
spec.loader.exec_module(multi_agent)

app = Flask(__name__)
CORS(app)

# Global event queue for streaming updates
_event_queues: Dict[str, queue.Queue] = {}
_event_lock = threading.Lock()

def get_or_create_queue(session_id: str) -> queue.Queue:
    with _event_lock:
        if session_id not in _event_queues:
            _event_queues[session_id] = queue.Queue()
        return _event_queues[session_id]

def remove_queue(session_id: str):
    with _event_lock:
        if session_id in _event_queues:
            del _event_queues[session_id]

def send_event(session_id: str, event_type: str, data: Any):
    """Send an event to the client via SSE."""
    q = get_or_create_queue(session_id)
    q.put({"type": event_type, "data": data, "timestamp": time.time()})

# Monkey-patch the logging to capture events
_original_log = multi_agent.log

def _patched_log(msg: Any = "", level: str = "INFO"):
    """Patched log function that also sends events to the web client."""
    _original_log(msg, level)
    # Broadcast to all active sessions
    with _event_lock:
        for sid, q in list(_event_queues.items()):
            try:
                q.put({
                    "type": "log",
                    "data": {"message": str(msg), "level": level},
                    "timestamp": time.time()
                }, block=False)
            except queue.Full:
                pass

multi_agent.log = _patched_log


class ProcessTracker:
    """Tracks the RAG process for visualization."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.graphrag_iterations: List[Dict[str, Any]] = []
        self.naiverag_iterations: List[Dict[str, Any]] = []
        self.amendment_data: Dict[str, Any] = {}
        self.aggregator_result: Dict[str, Any] = {}
        self.final_answer: str = ""
        
    def emit(self, event_type: str, data: Any):
        send_event(self.session_id, event_type, data)


def run_pipeline_with_tracking(query: str, mode: str, session_id: str) -> Dict[str, Any]:
    """
    Run the multi-agent pipeline with process tracking.
    
    Args:
        query: User's legal question
        mode: 'before_aggregation' or 'after_aggregation'
        session_id: Unique session identifier for SSE
    
    Returns:
        Complete result with process details
    """
    tracker = ProcessTracker(session_id)
    
    tracker.emit("status", {"message": "Starting pipeline...", "phase": "init"})
    
    # Detect language
    user_lang = multi_agent.detect_user_language(query)
    tracker.emit("language_detected", {"language": user_lang})
    
    # Build ChunkStore
    tracker.emit("status", {"message": "Building chunk store...", "phase": "init"})
    chunk_store = multi_agent.ChunkStore(multi_agent.LANGCHAIN_DIR, set(multi_agent.SKIP_FILES))
    
    # Results containers
    graphrag_result: Dict[str, Any] = {}
    naiverag_result: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    
    def run_graphrag_tracked():
        """Run GraphRAG with detailed tracking."""
        try:
            multi_agent.set_log_context("G", None)
            tracker.emit("pipeline_start", {"pipeline": "graphrag"})
            
            qaf_history: List[Dict[str, Any]] = []
            per_iteration: List[Dict[str, Any]] = []
            current_query = query
            final_answer = ""
            
            for i in range(1, multi_agent.MAX_ANSWER_JUDGE_ITERS + 1):
                multi_agent.set_log_context("G", f"{i}/{multi_agent.MAX_ANSWER_JUDGE_ITERS}")
                
                tracker.emit("iteration_start", {
                    "pipeline": "graphrag",
                    "iteration": i,
                    "max_iterations": multi_agent.MAX_ANSWER_JUDGE_ITERS,
                    "current_query": current_query
                })
                
                # Run retrieval
                tracker.emit("status", {
                    "message": f"GraphRAG Iteration {i}: Running retrieval...",
                    "phase": "retrieval",
                    "pipeline": "graphrag"
                })
                
                retr = multi_agent.run_retrieval_for_query_graph(
                    current_query, chunk_store, user_lang, cand_limit_override=None
                )
                context_text = retr["context_text"]
                context_summary = retr["context_summary"]
                diagnostics = retr["diagnostics"]
                
                tracker.emit("retrieval_complete", {
                    "pipeline": "graphrag",
                    "iteration": i,
                    "diagnostics": diagnostics,
                    "context_summary": context_summary[:500] + "..." if len(context_summary) > 500 else context_summary
                })
                
                # Generate answer
                tracker.emit("status", {
                    "message": f"GraphRAG Iteration {i}: Generating answer...",
                    "phase": "answering",
                    "pipeline": "graphrag"
                })
                
                answer = multi_agent.agent2_answer(
                    current_query, context_text, guidance=None, output_lang=user_lang
                )
                
                tracker.emit("answer_generated", {
                    "pipeline": "graphrag",
                    "iteration": i,
                    "answer": answer,
                    "query": current_query
                })
                
                # Answer Judge
                tracker.emit("status", {
                    "message": f"GraphRAG Iteration {i}: Running Answer Judge...",
                    "phase": "judging",
                    "pipeline": "graphrag"
                })
                
                aj = multi_agent.agent_aj_answer_judge(current_query, answer, qaf_history, user_lang)
                acceptable = bool(aj.get("acceptable"))
                problems = (aj.get("problems") or "").strip()
                suggestion = (aj.get("suggestion") or "").strip()
                notes = (aj.get("notes") or "").strip()
                
                tracker.emit("judge_decision", {
                    "pipeline": "graphrag",
                    "iteration": i,
                    "acceptable": acceptable,
                    "problems": problems,
                    "suggestion": suggestion,
                    "notes": notes
                })
                
                iteration_data = {
                    "iteration": i,
                    "query": current_query,
                    "answer": answer,
                    "judge": {
                        "acceptable": acceptable,
                        "problems": problems,
                        "suggestion": suggestion,
                        "notes": notes
                    },
                    "modified_query": None,
                    "context_summary": context_summary,
                    "retrieval_diagnostics": diagnostics
                }
                
                if acceptable:
                    final_answer = answer
                    per_iteration.append(iteration_data)
                    tracker.emit("iteration_complete", {
                        "pipeline": "graphrag",
                        "iteration": i,
                        "status": "accepted",
                        "final": True
                    })
                    break
                    
                if i >= multi_agent.MAX_ANSWER_JUDGE_ITERS:
                    final_answer = answer
                    iteration_data["judge"]["stopped_at_cap"] = True
                    per_iteration.append(iteration_data)
                    tracker.emit("iteration_complete", {
                        "pipeline": "graphrag",
                        "iteration": i,
                        "status": "cap_reached",
                        "final": True
                    })
                    break
                
                # Query Modifier
                tracker.emit("status", {
                    "message": f"GraphRAG Iteration {i}: Modifying query...",
                    "phase": "modifying",
                    "pipeline": "graphrag"
                })
                
                qm = multi_agent.agent_qm_modify_query(
                    current_query, answer, problems, suggestion, qaf_history, user_lang
                )
                modified_query = (qm.get("modified_query") or current_query).strip()
                rationale = (qm.get("rationale") or "").strip()
                
                tracker.emit("query_modified", {
                    "pipeline": "graphrag",
                    "iteration": i,
                    "original_query": current_query,
                    "modified_query": modified_query,
                    "rationale": rationale
                })
                
                qaf_history.append({
                    "query": current_query,
                    "answer": answer,
                    "feedback": {"problems": problems, "suggestion": suggestion},
                    "modified_query": modified_query
                })
                
                iteration_data["modified_query"] = modified_query
                iteration_data["modifier_rationale"] = rationale
                per_iteration.append(iteration_data)
                
                current_query = modified_query
                
                tracker.emit("iteration_complete", {
                    "pipeline": "graphrag",
                    "iteration": i,
                    "status": "continuing",
                    "final": False
                })
            
            if not final_answer and per_iteration:
                final_answer = per_iteration[-1].get("answer", "") or "(No answer produced)"
            
            graphrag_result.update({
                "final_answer": final_answer,
                "iterations_used": len(per_iteration),
                "per_iteration": per_iteration
            })
            
            tracker.emit("pipeline_complete", {
                "pipeline": "graphrag",
                "iterations_used": len(per_iteration),
                "final_answer": final_answer
            })
            
        except Exception as e:
            errors["graphrag"] = str(e)
            tracker.emit("pipeline_error", {"pipeline": "graphrag", "error": str(e)})
    
    def run_naiverag_tracked():
        """Run NaiveRAG with detailed tracking."""
        try:
            multi_agent.set_log_context("N", None)
            tracker.emit("pipeline_start", {"pipeline": "naiverag"})
            
            qaf_history: List[Dict[str, Any]] = []
            iteration_runs: List[Dict[str, Any]] = []
            final_answer = ""
            current_query = query.strip()
            
            for it in range(1, multi_agent.MAX_ANSWER_JUDGE_ITERS + 1):
                multi_agent.set_log_context("N", f"{it}/{multi_agent.MAX_ANSWER_JUDGE_ITERS}")
                
                tracker.emit("iteration_start", {
                    "pipeline": "naiverag",
                    "iteration": it,
                    "max_iterations": multi_agent.MAX_ANSWER_JUDGE_ITERS,
                    "current_query": current_query
                })
                
                # Embed query
                tracker.emit("status", {
                    "message": f"NaiveRAG Iteration {it}: Embedding query...",
                    "phase": "embedding",
                    "pipeline": "naiverag"
                })
                
                q_emb = multi_agent.embed_text(current_query)
                
                # Vector search
                tracker.emit("status", {
                    "message": f"NaiveRAG Iteration {it}: Vector search...",
                    "phase": "retrieval",
                    "pipeline": "naiverag"
                })
                
                candidates = multi_agent.vector_query_chunks(q_emb, k=multi_agent.TOP_K_CHUNKS)
                
                tracker.emit("retrieval_complete", {
                    "pipeline": "naiverag",
                    "iteration": it,
                    "num_candidates": len(candidates),
                    "top_chunks": [
                        {
                            "document_id": c.get("document_id"),
                            "chunk_id": c.get("chunk_id"),
                            "uu_number": c.get("uu_number"),
                            "score": c.get("score")
                        }
                        for c in candidates[:5]
                    ] if candidates else []
                })
                
                if not candidates:
                    context_text = "Tidak ada potongan teks yang ditemukan." if user_lang == "id" else "No relevant chunks found."
                else:
                    context_text = multi_agent.build_context_from_chunks(
                        candidates, max_chunks=multi_agent.NAIVE_MAX_CHUNKS_FINAL
                    )
                
                # Generate answer
                tracker.emit("status", {
                    "message": f"NaiveRAG Iteration {it}: Generating answer...",
                    "phase": "answering",
                    "pipeline": "naiverag"
                })
                
                answer = multi_agent.agent2_answer(
                    current_query, context_text, guidance=None, output_lang=user_lang
                )
                
                tracker.emit("answer_generated", {
                    "pipeline": "naiverag",
                    "iteration": it,
                    "answer": answer,
                    "query": current_query
                })
                
                meta_top = [
                    {
                        "document_id": c.get("document_id"),
                        "chunk_id": c.get("chunk_id"),
                        "uu_number": c.get("uu_number"),
                        "pages": c.get("pages"),
                        "score": c.get("score"),
                    }
                    for c in (candidates[:5] if candidates else [])
                ]
                
                if it >= multi_agent.MAX_ANSWER_JUDGE_ITERS:
                    final_answer = answer
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
                    tracker.emit("iteration_complete", {
                        "pipeline": "naiverag",
                        "iteration": it,
                        "status": "cap_reached",
                        "final": True
                    })
                    break
                
                # Answer Judge
                tracker.emit("status", {
                    "message": f"NaiveRAG Iteration {it}: Running Answer Judge...",
                    "phase": "judging",
                    "pipeline": "naiverag"
                })
                
                judge = multi_agent.answer_judge_naive(current_query, answer, qaf_history, output_lang=user_lang)
                decision = judge.get("decision", "insufficient")
                
                tracker.emit("judge_decision", {
                    "pipeline": "naiverag",
                    "iteration": it,
                    "decision": decision,
                    "reasoning": judge.get("reasoning", ""),
                    "problem": judge.get("problem", ""),
                    "suggested_solution": judge.get("suggested_solution", "")
                })
                
                if decision == "acceptable":
                    final_answer = answer
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
                    tracker.emit("iteration_complete", {
                        "pipeline": "naiverag",
                        "iteration": it,
                        "status": "accepted",
                        "final": True
                    })
                    break
                else:
                    fb = {
                        "problem": judge.get("problem", ""),
                        "suggested_solution": judge.get("suggested_solution", "")
                    }
                    qaf_history.append({
                        "iteration": it,
                        "query": current_query,
                        "answer": answer,
                        "feedback": fb
                    })
                    
                    # Query Modifier
                    tracker.emit("status", {
                        "message": f"NaiveRAG Iteration {it}: Modifying query...",
                        "phase": "modifying",
                        "pipeline": "naiverag"
                    })
                    
                    mod = multi_agent.query_modifier_naive(
                        current_query, answer, fb, qaf_history, output_lang=user_lang
                    )
                    new_query = mod.get("modified_query", "").strip() or current_query
                    notes = mod.get("notes", "")
                    
                    tracker.emit("query_modified", {
                        "pipeline": "naiverag",
                        "iteration": it,
                        "original_query": current_query,
                        "modified_query": new_query,
                        "notes": notes
                    })
                    
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
                    
                    tracker.emit("iteration_complete", {
                        "pipeline": "naiverag",
                        "iteration": it,
                        "status": "continuing",
                        "final": False
                    })
            
            if not final_answer and iteration_runs:
                final_answer = iteration_runs[-1].get("answer", "") or "(No answer produced)"
            
            naiverag_result.update({
                "final_answer": final_answer,
                "iterations_used": len(iteration_runs),
                "iteration_runs": iteration_runs
            })
            
            tracker.emit("pipeline_complete", {
                "pipeline": "naiverag",
                "iterations_used": len(iteration_runs),
                "final_answer": final_answer
            })
            
        except Exception as e:
            errors["naiverag"] = str(e)
            tracker.emit("pipeline_error", {"pipeline": "naiverag", "error": str(e)})
    
    def apply_amendments_tracked(answer: str, pipeline_name: str) -> Dict[str, Any]:
        """Apply amendment-aware processing with tracking."""
        tracker.emit("amendment_start", {"pipeline": pipeline_name})
        
        # Extract UU references
        tracker.emit("status", {
            "message": f"Extracting UU references from {pipeline_name} answer...",
            "phase": "uu_extraction",
            "pipeline": pipeline_name
        })
        
        refs = multi_agent.extract_uu_references_llm(answer)
        
        tracker.emit("uu_references_extracted", {
            "pipeline": pipeline_name,
            "references": refs
        })
        
        if not refs:
            tracker.emit("amendment_complete", {
                "pipeline": pipeline_name,
                "has_amendments": False,
                "message": "No UU references found"
            })
            return {
                "final_answer": answer,
                "has_amendments": False,
                "currency_warnings": "",
                "amendment_info": [],
                "amending_chunks_used": 0,
                "uu_references": []
            }
        
        # Traverse amendment chains
        tracker.emit("status", {
            "message": f"Traversing amendment chains for {pipeline_name}...",
            "phase": "chain_traversal",
            "pipeline": pipeline_name
        })
        
        chain_results = []
        amending_set = set()
        all_amendment_info = []
        
        for r in refs:
            num = r.get("number")
            yr = r.get("year")
            if num is None or yr is None:
                continue
            chain = multi_agent.traverse_amendment_chain_with_reset_cached(int(num), int(yr))
            chain_results.append(chain)
            if chain.get("has_amendments"):
                for uu in chain.get("relevant_uus", []):
                    if uu != chain.get("original_uu"):
                        amending_set.add(uu)
                all_amendment_info.extend(chain.get("amendment_info", []))
        
        tracker.emit("amendment_chains_retrieved", {
            "pipeline": pipeline_name,
            "chain_results": chain_results,
            "amending_uus": list(amending_set),
            "amendment_info": all_amendment_info
        })
        
        currency_warning = multi_agent.generate_currency_warning(chain_results) if chain_results else ""
        
        if not amending_set:
            final = (currency_warning + "\n" if currency_warning else "") + answer
            tracker.emit("amendment_complete", {
                "pipeline": pipeline_name,
                "has_amendments": False,
                "message": "No amending UUs found"
            })
            return {
                "final_answer": final,
                "has_amendments": False,
                "currency_warnings": currency_warning,
                "amendment_info": [],
                "amending_chunks_used": 0,
                "uu_references": refs
            }
        
        # Retrieve chunks from amending documents
        tracker.emit("status", {
            "message": f"Retrieving chunks from amending documents for {pipeline_name}...",
            "phase": "amending_retrieval",
            "pipeline": pipeline_name
        })
        
        q_emb = multi_agent.embed_text(query)
        amending_chunks = multi_agent.vector_query_chunks_filtered(
            q_emb, k=multi_agent.TOP_K_CHUNKS, uu_filters=list(amending_set)
        )
        
        if not amending_chunks:
            final = answer
            if currency_warning:
                final = currency_warning + "\n" + final
            final += "\nCatatan: UU yang dirujuk mengalami amandemen, namun tidak ditemukan potongan relevan yang mengubah jawaban di atas."
            tracker.emit("amendment_complete", {
                "pipeline": pipeline_name,
                "has_amendments": True,
                "message": "No relevant amending chunks found",
                "amending_chunks_used": 0
            })
            return {
                "final_answer": final,
                "has_amendments": True,
                "currency_warnings": currency_warning,
                "amendment_info": all_amendment_info,
                "amending_chunks_used": 0,
                "uu_references": refs
            }
        
        # Judge relevance
        tracker.emit("status", {
            "message": f"Judging amendment relevance for {pipeline_name}...",
            "phase": "relevance_judging",
            "pipeline": pipeline_name
        })
        
        rel = multi_agent.judge_amendment_relevance(query, answer, amending_chunks)
        
        tracker.emit("relevance_judge_decision", {
            "pipeline": pipeline_name,
            "is_relevant": rel.get("is_relevant"),
            "affected_aspects": rel.get("affected_aspects", []),
            "reasoning": rel.get("reasoning", "")
        })
        
        if not rel.get("is_relevant"):
            final = answer
            if currency_warning:
                final = currency_warning + "\n" + final
            final += "\nCatatan: Amandemen tidak mempengaruhi aspek yang ditanyakan; jawaban tetap berlaku."
            tracker.emit("amendment_complete", {
                "pipeline": pipeline_name,
                "has_amendments": True,
                "message": "Amendments not relevant to query",
                "relevance_result": rel
            })
            return {
                "final_answer": final,
                "has_amendments": True,
                "currency_warnings": currency_warning,
                "amendment_info": all_amendment_info,
                "amending_chunks_used": 0,
                "uu_references": refs,
                "relevance_result": rel
            }
        
        # Integrate amendments
        tracker.emit("status", {
            "message": f"Integrating amendments for {pipeline_name}...",
            "phase": "integration",
            "pipeline": pipeline_name
        })
        
        new_context = multi_agent.build_context_from_chunks(
            amending_chunks, max_chunks=multi_agent.MAX_CHUNKS_FINAL
        )
        integrated = multi_agent.integrate_amendments(
            query, answer, new_context, all_amendment_info, rel, output_lang=user_lang
        )
        final_answer = (currency_warning + "\n" if currency_warning else "") + integrated
        
        tracker.emit("amendment_complete", {
            "pipeline": pipeline_name,
            "has_amendments": True,
            "message": "Amendments integrated",
            "amending_chunks_used": len(amending_chunks),
            "relevance_result": rel
        })
        
        return {
            "final_answer": final_answer,
            "has_amendments": True,
            "currency_warnings": currency_warning,
            "amendment_info": all_amendment_info,
            "amending_chunks_used": len(amending_chunks),
            "uu_references": refs,
            "relevance_result": rel
        }
    
    # Run both pipelines in parallel
    tracker.emit("status", {"message": "Starting parallel pipeline execution...", "phase": "parallel_start"})
    
    t_graphrag = threading.Thread(target=run_graphrag_tracked, name="GraphRAGThread", daemon=True)
    t_naiverag = threading.Thread(target=run_naiverag_tracked, name="NaiveRAGThread", daemon=True)
    
    t_graphrag.start()
    t_naiverag.start()
    t_graphrag.join()
    t_naiverag.join()
    
    g_answer = (graphrag_result.get("final_answer") or "").strip()
    n_answer = (naiverag_result.get("final_answer") or "").strip()
    
    # Apply amendments based on mode
    if mode == "before_aggregation":
        tracker.emit("status", {
            "message": "Applying amendments BEFORE aggregation (per-pipeline)...",
            "phase": "amendment_before"
        })
        
        # Apply amendments to each pipeline's answer
        if g_answer:
            g_amendment = apply_amendments_tracked(g_answer, "graphrag")
            graphrag_result["final_answer"] = g_amendment["final_answer"]
            graphrag_result["amendment"] = g_amendment
        else:
            graphrag_result["amendment"] = {
                "has_amendments": False,
                "currency_warnings": "",
                "amendment_info": [],
                "amending_chunks_used": 0,
                "uu_references": []
            }
        
        if n_answer:
            n_amendment = apply_amendments_tracked(n_answer, "naiverag")
            naiverag_result["final_answer"] = n_amendment["final_answer"]
            naiverag_result["amendment"] = n_amendment
        else:
            naiverag_result["amendment"] = {
                "has_amendments": False,
                "currency_warnings": "",
                "amendment_info": [],
                "amending_chunks_used": 0,
                "uu_references": []
            }
        
        g_answer = graphrag_result.get("final_answer", "")
        n_answer = naiverag_result.get("final_answer", "")
    
    # Run aggregator
    tracker.emit("status", {"message": "Running aggregator...", "phase": "aggregation"})
    
    if "graphrag" in errors and not g_answer and n_answer:
        aggregator_result = {
            "decision": "choose_naiverag",
            "final_answer": n_answer,
            "rationale": "GraphRAG failed. Using NaiveRAG answer."
        }
    elif "naiverag" in errors and not n_answer and g_answer:
        aggregator_result = {
            "decision": "choose_graphrag",
            "final_answer": g_answer,
            "rationale": "NaiveRAG failed. Using GraphRAG answer."
        }
    else:
        g_meta = {
            "iterations_used": graphrag_result.get("iterations_used"),
            "context_hint": "Graph triples + chunks",
            "per_iteration_count": len(graphrag_result.get("per_iteration", []) or []),
            "amendment": graphrag_result.get("amendment", {})
        }
        n_meta = {
            "iterations_used": naiverag_result.get("iterations_used"),
            "context_hint": "Top chunk content",
            "iteration_runs_count": len(naiverag_result.get("iteration_runs", []) or []),
            "amendment": naiverag_result.get("amendment", {})
        }
        aggregator_result = multi_agent.aggregator_agent(
            query, g_answer, n_answer, g_meta, n_meta, user_lang
        )
    
    tracker.emit("aggregator_decision", aggregator_result)
    
    final_answer = aggregator_result.get("final_answer", "")
    
    # Apply amendments after aggregation if mode is after_aggregation
    if mode == "after_aggregation":
        tracker.emit("status", {
            "message": "Applying amendments AFTER aggregation...",
            "phase": "amendment_after"
        })
        
        amendment_result = apply_amendments_tracked(final_answer, "aggregated")
        final_answer = amendment_result["final_answer"]
        
        # Store amendment info at top level
        graphrag_result["amendment"] = {
            "has_amendments": False,
            "note": "Amendment processing done after aggregation"
        }
        naiverag_result["amendment"] = {
            "has_amendments": False,
            "note": "Amendment processing done after aggregation"
        }
        aggregator_result["amendment"] = amendment_result
    
    aggregator_result["final_answer"] = final_answer
    
    tracker.emit("complete", {
        "final_answer": final_answer,
        "graphrag_iterations": graphrag_result.get("iterations_used", 0),
        "naiverag_iterations": naiverag_result.get("iterations_used", 0),
        "aggregator_decision": aggregator_result.get("decision", "")
    })
    
    return {
        "final_answer": final_answer,
        "aggregator": aggregator_result,
        "graphrag": graphrag_result,
        "naiverag": naiverag_result,
        "mode": mode,
        "errors": errors if errors else None
    }


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query():
    """Process a legal question."""
    data = request.get_json()
    question = data.get("question", "").strip()
    mode = data.get("mode", "before_aggregation")
    session_id = data.get("session_id", str(time.time()))
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    if mode not in ("before_aggregation", "after_aggregation"):
        return jsonify({"error": "Invalid mode. Use 'before_aggregation' or 'after_aggregation'"}), 400
    
    # Create a new queue for this session
    get_or_create_queue(session_id)
    
    def run_in_background():
        try:
            result = run_pipeline_with_tracking(question, mode, session_id)
            send_event(session_id, "result", result)
        except Exception as e:
            send_event(session_id, "error", {"message": str(e)})
        finally:
            # Signal completion
            send_event(session_id, "done", {})
    
    # Start processing in background
    thread = threading.Thread(target=run_in_background, daemon=True)
    thread.start()
    
    return jsonify({"status": "processing", "session_id": session_id})


@app.route("/api/events/<session_id>")
def events(session_id: str):
    """Server-Sent Events endpoint for real-time updates."""
    def generate():
        q = get_or_create_queue(session_id)
        try:
            while True:
                try:
                    event = q.get(timeout=30)
                    if event.get("type") == "done":
                        yield f"data: {json.dumps(event)}\n\n"
                        break
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    # Send keepalive
                    yield f": keepalive\n\n"
        finally:
            remove_queue(session_id)
    
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.route("/api/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "timestamp": time.time()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)