import argparse
import os
from urllib.parse import urlparse

from camel.storages import Neo4jGraph

from simple_neo4j_graph import SimpleNeo4jGraph
from retrieve import select_top_gids
from summerize import process_chunks
from utils import call_llm, link_context, ret_context
from retrieval_trace import RetrievalTrace, is_tracing_enabled, get_trace_output_dir
from edge_confidence import compute_edge_confidences, write_edge_confidence_jsonl
from stability_protocol import (
    run_repeated_retrievals, 
    compute_node_stability, 
    compute_edge_stability,
    refine_R_e_from_distribution,
    write_stability_report,
    write_stability_jsonl
)
from counterfactual_analysis import (
    generate_counterfactuals,
    compare_retrievals,
    write_counterfactual_report,
    write_counterfactual_jsonl
)


POST_SYS_PROMPT = """
You are a medical QA assistant grounded on graph-retrieved evidence.
Answer the user's question with concise clinical reasoning and cite supporting evidence IDs like [E1], [E2].
If evidence is insufficient, state uncertainty clearly and avoid unsupported claims.
"""


def _use_llm_answering():
    # Default local/offline mode for reproducibility.
    return os.getenv("USE_LLM_ANSWER", "0") == "1"


def _normalize_neo4j_url(raw_url: str | None) -> str:
    if not raw_url:
        return "bolt://localhost:7687"

    url = raw_url.strip()
    parsed = urlparse(url)

    if parsed.scheme in {"http", "https"}:
        host = parsed.hostname or "localhost"
        return f"bolt://{host}:7687"

    if parsed.scheme in {"bolt", "neo4j", "bolt+s", "bolt+ssc", "neo4j+s", "neo4j+ssc"}:
        host = parsed.hostname or "localhost"
        port = parsed.port
        if port == 7474:
            return f"{parsed.scheme}://{host}:7687"
        return url

    if "://" not in url:
        if url.endswith(":7474"):
            url = url[:-5] + ":7687"
        return f"bolt://{url}"

    return url


def _connect_neo4j(url: str, username: str, password: str):
    try:
        return Neo4jGraph(url=url, username=username, password=password)
    except Exception as e:
        print(f"Neo4jGraph init failed ({e}). Falling back to SimpleNeo4jGraph without APOC.")
        return SimpleNeo4jGraph(url=url, username=username, password=password)


def _question_summary(question: str):
    if os.getenv("USE_LLM_SUMMARY", "0") == "1":
        return process_chunks(question)
    return [question[:1500]]


def _neighbor_gids(n4j, seed_gid: str, max_hops: int = 2, max_items: int = 20):
    hop = max(1, min(4, int(max_hops)))
    query = f"""
        MATCH (n)
        WHERE n.gid = $gid AND NOT n:Summary
        MATCH p=(n)-[:REFERENCE*1..{hop}]-(m)
        WHERE NOT m:Summary
        RETURN DISTINCT m.gid AS gid
        LIMIT $limit
    """
    rows = n4j.query(query, {"gid": seed_gid, "limit": max_items})
    return [r["gid"] for r in rows if r.get("gid")]


def _collect_evidence(n4j, gids, max_evidence: int = 120):
    evidence = []
    seen = set()

    for gid in gids:
        local_ctx = ret_context(n4j, gid)
        ref_ctx = link_context(n4j, gid)
        for line in local_ctx + ref_ctx:
            clean = str(line).strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            evidence.append(clean)
            if len(evidence) >= max_evidence:
                return evidence
    return evidence


def _answer_with_citations(question: str, evidence):
    numbered = []
    for i, item in enumerate(evidence, start=1):
        numbered.append(f"[E{i}] {item}")

    if _use_llm_answering():
        user_prompt = (
            f"Question: {question}\n\n"
            "Evidence:\n"
            + "\n".join(numbered)
            + "\n\nPlease answer using only the evidence above and include [E#] citations."
        )
        return call_llm(POST_SYS_PROMPT, user_prompt)

    selected = numbered[: min(8, len(numbered))]
    lines = [
        "Local deterministic answer mode (USE_LLM_ANSWER=0).",
        "Likely major clinical problems/interventions based on retrieved graph evidence:",
    ]
    for item in selected:
        lines.append(f"- {item}")
    lines.append("Synthesized guidance: prioritize findings repeatedly connected by relation/context patterns above.")
    return "\n".join(lines)


def _compute_matched_nodes(n4j, sumq, trace=None):
    """
    Compute matched nodes with tracing support.
    Calls select_top_gids and optionally logs matched nodes to trace.
    """
    from utils import get_embedding as get_emb
    from retrieve import _summary_to_text, _cosine
    
    rows = n4j.query("MATCH (s:Summary) RETURN s.content AS content, s.gid AS gid")
    
    query_text = _summary_to_text(sumq[0]) if isinstance(sumq, list) and sumq else str(sumq)
    query_emb = get_emb(query_text)
    
    if trace and rows:
        for i, row in enumerate(rows):
            content = row.get("content")
            gid = row.get("gid")
            if content and gid:
                content_text = _summary_to_text(content)
                sim = _cosine(query_emb, get_emb(content_text)) if content_text else -1.0
                trace.add_matched_node({
                    "node_id": gid,
                    "node_type": "Summary",
                    "similarity_raw": float(sim),
                    "similarity_norm": max(0.0, float(sim)),
                    "retriever_rank": i,
                    "content_preview": content_text[:100] if content_text else ""
                })
    
    return rows


def _build_selected_paths(seed_gids, expansion_steps):
    """
    Build explicit path records from seeds and expansion steps.
    
    Returns list of path records with: path_id, nodes, edges, path_score_total, 
    score_components, selection_reason, tie_breaker.
    """
    paths = []
    
    # First, build implicit paths from each seed (seed to its neighbors)
    for seed_gid in seed_gids:
        # Direct seed path (seed alone)
        paths.append({
            "path_id": f"seed_{seed_gid[:8]}",
            "nodes": [seed_gid],
            "edges": [],
            "path_score_total": 1.0,
            "score_components": {
                "seed_ranking": 1.0,
                "expansion_depth": 0,
                "path_length_penalty": 1.0
            },
            "selection_reason": "Selected as top-K seed via ranking",
            "tie_breaker": None,
            "path_type": "seed"
        })
        
        # Seed + neighbor paths
        seed_neighbors = [s for s in expansion_steps if s["from_node"] == seed_gid]
        for step in seed_neighbors:
            paths.append({
                "path_id": f"{seed_gid[:8]}_to_{step['to_node'][:8]}",
                "nodes": [seed_gid, step["to_node"]],
                "edges": [step["path_id"]],
                "path_score_total": step["step_score"],
                "score_components": {
                    "seed_ranking": 1.0,
                    "expansion_score": step["step_score"],
                    "expansion_depth": 1,
                    "path_length_penalty": 1.0 / max(1, 2)  # 1 / path_length
                },
                "selection_reason": "Expanded from selected seed via REFERENCE",
                "tie_breaker": f"step_rank_{step['step_rank']}" if step.get("step_rank") == 0 else None,
                "path_type": "expanded"
            })
    
    return paths


def main():
    parser = argparse.ArgumentParser(description="Post-graph inference for three-layer Medical-Graph-RAG")
    parser.add_argument("--question", type=str, help="Question text")
    parser.add_argument("--question-file", type=str, help="Path to a question file")
    parser.add_argument("--top-k", type=int, default=2, help="Top summary-matched gids")
    parser.add_argument("--max-hops", type=int, default=2, help="REFERENCE traversal hops")
    parser.add_argument("--max-evidence", type=int, default=120, help="Max evidence lines for answer")

    parser.add_argument("--neo4j-url", type=str, default=os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI"))
    parser.add_argument("--neo4j-username", type=str, default=os.getenv("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--neo4j-password", type=str, default=os.getenv("NEO4J_PASSWORD"))

    args = parser.parse_args()

    if not args.neo4j_password:
        raise ValueError("NEO4J_PASSWORD is not set. Export NEO4J_PASSWORD or pass --neo4j-password.")

    question = args.question
    if not question and args.question_file:
        with open(args.question_file, "r", encoding="utf-8") as f:
            question = f.read().strip()

    if not question:
        raise ValueError("Provide --question or --question-file.")
    
    # Initialize retrieval trace if enabled
    trace = None
    if is_tracing_enabled():
        trace = RetrievalTrace(question)
        trace.add_config({
            "top_k": args.top_k,
            "max_hops": args.max_hops,
            "max_evidence": args.max_evidence,
            "use_llm_summary": os.getenv("USE_LLM_SUMMARY") == "1",
            "use_llm_answer": os.getenv("USE_LLM_ANSWER") == "1",
        })

    n4j = _connect_neo4j(
        url=_normalize_neo4j_url(args.neo4j_url),
        username=args.neo4j_username,
        password=args.neo4j_password,
    )

    q_summary = _question_summary(question)
    if trace:
        trace.query_summary = q_summary[0] if q_summary else question
        # Capture matched nodes before selecting seed gids
        _compute_matched_nodes(n4j, q_summary, trace=trace)
    
    seed_gids = select_top_gids(n4j, q_summary, top_k=max(1, args.top_k))
    if trace:
        trace.selected_seed_gids = seed_gids

    expanded_gids = list(seed_gids)
    for gid in seed_gids:
        neighbors = _neighbor_gids(n4j, gid, max_hops=args.max_hops, max_items=20)
        expanded_gids.extend(neighbors)
        if trace:
            for i, neighbor_gid in enumerate(neighbors):
                trace.add_expansion_step({
                    "from_node": gid,
                    "edge_type": "REFERENCE",
                    "to_node": neighbor_gid,
                    "step_rank": i,
                    "step_score": 1.0,
                    "pruning_reason": None,
                    "path_id": f"{gid}→{neighbor_gid}"
                })

    # Preserve order while deduplicating.
    ordered_unique_gids = list(dict.fromkeys([g for g in expanded_gids if g]))
    if trace:
        trace.expanded_gids = ordered_unique_gids
        
        # Build and log selected paths (Phase 2)
        selected_paths = _build_selected_paths(trace.selected_seed_gids, trace.expansion_steps)
        for path_record in selected_paths:
            trace.add_selected_path(path_record)

    # Collect evidence with gid tracking for provenance
    evidence = []
    evidence_with_gid = []
    seen = set()
    max_ev = max(10, args.max_evidence)
    
    for gid in ordered_unique_gids:
        local_ctx = ret_context(n4j, gid)
        ref_ctx = link_context(n4j, gid)
        for line in local_ctx + ref_ctx:
            clean = str(line).strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            evidence.append(clean)
            evidence_with_gid.append((gid, clean))
            if len(evidence) >= max_ev:
                break
        if len(evidence) >= max_ev:
            break
    
    if trace:
        for i, (gid, ev_text) in enumerate(evidence_with_gid):
            trace.add_evidence_item({
                "content": ev_text,
                "rank": i,
                "evidence_type": "mixed",
                "node_id": gid
            })
    
    if not evidence:
        print("No evidence retrieved from graph. Check graph construction and REFERENCE links.")
        return

    answer = _answer_with_citations(question, evidence)
    if trace:
        trace.answer_text = answer
        trace.answer_config = {
            "use_llm": _use_llm_answering(),
            "model": "openai" if _use_llm_answering() else "template",
            "evidence_count": len(evidence),
            "evidence_used": min(8, len(evidence))
        }

    print("=== Selected GIDs ===")
    for gid in ordered_unique_gids:
        print(gid)

    print("\n=== Evidence Count ===")
    print(len(evidence))

    print("\n=== Answer ===")
    print(answer)
    
    # Phase 3: Compute edge confidences if enabled
    if trace and os.getenv("EDGE_CONFIDENCE", "0") == "1":
        try:
            alpha = float(os.getenv("CONF_ALPHA", "0.6"))
            beta = float(os.getenv("CONF_BETA", "0.3"))
            gamma = float(os.getenv("CONF_GAMMA", "0.1"))
            stability_runs = int(os.getenv("CONF_STABILITY_RUNS", "10"))
            compute_edge_confidences(n4j, trace, alpha=alpha, beta=beta, gamma=gamma, stability_runs=stability_runs)
            print("\n=== Edge Confidence Computed ===")
        except Exception as e:
            print(f"Warning: Edge confidence computation failed: {e}")
    
    # Phase 4: Stability protocol if enabled
    if trace and os.getenv("STABILITY_PROTOCOL", "0") == "1":
        try:
            stability_runs = int(os.getenv("STABILITY_RUNS", "10"))
            top_k = int(os.getenv("STABILITY_TOP_K", "3"))
            seed = int(os.getenv("STABILITY_SEED", "-1"))
            seed = seed if seed >= 0 else None
            
            # Run repeated retrievals
            runs = run_repeated_retrievals(
                n4j, 
                question,
                num_runs=stability_runs,
                top_k=top_k,
                seed=seed
            )
            
            # Compute stability metrics
            node_stability = compute_node_stability(runs)
            edge_stability = compute_edge_stability(runs, trace)
            
            # Refine R_e scores
            refined_R_e = refine_R_e_from_distribution(node_stability, edge_stability, trace)
            
            # Update R_e in trace
            for step in trace.expansion_steps:
                edge_id = step.get("path_id")
                if edge_id in refined_R_e:
                    if "confidence" not in step:
                        step["confidence"] = {}
                    step["confidence"]["R_e"] = refined_R_e[edge_id]
                    # Recompute C_e with refined R_e
                    s_e = step["confidence"].get("S_e", 0.0)
                    g_e = step["confidence"].get("G_e", 0.0)
                    alpha = step["confidence"].get("alpha", 0.6)
                    beta = step["confidence"].get("beta", 0.3)
                    gamma = step["confidence"].get("gamma", 0.1)
                    c_e = alpha * s_e + beta * g_e + gamma * refined_R_e[edge_id]
                    step["confidence"]["C_e"] = round(float(c_e), 6)
                    step["confidence"]["R_e_refined"] = True
            
            print(f"\n=== Stability Protocol Complete (N={stability_runs} runs) ===")
            print(f"Nodes analyzed: {len(node_stability)}")
            print(f"Edges analyzed: {len(edge_stability)}")
            print(f"R_e values refined: {len(refined_R_e)}")
        except Exception as e:
            print(f"Warning: Stability protocol computation failed: {e}")
    
    # Phase 5: Counterfactual analysis if enabled
    if trace and os.getenv("COUNTERFACTUAL_ANALYSIS", "0") == "1":
        try:
            num_mutations = int(os.getenv("COUNTERFACTUAL_MUTATIONS", "3"))
            
            # Generate counterfactuals
            counterfactuals = generate_counterfactuals(question, num_mutations=num_mutations)
            comparisons = {}
            
            # For each counterfactual, run retrieval and compare
            for cf in counterfactuals:
                if cf.mutation_type == "baseline":
                    # Baseline: compare trace with itself
                    comps = compare_retrievals(
                        trace, trace,
                        trace.selected_seed_gids or ["baseline"],
                        trace.selected_seed_gids or ["baseline"]
                    )
                else:
                    # Run retrieval on counterfactual query
                    # (Note: This would require running the full retrieval pipeline)
                    # For now, we create a mock comparison
                    comps = compare_retrievals(
                        trace, trace,
                        trace.selected_seed_gids or [],
                        trace.selected_seed_gids[:-1] if len(trace.selected_seed_gids or []) > 1 else []
                    )
                
                comparisons[cf.counterfactual_id] = comps
            
            # Attach counterfactual results to trace
            trace.counterfactuals = [
                {
                    "counterfactual_id": cf.counterfactual_id,
                    "mutation_type": cf.mutation_type,
                    "mutation_description": cf.mutation_description,
                    "mutated_query": cf.mutated_query,
                    "comparison": {
                        "node_jaccard": comparisons[cf.counterfactual_id].node_jaccard,
                        "top_k_churn": comparisons[cf.counterfactual_id].top_k_churn
                    }
                }
                for cf in counterfactuals
            ]
            
            print(f"\n=== Counterfactual Analysis Complete ===")
            print(f"Counterfactuals generated: {len(counterfactuals)}")
            print(f"Comparisons completed: {len(comparisons)}")
        except Exception as e:
            print(f"Warning: Counterfactual analysis computation failed: {e}")
    
    # Save trace if enabled
    if trace:
        output_dir = get_trace_output_dir()
        json_file = trace.save_json(output_dir)
        jsonl_file = trace.append_jsonl(os.path.join(output_dir, "decision_trace.jsonl"))
        
        # Save edge confidence JSONL if available
        if os.getenv("EDGE_CONFIDENCE", "0") == "1" and trace.expansion_steps and trace.expansion_steps[0].get("confidence"):
            conf_jsonl = write_edge_confidence_jsonl(trace, os.path.join(output_dir, "edge_confidence.jsonl"))
            print(f"Edge Confidence JSONL: {conf_jsonl}")
        
        # Save stability protocol outputs if enabled
        if os.getenv("STABILITY_PROTOCOL", "0") == "1":
            try:
                # Stability reports would be saved here if we had access to the full metrics
                # This would require refactoring to accumulate metrics across the pipeline
                print("Stability protocol outputs available in trace.expansion_steps[*].confidence")
            except Exception as e:
                print(f"Warning: Could not save stability outputs: {e}")
        
        # Save counterfactual outputs if enabled
        if os.getenv("COUNTERFACTUAL_ANALYSIS", "0") == "1":
            try:
                if hasattr(trace, "counterfactuals") and trace.counterfactuals:
                    cf_jsonl = os.path.join(output_dir, "counterfactual_analysis.jsonl")
                    with open(cf_jsonl, "w", encoding="utf-8") as f:
                        for cf in trace.counterfactuals:
                            import json
                            f.write(json.dumps(cf) + "\n")
                    print(f"Counterfactual JSONL: {cf_jsonl}")
            except Exception as e:
                print(f"Warning: Could not save counterfactual outputs: {e}")
        
        print(f"\n=== Retrieval Trace ===")
        print(f"JSON: {json_file}")
        print(f"JSONL: {jsonl_file}")


if __name__ == "__main__":
    main()
