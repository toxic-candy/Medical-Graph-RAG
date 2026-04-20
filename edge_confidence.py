import math
import os
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter

# Default configuration
DEFAULT_ALPHA = 0.6     #(similarity strength)
DEFAULT_BETA = 0.3       #(graph support)
DEFAULT_GAMMA = 0.1      #(retrieval stability)
DEFAULT_STABILITY_RUNS = 10  


def _get_match_similarity(trace, gid: str) -> float:
    """
    Lookup similarity_norm in trace.matched_nodes for a given gid.
    
    Args:
        trace: RetrievalTrace object
        gid: Node gid to lookup
        
    Returns:
        Float in [0, 1]; 0 if not found
    """
    for m in trace.matched_nodes:
        if m.get("node_id") == gid:
            return float(m.get("similarity_norm", 0.0))
    return 0.0


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a, b: Lists or arrays of floats
        
    Returns:
        Float cosine similarity in [-1, 1]; clipped to [0, 1]
    """
    try:
        import numpy as np
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
    except Exception:
        return 0.0


def compute_S_e(n4j, edge: Dict[str, Any], trace: Any) -> float:
    """
    Compute similarity strength (S_e) for an edge.
    
    S_e represents the semantic similarity of the target node to the query.
    We prefer using matched_nodes similarity_norm (already computed during ranking).
    
    Args:
        n4j: Neo4j graph connection
        edge: Edge record from trace.expansion_steps
        trace: RetrievalTrace object
        
    Returns:
        Float in [0, 1]
    """
    to_gid = edge.get("to_node")
    if not to_gid:
        return 0.0
    
    # First, try to find similarity from matched_nodes
    s = _get_match_similarity(trace, to_gid)
    if s > 0:
        return s
    
    # Fallback: compute embedding cosine (slower, but comprehensive)
    try:
        from utils import get_embedding
        from retrieve import _summary_to_text
        
        # Query the to_node content
        query = f"MATCH (n {{gid: '{to_gid}'}}) RETURN n.content AS content LIMIT 1"
        rows = n4j.query(query)
        if not rows:
            return 0.0
        
        content = rows[0].get("content")
        if not content:
            return 0.0
        
        # Compute embeddings
        q_text = trace.query_summary or trace.query_text
        q_emb = get_embedding(q_text)
        content_text = _summary_to_text(content)
        t_emb = get_embedding(content_text)
        
        sim = _cosine_similarity(q_emb, t_emb)
        return max(0.0, min(1.0, float(sim)))
    except Exception:
        return 0.0


def compute_G_e(n4j, edge: Dict[str, Any]) -> float:
    """
    Compute graph support (G_e) for an edge.
    
    G_e represents the structural support for the edge in the graph:
    - Support count: how many paths use this edge
    - Degree normalization: penalize high-degree hubs to avoid bias
    
    Args:
        n4j: Neo4j graph connection
        edge: Edge record from trace.expansion_steps
        
    Returns:
        Float in [0, 1]
    """
    from_gid = edge.get("from_node")
    to_gid = edge.get("to_node")
    
    if not from_gid or not to_gid:
        return 0.0
    
    try:
        # Count incoming edges to target node (support metric)
        q_support = f"MATCH (n {{gid: '{to_gid}'}})<-[r:REFERENCE]-() RETURN count(r) AS cnt"
        rows_sup = n4j.query(q_support)
        support_count = float(rows_sup[0].get("cnt", 0)) if rows_sup else 0.0
        
        # Compute degree normalization penalty
        # degree = total neighbors (both incoming and outgoing)
        q_deg_from = f"MATCH (a {{gid: '{from_gid}'}}) RETURN size((a)--()) AS deg"
        q_deg_to = f"MATCH (b {{gid: '{to_gid}'}}) RETURN size((b)--()) AS deg"
        
        deg_from = 1.0
        deg_to = 1.0
        try:
            rows = n4j.query(q_deg_from)
            if rows:
                deg_from = float(rows[0].get("deg", 1))
        except Exception:
            pass
        try:
            rows = n4j.query(q_deg_to)
            if rows:
                deg_to = float(rows[0].get("deg", 1))
        except Exception:
            pass
        
        # Degree-based penalty (log scale to avoid over-penalizing)
        avg_degree = (deg_from + deg_to) / 2.0
        degree_penalty = 1.0 / math.log(2.0 + avg_degree)
        
        # Normalize support to [0, 1] via sigmoid-like scaling
        g_raw = support_count / (1.0 + support_count)
        g = float(min(1.0, g_raw * degree_penalty))
        
    except Exception:
        g = 0.0
    
    return max(0.0, min(1.0, g))


def compute_R_e(n4j, edge: Dict[str, Any], trace: Any, runs: int = DEFAULT_STABILITY_RUNS) -> float:
    """
    Compute retrieval stability (R_e) for an edge.
    
    R_e represents the consistency of the edge across repeated retrieval runs.
    A simple approximation: frequency of edge in `runs` repeated seed selections + expansions.
    
    Args:
        n4j: Neo4j graph connection
        edge: Edge record from trace.expansion_steps
        trace: RetrievalTrace object
        runs: Number of repeated retrievals to sample (default 10)
        
    Returns:
        Float in [0, 1]
    """
    if runs <= 1:
        return 1.0  # No repeated runs == perfect stability (trivial)
    
    from_gid = edge.get("from_node")
    to_gid = edge.get("to_node")
    
    if not from_gid or not to_gid:
        return 0.0
    
    edge_key = (from_gid, to_gid)
    seen_count = 0
    
    try:
        from retrieve import select_top_gids
        
        # Rerun seed selection and 1-hop expansions
        for _ in range(runs):
            try:
                # Re-select seed nodes
                q_summary = [trace.query_summary or trace.query_text]
                seeds = select_top_gids(n4j, q_summary, top_k=max(1, len(trace.selected_seed_gids) or 2))
                
                if not seeds:
                    continue
                
                # Check if edge is in current run
                # (This is approximate; a full re-run would recompute embeddings, which is costly)
                # For stability, we check: is from_gid still a seed, and does to_gid connect to it?
                for seed in seeds:
                    if seed == from_gid:
                        # Check if to_gid is a 1-hop neighbor of seed
                        q_neighbors = f"MATCH (n {{gid: '{seed}'}})-[:REFERENCE]->(m {{gid: '{to_gid}'}}) RETURN m.gid LIMIT 1"
                        rows = n4j.query(q_neighbors)
                        if rows:
                            seen_count += 1
                            break
            except Exception:
                pass
    except Exception:
        pass
    
    return float(seen_count) / float(max(1, runs))


def compute_edge_confidences(
    n4j,
    trace: Any,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    stability_runs: int = DEFAULT_STABILITY_RUNS
) -> Any:
    """
    Attach confidence scores to all edges and paths in a trace.
    
    This function mutates trace.expansion_steps and trace.selected_paths in-place,
    adding 'confidence' and 'path_confidence' fields.
    
    Args:
        n4j: Neo4j graph connection
        trace: RetrievalTrace object (will be mutated)
        alpha: Weight for S_e in composite confidence (default 0.6)
        beta: Weight for G_e in composite confidence (default 0.3)
        gamma: Weight for R_e in composite confidence (default 0.1)
        stability_runs: Number of repeated runs for R_e computation (default 10)
        
    Returns:
        trace (mutated)
    """
    assert abs((alpha + beta + gamma) - 1.0) < 0.01, "alpha + beta + gamma must sum to 1.0"
    
    # Compute confidence for each expansion step (edge)
    edge_conf_map = {}  # path_id -> C_e
    
    for step in trace.expansion_steps:
        try:
            S_e = compute_S_e(n4j, step, trace)
            G_e = compute_G_e(n4j, step)
            R_e = compute_R_e(n4j, step, trace, runs=stability_runs)
            
            # Composite confidence
            C_e = float(alpha * S_e + beta * G_e + gamma * R_e)
            
            # Attach to step
            step["confidence"] = {
                "S_e": round(float(S_e), 6),
                "G_e": round(float(G_e), 6),
                "R_e": round(float(R_e), 6),
                "C_e": round(float(C_e), 6),
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma
            }
            
            # Store in map for path aggregation
            path_id = step.get("path_id")
            if path_id:
                edge_conf_map[path_id] = float(C_e)
        
        except Exception as e:
            # Silently fall back to zero confidence on error
            step["confidence"] = {
                "S_e": 0.0,
                "G_e": 0.0,
                "R_e": 0.0,
                "C_e": 0.0,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "error": str(e)[:100]
            }
    
    # Compute path-level confidence by aggregating edge confidences
    for path in trace.selected_paths:
        try:
            edge_ids = path.get("edges", [])
            
            if not edge_ids:
                # Seed path: use seed node's matched similarity
                seed_gid = path.get("nodes", [None])[0]
                seed_sim = _get_match_similarity(trace, seed_gid) if seed_gid else 0.0
                path["path_confidence"] = round(float(seed_sim), 6)
                path["path_confidence_components"] = {
                    "type": "seed",
                    "seed_similarity": round(float(seed_sim), 6)
                }
            else:
                # Expanded path: aggregate edge confidences
                edge_scores = [edge_conf_map.get(eid, 0.0) for eid in edge_ids if eid]
                
                if edge_scores:
                    mean_edge_conf = float(sum(edge_scores) / len(edge_scores))
                    # Apply path length penalty
                    plp = path.get("score_components", {}).get("path_length_penalty", 1.0)
                    path_conf = mean_edge_conf * plp
                else:
                    mean_edge_conf = 0.0
                    path_conf = 0.0
                    plp = 1.0
                
                path["path_confidence"] = round(float(path_conf), 6)
                path["path_confidence_components"] = {
                    "type": "expanded",
                    "mean_edge_confidence": round(mean_edge_conf, 6),
                    "path_length_penalty": round(float(plp), 6),
                    "num_edges": len(edge_ids)
                }
        
        except Exception as e:
            path["path_confidence"] = 0.0
            path["path_confidence_components"] = {"error": str(e)[:100]}
    
    return trace


def write_edge_confidence_jsonl(trace: Any, output_file: str) -> str:
    """
    Write edge confidence records to JSONL file.
    
    Args:
        trace: RetrievalTrace object with confidence attached
        output_file: Path to output JSONL file
        
    Returns:
        Path to file
    """
    import json
    import os
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    with open(output_file, "a", encoding="utf-8") as f:
        for step in trace.expansion_steps:
            conf = step.get("confidence", {})
            record = {
                "query_id": trace.query_id,
                "edge_id": step.get("path_id"),
                "from_node": step.get("from_node"),
                "to_node": step.get("to_node"),
                "S_e": conf.get("S_e", 0.0),
                "G_e": conf.get("G_e", 0.0),
                "R_e": conf.get("R_e", 0.0),
                "C_e": conf.get("C_e", 0.0),
                "rank": step.get("step_rank")
            }
            f.write(json.dumps(record) + "\n")
    
    return output_file


if __name__ == "__main__":
    print("edge_confidence module loaded successfully.")
    print(f"Defaults: alpha={DEFAULT_ALPHA}, beta={DEFAULT_BETA}, gamma={DEFAULT_GAMMA}, stability_runs={DEFAULT_STABILITY_RUNS}")
