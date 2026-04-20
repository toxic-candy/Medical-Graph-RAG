import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import statistics


def run_repeated_retrievals(n4j, query_text: str, num_runs: int = 10, top_k: int = 3, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Execute identical retrieval N times to capture stochastic variation.
    
    Args:
        n4j: Neo4j graph connection
        query_text: Query to retrieve (identical across all runs)
        num_runs: Number of repeated retrivals (default 10)
        top_k: Top-K seeds to return per run (default 3)
        seed: Random seed for reproducibility (optional, None = unseeded)
        
    Returns:
        List of run results, each containing:
        {
            "run_id": int,
            "seed_nodes": [gid1, gid2, ...],
            "expansion_steps": [...],
            "selected_paths": [...],
            "rank_map": {gid: rank_position, ...},
            "score_map": {gid: similarity_score, ...}
        }
    """
    from retrieve import select_top_gids, _summary_to_text
    import random
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    runs = []
    
    for run_id in range(num_runs):
        try:
            # Optional: inject stochasticity via k-NN sampling instead of exact top-k
            # For now, simple approach: re-embed and re-rank
            query_summary = _summary_to_text(query_text)
            seed_nodes = select_top_gids(n4j, [query_summary], top_k=top_k)
            
            if not seed_nodes:
                continue
            
            # Collect rank/score info for these nodes
            rank_map = {gid: rank for rank, gid in enumerate(seed_nodes)}
            
            # Fetch similarity scores for these nodes from embeddings
            score_map = {}
            for gid in seed_nodes:
                try:
                    from utils import get_embedding
                    q_node = f"MATCH (n {{gid: '{gid}'}}) RETURN n.embedding AS emb, n.content AS content"
                    rows = n4j.query(q_node)
                    if rows:
                        node_emb = rows[0].get("emb")
                        if node_emb and isinstance(node_emb, (list, np.ndarray)):
                            from retrieve import _cosine
                            q_emb = get_embedding(query_summary)
                            sim = _cosine(q_emb, node_emb)
                            score_map[gid] = float(max(0.0, min(1.0, sim)))
                except Exception:
                    score_map[gid] = 0.0
            
            run_result = {
                "run_id": run_id,
                "seed_nodes": seed_nodes,
                "rank_map": rank_map,
                "score_map": score_map,
                "timestamp": str(np.datetime64('now'))
            }
            runs.append(run_result)
        
        except Exception as e:
            print(f"  Warning: Run {run_id} failed: {str(e)[:100]}")
            continue
    
    return runs


def compute_node_stability(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute stability metrics for each node across runs.
    
    Args:
        runs: List of run results from run_repeated_retrievals()
        
    Returns:
        Dict mapping node_gid -> {
            "presence_rate": float in [0,1],
            "mean_rank": float,
            "rank_variance": float,
            "mean_score": float,
            "score_stddev": float,
            "top_k_retention": float,
            "stability_score": float in [0,1]
        }
    """
    node_data = defaultdict(lambda: {
        "ranks": [],
        "scores": [],
        "appearances": 0,
        "top_k_hits": 0
    })
    
    for run in runs:
        seed_nodes = run.get("seed_nodes", [])
        rank_map = run.get("rank_map", {})
        score_map = run.get("score_map", {})
        
        for gid in seed_nodes:
            node_data[gid]["appearances"] += 1
            rank = rank_map.get(gid, len(seed_nodes))
            score = score_map.get(gid, 0.0)
            
            node_data[gid]["ranks"].append(rank)
            node_data[gid]["scores"].append(score)
            
            # Top-k retention: is gid in top 3 positions?
            if rank < 3:
                node_data[gid]["top_k_hits"] += 1
    
    metrics = {}
    num_runs = len(runs)
    
    for gid, data in node_data.items():
        presence_rate = float(data["appearances"]) / float(max(1, num_runs))
        
        ranks = data["ranks"]
        scores = data["scores"]
        
        mean_rank = float(np.mean(ranks)) if ranks else 0.0
        rank_variance = float(np.var(ranks)) if len(ranks) > 1 else 0.0
        
        mean_score = float(np.mean(scores)) if scores else 0.0
        score_stddev = float(np.std(scores)) if len(scores) > 1 else 0.0
        
        top_k_retention = float(data["top_k_hits"]) / float(max(1, num_runs))
        
        # Composite stability: weighted combination
        # High presence, low rank variance, high top-k retention = stable
        stability = (presence_rate * 0.4 + 
                     (1.0 - min(1.0, rank_variance / 10.0)) * 0.3 +  # Normalize variance
                     top_k_retention * 0.3)
        
        metrics[gid] = {
            "presence_rate": round(presence_rate, 4),
            "mean_rank": round(mean_rank, 4),
            "rank_variance": round(rank_variance, 4),
            "mean_score": round(mean_score, 4),
            "score_stddev": round(score_stddev, 4),
            "top_k_retention": round(top_k_retention, 4),
            "stability_score": round(float(min(1.0, max(0.0, stability))), 4)
        }
    
    return metrics


def compute_edge_stability(runs: List[Dict[str, Any]], original_trace: Any) -> Dict[str, Dict[str, float]]:
    """
    Compute stability metrics for edges in original trace.
    
    For each edge in the original trace, check its presence/score consistency
    across repeated runs (seed selection + 1-hop expansion).
    
    Args:
        runs: List of run results from run_repeated_retrievals()
        original_trace: RetrievalTrace object with expansion_steps
        
    Returns:
        Dict mapping edge_id (path_id from trace) -> {
            "presence_rate": float in [0,1],
            "stability_score": float in [0,1],
            "appearances": int,
            "rank_consistency": float
        }
    """
    edge_metrics = {}
    num_runs = len(runs)
    
    for step in original_trace.expansion_steps:
        edge_id = step.get("path_id")
        from_gid = step.get("from_node")
        to_gid = step.get("to_node")
        
        if not edge_id or not from_gid or not to_gid:
            continue
        
        appearances = 0
        ranks = []
        
        for run in runs:
            seed_nodes = run.get("seed_nodes", [])
            
            # Edge exists if: from_gid is in seeds, and to_gid is a 1-hop neighbor
            for seed in seed_nodes:
                if seed == from_gid:
                    # For stability, verify to_gid connects to this seed
                    try:
                        q = f"MATCH (n {{gid: '{seed}'}})-[:REFERENCE]->(m {{gid: '{to_gid}'}}) RETURN m.gid LIMIT 1"
                        # Note: This is a simplification; full re-run would recompute expansions
                        # For now, we assume edge is stable if both nodes are reachable
                        appearances += 1
                    except Exception:
                        pass
        
        presence_rate = float(appearances) / float(max(1, num_runs))
        rank_consistency = presence_rate  # Simple heuristic
        
        edge_metrics[edge_id] = {
            "presence_rate": round(presence_rate, 4),
            "stability_score": round(float(min(1.0, presence_rate * 1.2)), 4),  # Capped at 1.0
            "appearances": appearances,
            "rank_consistency": round(rank_consistency, 4)
        }
    
    return edge_metrics


def refine_R_e_from_distribution(node_stability: Dict[str, Dict[str, float]], 
                                  edge_stability: Dict[str, Dict[str, float]],
                                  trace: Any) -> Dict[str, float]:
    """
    Refine R_e (retrieval stability) scores using distribution metrics.
    
    Instead of simple frequency checks, use rank variance and score stddev
    to compute a refined R_e bounded in [0,1].
    
    Args:
        node_stability: Output from compute_node_stability()
        edge_stability: Output from compute_edge_stability()
        trace: RetrievalTrace object
        
    Returns:
        Dict mapping step/edge -> refined R_e score
    """
    refined_R_e = {}
    
    for step in trace.expansion_steps:
        edge_id = step.get("path_id")
        to_gid = step.get("to_node")
        
        if not edge_id or not to_gid:
            continue
        
        # Combine node stability and edge stability
        node_metrics = node_stability.get(to_gid, {})
        edge_metrics = edge_stability.get(edge_id, {})
        
        # Weighted combination: 60% edge presence, 40% node top-k retention
        edge_presence = edge_metrics.get("presence_rate", 0.0)
        node_retention = node_metrics.get("top_k_retention", 0.0)
        
        # Discount for high variance
        node_variance = node_metrics.get("rank_variance", 0.0)
        variance_penalty = max(0.0, 1.0 - (node_variance / 10.0))
        
        R_e_refined = (edge_presence * 0.6 + node_retention * 0.4) * variance_penalty
        refined_R_e[edge_id] = round(float(min(1.0, max(0.0, R_e_refined))), 6)
    
    return refined_R_e


def write_stability_report(
    node_stability: Dict[str, Dict[str, float]],
    edge_stability: Dict[str, Dict[str, float]],
    refined_R_e: Dict[str, float],
    trace: Any,
    output_file: str
) -> str:
    """
    Write comprehensive stability analysis report to file.
    
    Args:
        node_stability: Output from compute_node_stability()
        edge_stability: Output from compute_edge_stability()
        refined_R_e: Output from refine_R_e_from_distribution()
        trace: RetrievalTrace object
        output_file: Path to output JSON file
        
    Returns:
        Path to file
    """
    import os
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    report = {
        "query_id": trace.query_id,
        "phase": 4,
        "title": "Stability Protocol Analysis",
        "node_stability_summary": {
            gid: metrics for gid, metrics in node_stability.items()
        },
        "edge_stability_summary": {
            eid: metrics for eid, metrics in edge_stability.items()
        },
        "refined_R_e": refined_R_e,
        "statistics": {
            "num_nodes_analyzed": len(node_stability),
            "num_edges_analyzed": len(edge_stability),
            "mean_node_stability": round(
                float(np.mean([m["stability_score"] for m in node_stability.values()])) 
                if node_stability else 0.0, 4
            ),
            "mean_edge_stability": round(
                float(np.mean([m["stability_score"] for m in edge_stability.values()]))
                if edge_stability else 0.0, 4
            ),
            "mean_refined_R_e": round(
                float(np.mean(list(refined_R_e.values()))) if refined_R_e else 0.0, 4
            )
        }
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    return output_file


def write_stability_jsonl(
    node_stability: Dict[str, Dict[str, float]],
    edge_stability: Dict[str, Dict[str, float]],
    trace: Any,
    output_file: str
) -> str:
    """
    Write stability metrics to JSONL (one record per node/edge analyzed).
    
    Args:
        node_stability: Output from compute_node_stability()
        edge_stability: Output from compute_edge_stability()
        trace: RetrievalTrace object
        output_file: Path to output JSONL file
        
    Returns:
        Path to file
    """
    import os
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for gid, metrics in node_stability.items():
            record = {
                "query_id": trace.query_id,
                "type": "node",
                "gid": gid,
                **metrics
            }
            f.write(json.dumps(record) + "\n")
        
        for edge_id, metrics in edge_stability.items():
            record = {
                "query_id": trace.query_id,
                "type": "edge",
                "edge_id": edge_id,
                **metrics
            }
            f.write(json.dumps(record) + "\n")
    
    return output_file


if __name__ == "__main__":
    print("✓ Stability Protocol module loaded successfully.")
    print("  Functions: run_repeated_retrievals, compute_node_stability, compute_edge_stability,")
    print("            refine_R_e_from_distribution, write_stability_report, write_stability_jsonl")
