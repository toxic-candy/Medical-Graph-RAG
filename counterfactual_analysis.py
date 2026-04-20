import json
import os
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class Counterfactual:
    """Represents a single counterfactual variant of the original query."""
    counterfactual_id: str
    mutation_type: str  # "remove_symptom", "modify_lab", "negate_evidence"
    mutation_description: str
    mutated_query: str
    mutation_metadata: Dict[str, Any]


@dataclass
class ComparisonResult:
    """Comparison of original vs counterfactual retrieval results."""
    baseline_gids: List[str]
    counterfactual_gids: List[str]
    node_jaccard: float
    node_jaccard_complement: float  # Nodes lost in counterfactual
    edge_rank_delta: Dict[str, float]  # Per-edge rank displacement
    path_score_delta: List[float]  # Path scores changed
    sensitive_edges: List[str]  # Edges that disappeared/reranked significantly
    top_k_churn: float  # Fraction of top-k nodes that changed


class QueryPerturbationEngine:
    """Generates controlled perturbations of clinical queries."""
    
    @staticmethod
    def remove_symptom(query: str, symptom_list: Optional[List[str]] = None) -> Counterfactual:
        """Remove a symptom token from query.
        
        Args:
            query: Original query text
            symptom_list: List of symptoms to try removing (auto-detected if None)
            
        Returns:
            Counterfactual with symptom removed
        """
        # Common symptoms to remove
        default_symptoms = [
            "fever", "cough", "dyspnea", "chest pain", "fatigue",
            "headache", "nausea", "vomiting", "diarrhea", "rash",
            "tachycardia", "hypoxia", "hypotension", "confusion"
        ]
        
        targets = symptom_list or default_symptoms
        
        for symptom in targets:
            pattern = r'\b' + re.escape(symptom) + r'\b'
            if re.search(pattern, query, re.IGNORECASE):
                mutated = re.sub(pattern, '', query, flags=re.IGNORECASE).strip()
                mutated = re.sub(r'\s+', ' ', mutated)
                
                return Counterfactual(
                    counterfactual_id=f"cf_remove_{symptom}_1",
                    mutation_type="remove_symptom",
                    mutation_description=f"Removed symptom: '{symptom}'",
                    mutated_query=mutated,
                    mutation_metadata={"original_symptom": symptom}
                )
        
        # Fallback: remove first word (naive)
        words = query.split()
        if len(words) > 2:
            mutated = " ".join(words[1:])
            return Counterfactual(
                counterfactual_id="cf_remove_word_1",
                mutation_type="remove_symptom",
                mutation_description=f"Removed word: '{words[0]}'",
                mutated_query=mutated,
                mutation_metadata={"removed_word": words[0]}
            )
        
        # Fallback: return original if no removal possible
        return Counterfactual(
            counterfactual_id="cf_identical_1",
            mutation_type="no_mutation",
            mutation_description="Could not identify removable symptom",
            mutated_query=query,
            mutation_metadata={}
        )
    
    @staticmethod
    def modify_lab_value(query: str, lab_name: str = "WBC", delta: float = 0.5) -> Counterfactual:
        """Modify a lab value by a clinical delta.
        
        Args:
            query: Original query text
            lab_name: Lab value name (e.g., "WBC", "glucose")
            delta: Multiplicative delta (1.5 = 50% increase, 0.5 = 50% decrease)
            
        Returns:
            Counterfactual with lab value modified
        """
        pattern = r'(?i)' + re.escape(lab_name) + r'[\s:\-]*(\d+(?:\.\d+)?)'
        match = re.search(pattern, query)
        
        if match:
            original_val = float(match.group(1))
            modified_val = original_val * delta
            mutated = re.sub(pattern, f"{lab_name}: {modified_val:.1f}", query, flags=re.IGNORECASE)
            
            return Counterfactual(
                counterfactual_id=f"cf_modify_{lab_name}_{delta}_1",
                mutation_type="modify_lab",
                mutation_description=f"Modified {lab_name}: {original_val:.1f} → {modified_val:.1f}",
                mutated_query=mutated,
                mutation_metadata={
                    "lab_name": lab_name,
                    "original_value": original_val,
                    "modified_value": modified_val,
                    "delta_factor": delta
                }
            )
        
        # Fallback: append lab value
        mutated = f"{query} {lab_name}: {100 * delta:.1f}"
        return Counterfactual(
            counterfactual_id=f"cf_add_{lab_name}_1",
            mutation_type="modify_lab",
            mutation_description=f"Added {lab_name}: {100 * delta:.1f}",
            mutated_query=mutated,
            mutation_metadata={"lab_name": lab_name, "value": 100 * delta}
        )
    
    @staticmethod
    def negate_evidence(query: str) -> Counterfactual:
        """Negate a key statement in the query (e.g., "no fever" instead of "fever").
        
        Args:
            query: Original query text
            
        Returns:
            Counterfactual with evidence negated
        """
        negations = [
            ("fever", "no fever"),
            ("chest pain", "no chest pain"),
            ("hypoxia", "no hypoxia"),
            ("cough", "no cough"),
        ]
        
        for positive, negative in negations:
            if re.search(r'\b' + positive + r'\b', query, re.IGNORECASE):
                # Check if already negated
                if re.search(r'\b(no|negative)\s+' + positive + r'\b', query, re.IGNORECASE):
                    continue
                
                mutated = re.sub(
                    r'\b' + positive + r'\b',
                    'no ' + positive,
                    query,
                    flags=re.IGNORECASE,
                    count=1
                )
                
                return Counterfactual(
                    counterfactual_id=f"cf_negate_{positive.replace(' ', '_')}_1",
                    mutation_type="negate_evidence",
                    mutation_description=f"Negated: '{positive}' → 'no {positive}'",
                    mutated_query=mutated,
                    mutation_metadata={"negated_claim": positive}
                )
        
        # Fallback: add "not" prefix
        mutated = "not " + query
        return Counterfactual(
            counterfactual_id="cf_negate_all_1",
            mutation_type="negate_evidence",
            mutation_description="Prepended 'not' to entire query",
            mutated_query=mutated,
            mutation_metadata={"strategy": "prefix_negation"}
        )


def generate_counterfactuals(
    original_query: str,
    num_mutations: int = 3
) -> List[Counterfactual]:
    """
    Generate a set of diverse counterfactual queries from the original.
    
    Args:
        original_query: Original clinical query
        num_mutations: Number of counterfactuals to generate (default 3)
        
    Returns:
        List of Counterfactual objects
    """
    engine = QueryPerturbationEngine()
    counterfactuals = []
    
    # Always include original as baseline
    original = Counterfactual(
        counterfactual_id="cf_baseline_0",
        mutation_type="baseline",
        mutation_description="Original query (baseline)",
        mutated_query=original_query,
        mutation_metadata={}
    )
    counterfactuals.append(original)
    
    # Generate perturbations
    mutations = [
        lambda: engine.remove_symptom(original_query),
        lambda: engine.modify_lab_value(original_query, "WBC", 1.5),
        lambda: engine.modify_lab_value(original_query, "glucose", 0.5),
        lambda: engine.negate_evidence(original_query),
    ]
    
    for i, mutation_fn in enumerate(mutations[:num_mutations]):
        try:
            cf = mutation_fn()
            counterfactuals.append(cf)
        except Exception as e:
            print(f"  Warning: Counterfactual {i} generation failed: {str(e)[:100]}")
    
    return counterfactuals


def compare_retrievals(
    baseline_trace: Any,
    counterfactual_trace: Any,
    baseline_gids: List[str],
    counterfactual_gids: List[str]
) -> ComparisonResult:
    """
    Compare baseline vs counterfactual retrieval outcomes.
    
    Args:
        baseline_trace: RetrievalTrace from original query
        counterfactual_trace: RetrievalTrace from counterfactual query
        baseline_gids: Seed nodes from baseline
        counterfactual_gids: Seed nodes from counterfactual
        
    Returns:
        ComparisonResult with overlap, rank deltas, sensitivity metrics
    """
    # Node-level comparison
    baseline_set = set(baseline_gids)
    counterfactual_set = set(counterfactual_gids)
    
    intersection = baseline_set & counterfactual_set
    union = baseline_set | counterfactual_set
    
    node_jaccard = float(len(intersection)) / float(max(1, len(union)))
    
    # Nodes lost in counterfactual
    lost_nodes = baseline_set - counterfactual_set
    node_jaccard_complement = float(len(lost_nodes)) / float(max(1, len(baseline_set)))
    
    # Edge-level comparison (rank displacement)
    baseline_edges = {step.get("path_id"): step for step in baseline_trace.expansion_steps}
    counterfactual_edges = {step.get("path_id"): step for step in counterfactual_trace.expansion_steps}
    
    edge_rank_delta = {}
    for edge_id, baseline_step in baseline_edges.items():
        if edge_id in counterfactual_edges:
            cf_step = counterfactual_edges[edge_id]
            baseline_rank = float(baseline_step.get("step_rank", 999))
            cf_rank = float(cf_step.get("step_rank", 999))
            delta = abs(baseline_rank - cf_rank)
            edge_rank_delta[edge_id] = delta
    
    # Path score deltas
    path_score_delta = []
    baseline_paths = {p.get("path_id"): p for p in baseline_trace.selected_paths}
    counterfactual_paths = {p.get("path_id"): p for p in counterfactual_trace.selected_paths}
    
    for path_id, baseline_path in baseline_paths.items():
        if path_id in counterfactual_paths:
            cf_path = counterfactual_paths[path_id]
            baseline_score = float(baseline_path.get("path_score_total", 0.0))
            cf_score = float(cf_path.get("path_score_total", 0.0))
            delta = abs(baseline_score - cf_score)
            path_score_delta.append(delta)
    
    # Sensitive edges: those with high rank displacement
    sensitive_edges = [
        edge_id for edge_id, delta in edge_rank_delta.items() if delta > 2
    ]
    
    # Top-k churn: fraction of top-3 nodes that changed
    baseline_top_k = set(baseline_gids[:3])
    counterfactual_top_k = set(counterfactual_gids[:3])
    top_k_churn = float(len(baseline_top_k - counterfactual_top_k)) / len(baseline_top_k)
    
    return ComparisonResult(
        baseline_gids=baseline_gids,
        counterfactual_gids=counterfactual_gids,
        node_jaccard=round(node_jaccard, 4),
        node_jaccard_complement=round(node_jaccard_complement, 4),
        edge_rank_delta=edge_rank_delta,
        path_score_delta=path_score_delta,
        sensitive_edges=sensitive_edges,
        top_k_churn=round(top_k_churn, 4)
    )


def write_counterfactual_report(
    counterfactuals: List[Counterfactual],
    comparisons: Dict[str, ComparisonResult],
    baseline_trace: Any,
    output_file: str
) -> str:
    """
    Write comprehensive counterfactual analysis report.
    
    Args:
        counterfactuals: List of Counterfactual objects generated
        comparisons: Dict mapping counterfactual_id -> ComparisonResult
        baseline_trace: Original RetrievalTrace
        output_file: Path to output JSON file
        
    Returns:
        Path to file
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    # Identify most sensitive edges/paths
    all_sensitive_edges = set()
    for comp in comparisons.values():
        all_sensitive_edges.update(comp.sensitive_edges)
    
    sensitivity_ranking = {}
    for edge_id in all_sensitive_edges:
        appearances = sum(1 for comp in comparisons.values() if edge_id in comp.sensitive_edges)
        sensitivity_ranking[edge_id] = appearances
    
    # Sort by sensitivity
    top_sensitive = sorted(sensitivity_ranking.items(), key=lambda x: x[1], reverse=True)[:5]
    
    report = {
        "query_id": baseline_trace.query_id,
        "phase": 5,
        "title": "Counterfactual Retrieval Analysis",
        "baseline_query": baseline_trace.query_text,
        "counterfactuals": [
            {
                "counterfactual_id": cf.counterfactual_id,
                "mutation_type": cf.mutation_type,
                "mutation_description": cf.mutation_description,
                "mutated_query": cf.mutated_query,
                "mutation_metadata": cf.mutation_metadata
            }
            for cf in counterfactuals
        ],
        "comparisons": {
            cf_id: {
                "node_jaccard": comp.node_jaccard,
                "node_jaccard_complement": comp.node_jaccard_complement,
                "top_k_churn": comp.top_k_churn,
                "num_sensitive_edges": len(comp.sensitive_edges),
                "sensitive_edges": comp.sensitive_edges,
                "mean_edge_rank_delta": round(
                    float(np.mean(list(comp.edge_rank_delta.values()))) 
                    if comp.edge_rank_delta else 0.0, 4
                ),
                "mean_path_score_delta": round(
                    float(np.mean(comp.path_score_delta)) 
                    if comp.path_score_delta else 0.0, 4
                )
            }
            for cf_id, comp in comparisons.items()
        },
        "sensitivity_ranking": [
            {"edge_id": edge_id, "sensitivity_count": count}
            for edge_id, count in top_sensitive
        ],
        "statistics": {
            "num_counterfactuals": len(counterfactuals),
            "num_comparisons": len(comparisons),
            "mean_node_jaccard": round(
                float(np.mean([c.node_jaccard for c in comparisons.values()])) 
                if comparisons else 0.0, 4
            ),
            "mean_top_k_churn": round(
                float(np.mean([c.top_k_churn for c in comparisons.values()]))
                if comparisons else 0.0, 4
            ),
            "total_sensitive_edges": len(all_sensitive_edges)
        }
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    return output_file


def write_counterfactual_jsonl(
    counterfactuals: List[Counterfactual],
    comparisons: Dict[str, ComparisonResult],
    baseline_trace: Any,
    output_file: str
) -> str:
    """
    Write counterfactual analysis to JSONL (one record per counterfactual).
    
    Args:
        counterfactuals: List of Counterfactual objects
        comparisons: Dict mapping counterfactual_id -> ComparisonResult
        baseline_trace: Original RetrievalTrace
        output_file: Path to output JSONL file
        
    Returns:
        Path to file
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for cf in counterfactuals:
            comp = comparisons.get(cf.counterfactual_id)
            
            record = {
                "query_id": baseline_trace.query_id,
                "counterfactual_id": cf.counterfactual_id,
                "mutation_type": cf.mutation_type,
                "mutation_description": cf.mutation_description,
                "mutated_query": cf.mutated_query,
                "mutation_metadata": cf.mutation_metadata
            }
            
            if comp:
                record.update({
                    "node_jaccard": comp.node_jaccard,
                    "node_jaccard_complement": comp.node_jaccard_complement,
                    "top_k_churn": comp.top_k_churn,
                    "num_sensitive_edges": len(comp.sensitive_edges),
                    "sensitive_edges": comp.sensitive_edges
                })
            
            f.write(json.dumps(record) + "\n")
    
    return output_file


if __name__ == "__main__":
    print("✓ Counterfactual Analysis module loaded successfully.")
    print("  Classes: QueryPerturbationEngine, Counterfactual, ComparisonResult")
    print("  Functions: generate_counterfactuals, compare_retrievals, write_counterfactual_report,")
    print("            write_counterfactual_jsonl")
