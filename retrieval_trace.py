import json
import os
import uuid
import time
from typing import Any, Dict, List, Optional
from datetime import datetime


class RetrievalTrace:
    """
    Represents a single retrieval run with full decision traces.
    
    Attributes:
        query_id: Unique identifier for this trace
        query_text: Original user question
        query_summary: Processed/summarized query text
        timestamp: Unix timestamp when trace started
        config: Configuration snapshot (models, parameters, flags)
        matched_nodes: List of nodes matched by retrieval ranking
        expansion_steps: List of path expansion decisions
        evidence_items: List of collected evidence lines with sources
        answer_text: Final generated answer
        answer_config: Configuration for answer synthesis
        statistics: Summary statistics
    """
    
    def __init__(self, question: str):
        self.query_id = str(uuid.uuid4())
        self.query_text = question
        self.query_summary = None
        self.timestamp = time.time()
        self.datetime_iso = datetime.utcnow().isoformat() + "Z"
        
        self.config = {}
        self.matched_nodes = []
        self.expansion_steps = []
        self.evidence_items = []
        self.evidence_by_gid = {}
        self.answer_text = None
        self.answer_config = {}
        self.selected_seed_gids = []
        self.expanded_gids = []
        
        self.statistics = {
            "matched_node_count": 0,
            "expansion_step_count": 0,
            "evidence_item_count": 0,
            "unique_gids_used": 0,
        }
    
    def add_config(self, config_dict: Dict[str, Any]):
        self.config.update(config_dict)
    
    def add_matched_node(self, node_record: Dict[str, Any]):
        self.matched_nodes.append(node_record)
        self.statistics["matched_node_count"] = len(self.matched_nodes)
    
    def add_expansion_step(self, step_record: Dict[str, Any]):
        self.expansion_steps.append(step_record)
        self.statistics["expansion_step_count"] = len(self.expansion_steps)
    
    def add_evidence_item(self, evidence_record: Dict[str, Any]):
        self.evidence_items.append(evidence_record)
        gid = evidence_record.get("gid")
        if gid:
            self.evidence_by_gid.setdefault(gid, []).append(evidence_record)
        self.statistics["evidence_item_count"] = len(self.evidence_items)
        self.statistics["unique_gids_used"] = len(self.evidence_by_gid)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "query_summary": self.query_summary,
            "timestamp": self.timestamp,
            "datetime_iso": self.datetime_iso,
            "config": self.config,
            "matched_nodes": self.matched_nodes,
            "expansion_steps": self.expansion_steps,
            "selected_seed_gids": self.selected_seed_gids,
            "expanded_gids": self.expanded_gids,
            "evidence_items": self.evidence_items,
            "answer_text": self.answer_text,
            "answer_config": self.answer_config,
            "statistics": self.statistics,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict())
    
    def save_json(self, output_dir: str = "./retrieval_traces") -> str:
        """
        Save trace to JSON file.
        
        Args:
            output_dir: Directory to save traces
            
        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/trace_{self.query_id}.json"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return filename
    
    def append_jsonl(self, output_file: str = "./retrieval_traces/decision_trace.jsonl") -> str:
        """
        Append trace to JSONL file.
        
        Args:
            output_file: Path to JSONL file
            
        Returns:
            Path to saved file
        """
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(self.to_jsonl_line() + "\n")
        return output_file
    
    def summary_markdown(self) -> str:
        lines = [
            f"# Retrieval Trace Summary",
            f"",
            f"**Query ID:** `{self.query_id}`",
            f"**Timestamp:** {self.datetime_iso}",
            f"",
            f"## Query",
            f"```",
            f"{self.query_text}",
            f"```",
            f"",
            f"## Statistics",
            f"- Matched nodes: {self.statistics['matched_node_count']}",
            f"- Expansion steps: {self.statistics['expansion_step_count']}",
            f"- Evidence items: {self.statistics['evidence_item_count']}",
            f"- Unique GIDs used: {self.statistics['unique_gids_used']}",
            f"",
            f"## Config",
            f"```json",
            f"{json.dumps(self.config, indent=2)}",
            f"```",
            f"",
        ]
        
        if self.matched_nodes:
            lines.extend([
                f"## Top Matched Nodes",
                f"",
            ])
            for node in self.matched_nodes[:5]:
                sim = node.get("similarity_norm", 0)
                lines.append(f"- **{node.get('node_id')}** (type: {node.get('node_type')}, sim: {sim:.3f})")
            lines.append("")
        
        if self.selected_seed_gids:
            lines.extend([
                f"## Selected Seed GIDs",
                f"- {', '.join(self.selected_seed_gids)}",
                f"",
            ])
        
        if self.answer_text:
            lines.extend([
                f"## Answer",
                f"```",
                f"{self.answer_text}",
                f"```",
                f"",
            ])
        
        return "\n".join(lines)


def is_tracing_enabled() -> bool:
    return os.getenv("TRACE_RETRIEVAL", "0") == "1"


def get_trace_output_dir() -> str:
    return os.getenv("TRACE_OUTPUT_DIR", "./retrieval_traces")


def get_trace_verbosity() -> str:
    return os.getenv("TRACE_VERBOSITY", "full")
