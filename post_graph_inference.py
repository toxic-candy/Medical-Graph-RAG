import argparse
import os
from urllib.parse import urlparse

from camel.storages import Neo4jGraph

from simple_neo4j_graph import SimpleNeo4jGraph
from retrieve import select_top_gids
from summerize import process_chunks
from utils import call_llm, link_context, ret_context


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

    n4j = _connect_neo4j(
        url=_normalize_neo4j_url(args.neo4j_url),
        username=args.neo4j_username,
        password=args.neo4j_password,
    )

    q_summary = _question_summary(question)
    seed_gids = select_top_gids(n4j, q_summary, top_k=max(1, args.top_k))

    expanded_gids = list(seed_gids)
    for gid in seed_gids:
        expanded_gids.extend(_neighbor_gids(n4j, gid, max_hops=args.max_hops, max_items=20))

    # Preserve order while deduplicating.
    ordered_unique_gids = list(dict.fromkeys([g for g in expanded_gids if g]))

    evidence = _collect_evidence(n4j, ordered_unique_gids, max_evidence=max(10, args.max_evidence))
    if not evidence:
        print("No evidence retrieved from graph. Check graph construction and REFERENCE links.")
        return

    answer = _answer_with_citations(question, evidence)

    print("=== Selected GIDs ===")
    for gid in ordered_unique_gids:
        print(gid)

    print("\n=== Evidence Count ===")
    print(len(evidence))

    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
