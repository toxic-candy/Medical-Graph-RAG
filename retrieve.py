from utils import *
import numpy as np

sys_p = """
Assess the similarity of the two provided summaries and return a rating from these options: 'very similar', 'similar', 'general', 'not similar', 'totally not similar'. Provide only the rating.
"""


def _summary_to_text(summary_value):
    if isinstance(summary_value, list):
        return " ".join([str(x) for x in summary_value if x is not None]).strip()
    if summary_value is None:
        return ""
    return str(summary_value)


def _rating_to_score(rate):
    if "totally not similar" in rate:
        return 0
    if "not similar" in rate:
        return 1
    if "general" in rate:
        return 2
    if "very similar" in rate:
        return 4
    if "similar" in rate:
        return 3
    print("llm returns no relevant rate")
    return -1


def _cosine(a, b):
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return -1.0
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return -1.0
    return float(np.dot(va, vb) / (na * nb))


def _use_remote_retrieval_rater():
    # Keep the default local and deterministic.
    return os.getenv("USE_LLM_RETRIEVAL_RATER", "0") == "1"


def select_top_gids(n4j, sumq, top_k=3):
    rows = n4j.query(
        """
        MATCH (s:Summary)
        RETURN s.content AS content, s.gid AS gid
        """
    )

    query_summary = _summary_to_text(sumq[0] if isinstance(sumq, list) and sumq else sumq)
    query_embedding = get_embedding(query_summary)

    if not rows:
        # No Summary nodes — fall back to sampling entity nodes with embeddings
        # and ranking by cosine similarity to the question.
        entity_rows = n4j.query(
            """
            MATCH (n)
            WHERE n.embedding IS NOT NULL AND n.gid IS NOT NULL AND NOT n:Summary
            RETURN n.gid AS gid, n.embedding AS embedding
            LIMIT 500
            """
        )
        if not entity_rows:
            return []
        scored = []
        for row in entity_rows:
            emb = row.get("embedding")
            gid = row.get("gid")
            if emb and gid:
                scored.append((_cosine(emb, query_embedding), gid))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Deduplicate GIDs while preserving ranking order
        seen = set()
        top = []
        for _, gid in scored:
            if gid not in seen:
                seen.add(gid)
                top.append(gid)
            if len(top) >= max(1, top_k):
                break
        return top

    scored = []
    for row in rows:
        summary_text = _summary_to_text(row.get("content"))
        if not summary_text:
            continue
        if _use_remote_retrieval_rater():
            try:
                rate = call_llm(
                    sys_p,
                    "The two summaries for comparison are: \n Summary 1: "
                    + summary_text
                    + "\n Summary 2: "
                    + query_summary,
                )
                score = _rating_to_score(rate)
            except Exception:
                score = _cosine(get_embedding(summary_text), query_embedding)
        else:
            score = _cosine(get_embedding(summary_text), query_embedding)
        scored.append((score, row.get("gid")))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [gid for _, gid in scored[: max(1, top_k)] if gid]
    return top

def seq_ret(n4j, sumq):
    top = select_top_gids(n4j, sumq, top_k=1)
    if not top:
        raise ValueError("No Summary nodes found for retrieval")
    return top[0]
