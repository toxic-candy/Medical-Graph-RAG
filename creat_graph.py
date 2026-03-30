import os
import re
import signal

from openai import OpenAI
from camel.loaders import UnstructuredIO
from camel.storages.graph_storages.graph_element import GraphElement, Node, Relationship

from utils import *


KG_SYSTEM_PROMPT = (
    "You extract medical entities and relationships from text. "
    "Return ONLY lines in this exact format: "
    "Node(id='ENTITY', type='TYPE') and "
    "Relationship(subj=Node(id='S', type='ST'), obj=Node(id='O', type='OT'), type='REL')."
)


class _HardTimeoutError(TimeoutError):
    pass


def _run_with_hard_timeout(seconds, func, *args, **kwargs):
    def _handler(signum, frame):
        raise _HardTimeoutError(f"LLM request exceeded {seconds}s hard timeout")

    prev_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        return func(*args, **kwargs)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)


def _extract_graph_elements_from_text(raw_text, source_element):
    if os.getenv("USE_LLM_EXTRACTION", "0") != "1":
        return _fallback_extract_graph_elements(raw_text, source_element)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_API_BASE_URL", "https://openrouter.ai/api/v1"),
        timeout=20,
        max_retries=0,
    )
    model = os.getenv("OPENAI_MODEL", "meta-llama/llama-3-8b-instruct")

    user_prompt = (
        "Extract nodes and directed relationships from this medical content. "
        "Use concise entity IDs and meaningful types.\n\n"
        f"CONTENT:\n{raw_text}"
    )

    try:
        response = _run_with_hard_timeout(
            45,
            client.chat.completions.create,
            model=model,
            messages=[
                {"role": "system", "content": KG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1200,
            temperature=0.0,
            timeout=20,
        )
        content = response.choices[0].message.content or ""
    except Exception:
        return _fallback_extract_graph_elements(raw_text, source_element)

    node_pattern = r"Node\(id='(.*?)', type='(.*?)'\)"
    rel_pattern = (
        r"Relationship\(subj=Node\(id='(.*?)', type='(.*?)'\), "
        r"obj=Node\(id='(.*?)', type='(.*?)'\), type='(.*?)'\)"
    )

    nodes = {}
    relationships = []

    for match in re.finditer(node_pattern, content):
        node_id, node_type = match.groups()
        node_id = node_id.strip()
        node_type = node_type.strip() or "Entity"
        if node_id and node_id not in nodes:
            nodes[node_id] = Node(
                id=node_id,
                type=node_type,
                properties={"source": "openrouter_llama_extracted"},
            )

    for match in re.finditer(rel_pattern, content):
        subj_id, subj_type, obj_id, obj_type, rel_type = match.groups()
        subj_id = subj_id.strip()
        obj_id = obj_id.strip()
        subj_type = subj_type.strip() or "Entity"
        obj_type = obj_type.strip() or "Entity"
        rel_type = rel_type.strip() or "RELATED_TO"

        if not subj_id or not obj_id:
            continue

        if subj_id not in nodes:
            nodes[subj_id] = Node(
                id=subj_id,
                type=subj_type,
                properties={"source": "openrouter_llama_extracted"},
            )
        if obj_id not in nodes:
            nodes[obj_id] = Node(
                id=obj_id,
                type=obj_type,
                properties={"source": "openrouter_llama_extracted"},
            )

        relationships.append(
            Relationship(
                subj=nodes[subj_id],
                obj=nodes[obj_id],
                type=rel_type,
                properties={"source": "openrouter_llama_extracted"},
            )
        )

    graph_element = GraphElement(
        nodes=list(nodes.values()),
        relationships=relationships,
        source=source_element,
    )
    if not graph_element.nodes:
        return _fallback_extract_graph_elements(raw_text, source_element)
    return graph_element


def _fallback_extract_graph_elements(raw_text, source_element):
    nodes = {}
    relationships = []

    def _add_node(node_id, node_type):
        key = node_id.strip()
        if key and key not in nodes:
            nodes[key] = Node(
                id=key,
                type=node_type,
                properties={"source": "structured_fallback"},
            )
        return nodes.get(key)

    patient_match = re.search(r"patient_id:\s*(\d+)", raw_text)
    patient_id = f"patient_{patient_match.group(1)}" if patient_match else "patient_unknown"
    patient_node = _add_node(patient_id, "Patient")

    for diagnosis in re.findall(r"\[\d+\]\s*([^;\.\n]+)", raw_text):
        d = diagnosis.strip()
        if not d:
            continue
        d_node = _add_node(d, "Diagnosis")
        relationships.append(
            Relationship(
                subj=patient_node,
                obj=d_node,
                type="HAS_DIAGNOSIS",
                properties={"source": "structured_fallback"},
            )
        )

    proc_line = re.search(r"Procedures:\s*(.*)", raw_text)
    if proc_line:
        for proc in [p.strip() for p in proc_line.group(1).split(";") if p.strip()]:
            p_node = _add_node(proc, "Procedure")
            relationships.append(
                Relationship(
                    subj=patient_node,
                    obj=p_node,
                    type="UNDERWENT",
                    properties={"source": "structured_fallback"},
                )
            )

    meds_line = re.search(r"Medications:\s*(.*)", raw_text)
    if meds_line:
        for med in [m.strip() for m in meds_line.group(1).split(";") if m.strip()]:
            med_name = med.split(" via ")[0].strip()
            m_node = _add_node(med_name, "Medication")
            relationships.append(
                Relationship(
                    subj=patient_node,
                    obj=m_node,
                    type="RECEIVED_MEDICATION",
                    properties={"source": "structured_fallback"},
                )
            )

    labs_line = re.search(r"Recent labs:\s*(.*)", raw_text)
    if labs_line:
        for lab in [l.strip() for l in labs_line.group(1).split(";") if l.strip()]:
            lab_name = lab.split(":")[0].strip()
            if not lab_name:
                continue
            l_node = _add_node(lab_name, "LabTest")
            relationships.append(
                Relationship(
                    subj=patient_node,
                    obj=l_node,
                    type="HAS_LAB",
                    properties={"source": "structured_fallback"},
                )
            )

    return GraphElement(
        nodes=list(nodes.values()),
        relationships=relationships,
        source=source_element,
    )


def creat_metagraph(args, content, gid, n4j):

    # Set instance
    uio = UnstructuredIO()
    whole_chunk = content

    if args.grained_chunk == True:
        # Lazy import avoids pulling LangChain/Pydantic stack unless requested.
        from data_chunk import run_chunk
        content = run_chunk(content)
    else:
        content = [content]
    for cont in content:
        element_example = uio.create_element_from_text(text=cont)

        graph_elements = _extract_graph_elements_from_text(cont, element_example)
        if not graph_elements.nodes:
            continue
        graph_elements = add_ge_emb(graph_elements)
        graph_elements = add_gid(graph_elements, gid)

        n4j.add_graph_elements(graph_elements=[graph_elements])
    if args.ingraphmerge:
        merge_similar_nodes(n4j, gid)
    add_sum(n4j, whole_chunk, gid)
    return n4j

