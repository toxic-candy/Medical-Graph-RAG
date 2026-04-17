import os
import re

from openai import OpenAI
from camel.loaders import UnstructuredIO
from camel.storages.graph_storages.graph_element import GraphElement, Node, Relationship

from utils import *
from cxr_integration import extract_cxr_mentions_from_text


KG_SYSTEM_PROMPT = (
    "You are a medical knowledge graph extraction engine. "
    "Given clinical text, extract TYPED medical entities and the relationships BETWEEN them. "
    "Do NOT create a central Patient hub node that connects to everything. "
    "Instead, create direct relationships between medical concepts.\n\n"
    "Entity types to use: Disease, Symptom, Medication, Medical_Test, Procedure, "
    "Condition, Anatomy, Measurement, Diagnosis, LabTest, Hormone, Clinical_Finding\n\n"
    "Relationship types to use: treats, caused_by, detects, associated_with, "
    "prescribed_for, performed_for, indicates, monitors, contraindicates, "
    "complication_of, symptom_of, diagnosed_by, administered_via, "
    "measured_by, affects, produces, risk_factor_for\n\n"
    "Return ONLY lines in these exact formats:\n"
    "Node(id='ENTITY_NAME', type='TYPE', description='BRIEF_DESCRIPTION')\n"
    "Relationship(subj=Node(id='S', type='ST'), obj=Node(id='O', type='OT'), type='REL_TYPE')\n\n"
    "Example output:\n"
    "Node(id='Vascular Dementia', type='Disease', description='A form of dementia caused by impaired blood flow to the brain')\n"
    "Node(id='MRI', type='Medical_Test', description='Imaging technique showing structural brain changes')\n"
    "Node(id='Chronic Ischemic Damage', type='Condition', description='Brain damage due to reduced blood supply')\n"
    "Relationship(subj=Node(id='Vascular Dementia', type='Disease'), obj=Node(id='Chronic Ischemic Damage', type='Condition'), type='caused_by')\n"
    "Relationship(subj=Node(id='MRI', type='Medical_Test'), obj=Node(id='Chronic Ischemic Damage', type='Condition'), type='detects')\n"
)




def _extract_graph_elements_from_text(raw_text, source_element):
    if os.getenv("USE_LLM_EXTRACTION", "0") != "1":
        base_ge = _fallback_extract_graph_elements(raw_text, source_element)
        return _augment_with_cxr(raw_text, base_ge)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_API_BASE_URL", "https://openrouter.ai/api/v1"),
        timeout=20,
        max_retries=0,
    )
    model = os.getenv("OPENAI_MODEL", "meta-llama/llama-3-8b-instruct")

    user_prompt = (
        "Extract typed medical entities and the direct relationships between them. "
        "Do NOT make a Patient node the center of every relationship. "
        "Focus on medical concept interconnections.\n\n"
        f"CONTENT:\n{raw_text}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": KG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1200,
            temperature=0.0,
            timeout=45,
        )
        content = response.choices[0].message.content or ""
    except Exception:
        base_ge = _fallback_extract_graph_elements(raw_text, source_element)
        return _augment_with_cxr(raw_text, base_ge)

    # Parse nodes — with optional description field
    node_pattern = r"Node\(id='(.*?)', type='(.*?)'(?:, description='(.*?)')?\)"
    rel_pattern = (
        r"Relationship\(subj=Node\(id='(.*?)', type='(.*?)'\), "
        r"obj=Node\(id='(.*?)', type='(.*?)'\), type='(.*?)'\)"
    )

    nodes = {}
    relationships = []

    for match in re.finditer(node_pattern, content):
        node_id, node_type = match.group(1).strip(), match.group(2).strip()
        node_desc = (match.group(3) or "").strip()
        node_type = node_type or "Entity"
        if node_id and node_id not in nodes:
            props = {"source": "llm_extracted"}
            if node_desc:
                props["description"] = node_desc
            nodes[node_id] = Node(id=node_id, type=node_type, properties=props)

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
                properties={"source": "llm_extracted"},
            )
        if obj_id not in nodes:
            nodes[obj_id] = Node(
                id=obj_id,
                type=obj_type,
                properties={"source": "llm_extracted"},
            )

        relationships.append(
            Relationship(
                subj=nodes[subj_id],
                obj=nodes[obj_id],
                type=rel_type,
                properties={"source": "llm_extracted"},
            )
        )

    graph_element = GraphElement(
        nodes=list(nodes.values()),
        relationships=relationships,
        source=source_element,
    )
    if not graph_element.nodes:
        graph_element = _fallback_extract_graph_elements(raw_text, source_element)
    return _augment_with_cxr(raw_text, graph_element)


def _icd_chapter(code_str):
    """Extract the ICD chapter prefix (first 3 chars) for grouping."""
    code = re.sub(r"[^A-Za-z0-9]", "", code_str.strip())
    return code[:3] if len(code) >= 3 else code


def _augment_with_cxr(raw_text, graph_element):
    patient_subject_id, cxr_mentions = extract_cxr_mentions_from_text(raw_text)
    if not patient_subject_id or not cxr_mentions:
        return graph_element

    node_index = {str(node.id): node for node in graph_element.nodes}
    rel_index = {
        (str(rel.subj.id), str(rel.obj.id), str(rel.type).lower())
        for rel in graph_element.relationships
    }

    patient_node_id = f"patient_{patient_subject_id}"
    patient_node = node_index.get(patient_node_id)
    if patient_node is None:
        patient_node = Node(
            id=patient_node_id,
            type="Patient",
            properties={"source": "cxr_integration"},
        )
        node_index[patient_node_id] = patient_node

    for mention in cxr_mentions:
        study_id = mention.get("study_id", "")
        if not study_id:
            continue
        impression = (mention.get("impression") or "").strip()
        cxr_node_id = f"cxr_{patient_subject_id}_{study_id}"

        cxr_node = node_index.get(cxr_node_id)
        description = f"Chest X-ray report study_id={study_id}"
        if impression:
            description = f"{description}; impression: {impression}"

        if cxr_node is None:
            props = {"source": "cxr_integration", "study_id": study_id}
            if description:
                props["description"] = description
            cxr_node = Node(id=cxr_node_id, type="CXR_Report", properties=props)
            node_index[cxr_node_id] = cxr_node
        else:
            if cxr_node.properties is None:
                cxr_node.properties = {}
            cxr_node.properties.setdefault("study_id", study_id)
            cxr_node.properties.setdefault("source", "cxr_integration")
            if description and not cxr_node.properties.get("description"):
                cxr_node.properties["description"] = description

        rel_key = (patient_node_id, cxr_node_id, "has_cxr")
        if rel_key in rel_index:
            continue
        graph_element.relationships.append(
            Relationship(
                subj=patient_node,
                obj=cxr_node,
                type="has_cxr",
                properties={"source": "cxr_integration", "study_id": study_id},
            )
        )
        rel_index.add(rel_key)

    graph_element.nodes = list(node_index.values())
    return graph_element


def _fallback_extract_graph_elements(raw_text, source_element):
    """Structured fallback: build a medical knowledge graph from structured
    clinical text.  Creates a distributed mesh of inter-entity relationships
    rather than a patient-centric star or any single hub node.
    """
    nodes = {}
    relationships = []

    def _add_node(node_id, node_type, description=""):
        key = node_id.strip()
        if key and key not in nodes:
            props = {"source": "structured_fallback"}
            if description:
                props["description"] = description
            nodes[key] = Node(id=key, type=node_type, properties=props)
        return nodes.get(key)

    # =====================================================================
    # 1) DICTIONARY DATA  (bottom / middle layer)
    #    Create clustered sub-graphs using ICD code grouping.
    # =====================================================================

    # --- Diagnosis dictionary entries ---
    icd_diag_groups = {}  # chapter_prefix -> list of nodes
    for match in re.finditer(
        r"DIAGNOSIS\s+code=(\S+)\s+icd_version=(\S+)\s+name=(.+?)(?:\n|$)", raw_text
    ):
        code, ver, name = match.group(1), match.group(2), match.group(3).strip()
        d_node = _add_node(name, "Diagnosis", f"ICD-{ver} code {code}: {name}")
        chapter = _icd_chapter(code)
        icd_diag_groups.setdefault(chapter, []).append(d_node)

    # Link diagnoses within the same ICD chapter (chain rather than clique)
    for chapter, group_nodes in icd_diag_groups.items():
        for i in range(len(group_nodes) - 1):
            relationships.append(
                Relationship(
                    subj=group_nodes[i],
                    obj=group_nodes[i + 1],
                    type="icd_related",
                    properties={"source": "structured_fallback", "icd_chapter": chapter},
                )
            )

    # --- Procedure dictionary entries ---
    icd_proc_groups = {}
    for match in re.finditer(
        r"PROCEDURE\s+code=(\S+)\s+icd_version=(\S+)\s+name=(.+?)(?:\n|$)", raw_text
    ):
        code, ver, name = match.group(1), match.group(2), match.group(3).strip()
        p_node = _add_node(name, "Procedure", f"ICD-{ver} procedure code {code}: {name}")
        chapter = _icd_chapter(code)
        icd_proc_groups.setdefault(chapter, []).append(p_node)

    for chapter, group_nodes in icd_proc_groups.items():
        for i in range(len(group_nodes) - 1):
            relationships.append(
                Relationship(
                    subj=group_nodes[i],
                    obj=group_nodes[i + 1],
                    type="icd_related",
                    properties={"source": "structured_fallback", "icd_chapter": chapter},
                )
            )

    # --- Lab dictionary entries ---
    lab_category_groups = {}
    for match in re.finditer(
        r"LAB_TEST\s+itemid=(\S+)\s+label=(.+?)\s+fluid=(\S+)\s+category=(.+?)(?:\n|$)",
        raw_text,
    ):
        itemid, label, fluid, category = (
            match.group(1),
            match.group(2).strip(),
            match.group(3).strip(),
            match.group(4).strip(),
        )
        desc = f"Lab test (item {itemid}): {label}, fluid={fluid}, category={category}"
        l_node = _add_node(label, "LabTest", desc)
        group_key = f"{fluid}_{category}"
        lab_category_groups.setdefault(group_key, []).append(l_node)

    # Chain labs within the same category
    for group_key, group_nodes in lab_category_groups.items():
        for i in range(len(group_nodes) - 1):
            relationships.append(
                Relationship(
                    subj=group_nodes[i],
                    obj=group_nodes[i + 1],
                    type="same_category",
                    properties={"source": "structured_fallback", "category": group_key},
                )
            )

    # Cross-link: connect each ICD diag chapter head to related procedure chapter head
    diag_heads = {ch: ns[0] for ch, ns in icd_diag_groups.items() if ns}
    proc_heads = {ch: ns[0] for ch, ns in icd_proc_groups.items() if ns}
    for chapter in set(diag_heads) & set(proc_heads):
        relationships.append(
            Relationship(
                subj=proc_heads[chapter],
                obj=diag_heads[chapter],
                type="procedure_for_category",
                properties={"source": "structured_fallback"},
            )
        )

    # =====================================================================
    # 2) GUIDELINE DATA  (middle layer)
    # =====================================================================

    guideline_conditions = []
    for match in re.finditer(r"Condition\s+\d+:\s+Consider diagnosis\s+'([^']+)'", raw_text):
        name = match.group(1).strip()
        g_node = _add_node(name, "Diagnosis", f"Guideline-referenced diagnosis: {name}")
        guideline_conditions.append(g_node)

    guideline_interventions = []
    for match in re.finditer(r"Intervention\s+\d+:\s+Procedure option\s+'([^']+)'", raw_text):
        name = match.group(1).strip()
        g_node = _add_node(name, "Procedure", f"Guideline-referenced procedure: {name}")
        guideline_interventions.append(g_node)

    # Round-robin link interventions to conditions (1:1 instead of many-to-many)
    if guideline_conditions and guideline_interventions:
        n_cond = len(guideline_conditions)
        for i, p_node in enumerate(guideline_interventions):
            d_node = guideline_conditions[i % n_cond]
            relationships.append(
                Relationship(
                    subj=p_node,
                    obj=d_node,
                    type="indicated_for",
                    properties={"source": "structured_fallback"},
                )
            )

    # Chain guideline conditions (sequential, not clique)
    for i in range(len(guideline_conditions) - 1):
        relationships.append(
            Relationship(
                subj=guideline_conditions[i],
                obj=guideline_conditions[i + 1],
                type="associated_with",
                properties={"source": "structured_fallback"},
            )
        )

    # =====================================================================
    # 3) PATIENT CLINICAL DATA  (top layer)
    #    Distribute relationships evenly — NO single diagnosis hub.
    # =====================================================================

    patient_match = re.search(r"patient_id:\s*(\d+)", raw_text)
    patient_id = f"patient_{patient_match.group(1)}" if patient_match else None
    patient_node = _add_node(patient_id, "Patient") if patient_id else None

    # Split by admission blocks
    admission_blocks = re.split(r"(?=Admission summary:)", raw_text)

    for block in admission_blocks:
        # --- Diagnoses ---
        block_diagnoses = []
        for diag in re.findall(r"\[\d+\]\s*([^;\.\n]+)", block):
            d = diag.strip()
            if not d:
                continue
            d_node = _add_node(d, "Diagnosis", f"Clinical diagnosis: {d}")
            block_diagnoses.append(d_node)

        # --- Procedures ---
        block_procedures = []
        proc_line = re.search(r"Procedures:\s*(.*)", block)
        if proc_line:
            for proc in [p.strip() for p in proc_line.group(1).split(";") if p.strip()]:
                proc_name = re.sub(r"\s+on\s+\d{4}-\d{2}-\d{2}.*", "", proc).strip()
                if not proc_name:
                    continue
                p_node = _add_node(proc_name, "Procedure", f"Medical procedure: {proc_name}")
                block_procedures.append(p_node)

        # --- Medications ---
        block_medications = []
        meds_line = re.search(r"Medications:\s*(.*)", block)
        if meds_line:
            for med in [m.strip() for m in meds_line.group(1).split(";") if m.strip()]:
                med_name = med.split(" via ")[0].strip()
                route = med.split(" via ")[1].strip() if " via " in med else ""
                if not med_name:
                    continue
                desc = f"Medication: {med_name}"
                if route:
                    desc += f" administered via {route}"
                m_node = _add_node(med_name, "Medication", desc)
                block_medications.append(m_node)

        # --- Lab Tests ---
        block_labs = []
        labs_line = re.search(r"Recent labs:\s*(.*)", block)
        if labs_line:
            for lab in [l.strip() for l in labs_line.group(1).split(";") if l.strip()]:
                lab_name = lab.split(":")[0].strip()
                lab_value = lab.split(":")[1].strip() if ":" in lab else ""
                if not lab_name:
                    continue
                desc = f"Laboratory test: {lab_name}"
                if lab_value:
                    desc += f", result: {lab_value}"
                l_node = _add_node(lab_name, "LabTest", desc)
                block_labs.append(l_node)

        # === INTER-ENTITY RELATIONSHIPS (distributed, no mega-hub) ===

        n_diag = len(block_diagnoses)
        if n_diag == 0:
            continue

        # Procedure --performed_for--> Diagnosis (round-robin, 1:1)
        for i, p_node in enumerate(block_procedures):
            d_node = block_diagnoses[i % n_diag]
            relationships.append(
                Relationship(
                    subj=p_node, obj=d_node, type="performed_for",
                    properties={"source": "structured_fallback"},
                )
            )

        # Medication --treats--> Diagnosis (round-robin, 1:1)
        for i, m_node in enumerate(block_medications):
            d_node = block_diagnoses[i % n_diag]
            relationships.append(
                Relationship(
                    subj=m_node, obj=d_node, type="treats",
                    properties={"source": "structured_fallback"},
                )
            )

        # LabTest --monitors--> Diagnosis (round-robin, 1:1)
        for i, l_node in enumerate(block_labs):
            d_node = block_diagnoses[i % n_diag]
            relationships.append(
                Relationship(
                    subj=l_node, obj=d_node, type="monitors",
                    properties={"source": "structured_fallback"},
                )
            )

        # Diagnosis chain (sequential, not clique — avoids O(n²) edges)
        for i in range(n_diag - 1):
            relationships.append(
                Relationship(
                    subj=block_diagnoses[i],
                    obj=block_diagnoses[i + 1],
                    type="associated_with",
                    properties={"source": "structured_fallback"},
                )
            )

        # Medication --administered_via--> route grouping
        # (connect medications sharing the same route)
        route_groups = {}
        for m_node in block_medications:
            desc = (m_node.properties or {}).get("description", "")
            route_match = re.search(r"via (\S+)", desc)
            if route_match:
                route = route_match.group(1)
                route_groups.setdefault(route, []).append(m_node)
        for route, meds in route_groups.items():
            for i in range(len(meds) - 1):
                relationships.append(
                    Relationship(
                        subj=meds[i], obj=meds[i + 1], type="same_route",
                        properties={"source": "structured_fallback", "route": route},
                    )
                )

    # Patient gets a single link to just the FIRST diagnosis (minimal presence)
    if patient_node and block_diagnoses:
        relationships.append(
            Relationship(
                subj=patient_node,
                obj=block_diagnoses[0],
                type="diagnosed_with",
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

    if args.grained_chunk == True:
        # Ensure patient-level CXR links are preserved even if fine-grained
        # chunking splits patient_id and CXR lines across different chunks.
        whole_element = uio.create_element_from_text(text=whole_chunk)
        cxr_only = GraphElement(nodes=[], relationships=[], source=whole_element)
        cxr_only = _augment_with_cxr(whole_chunk, cxr_only)
        if cxr_only.nodes:
            cxr_only = add_ge_emb(cxr_only)
            cxr_only = add_gid(cxr_only, gid)
            n4j.add_graph_elements(graph_elements=[cxr_only])

    if args.ingraphmerge:
        merge_similar_nodes(n4j, gid)
    add_sum(n4j, whole_chunk, gid)
    return n4j
