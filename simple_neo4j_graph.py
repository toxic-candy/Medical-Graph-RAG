import re

from neo4j import GraphDatabase


def _safe_label(raw: str) -> str:
    """Sanitise a string so it is safe to use as a Neo4j label.

    Neo4j labels must start with a letter and contain only alphanumeric
    characters and underscores.  We strip everything else and fall back to
    ``Entity`` if the result would be empty.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", raw.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "Entity"
    return cleaned


def _safe_rel_type(raw: str) -> str:
    """Sanitise a string so it is safe as a Neo4j relationship type."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", raw.strip().upper())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "RELATED_TO"


class SimpleNeo4jGraph:
    """Fallback Neo4j client that does not require APOC procedures.

    Stores nodes with **typed labels** (e.g. ``:Entity:Disease``) and
    relationships with **meaningful types** (e.g. ``-[:TREATS]->``).
    """

    def __init__(self, url, username, password):
        self.driver = GraphDatabase.driver(url, auth=(username, password))
        self.driver.verify_connectivity()

    def query(self, query, params=None):
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def add_graph_elements(self, graph_elements):
        for element in graph_elements:
            # --- Nodes ---
            for node in element.nodes:
                props = dict(node.properties or {})
                gid = props.get("gid")
                node_type = _safe_label(str(node.type)) if node.type else "Entity"

                # Always include :Entity as a base label for backward compat.
                # Add the typed label as a second label.
                labels = f"Entity:{node_type}" if node_type != "Entity" else "Entity"

                q = f"""
                MERGE (n:{labels} {{id: $id, gid: $gid}})
                SET n.type = $type
                SET n += $props
                RETURN n
                """
                self.query(
                    q,
                    {
                        "id": str(node.id),
                        "gid": gid,
                        "type": str(node.type),
                        "props": props,
                    },
                )

            # --- Relationships ---
            for rel in element.relationships:
                props = dict(rel.properties or {})
                gid = props.get("gid")
                rel_type = _safe_rel_type(str(rel.type)) if rel.type else "RELATED_TO"

                q = f"""
                MATCH (s:Entity {{id: $subj_id, gid: $gid}})
                MATCH (o:Entity {{id: $obj_id, gid: $gid}})
                MERGE (s)-[r:{rel_type} {{gid: $gid, subj_id: $subj_id, obj_id: $obj_id}}]->(o)
                SET r += $props
                SET r.rel_type = $rel_type_str
                RETURN r
                """
                self.query(
                    q,
                    {
                        "subj_id": str(rel.subj.id),
                        "obj_id": str(rel.obj.id),
                        "gid": gid,
                        "rel_type_str": str(rel.type),
                        "props": props,
                    },
                )
