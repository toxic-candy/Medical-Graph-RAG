from neo4j import GraphDatabase


class SimpleNeo4jGraph:
    """Fallback Neo4j client that does not require APOC procedures."""

    def __init__(self, url, username, password):
        self.driver = GraphDatabase.driver(url, auth=(username, password))
        self.driver.verify_connectivity()

    def query(self, query, params=None):
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def add_graph_elements(self, graph_elements):
        for element in graph_elements:
            for node in element.nodes:
                props = dict(node.properties or {})
                gid = props.get("gid")
                q = """
                MERGE (n:Entity {id: $id, gid: $gid})
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

            for rel in element.relationships:
                props = dict(rel.properties or {})
                gid = props.get("gid")
                q = """
                MATCH (s:Entity {id: $subj_id, gid: $gid})
                MATCH (o:Entity {id: $obj_id, gid: $gid})
                MERGE (s)-[r:REL {gid: $gid, subj_id: $subj_id, obj_id: $obj_id, rel_type: $rel_type}]->(o)
                SET r += $props
                RETURN r
                """
                self.query(
                    q,
                    {
                        "subj_id": str(rel.subj.id),
                        "obj_id": str(rel.obj.id),
                        "gid": gid,
                        "rel_type": str(rel.type),
                        "props": props,
                    },
                )
