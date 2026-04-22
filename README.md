### Step 1: Create and activate environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 2>/dev/null || true
pip install requests neo4j pandas
```

### Step 2: Start Neo4j (local Docker)

```bash
docker rm -f medpneu-neo4j >/dev/null 2>&1 || true
docker run -d \
  --name medpneu-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/test1234 \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5
```

Wait until ready:

```bash
for i in $(seq 1 30); do
  if docker logs medpneu-neo4j 2>&1 | rg -q 'Bolt enabled on|Started|Remote interface available'; then
    echo NEO4J_READY
    break
  fi
  sleep 2
done
```

### Step 3: Prepare top-layer report text from MIMIC demo

Generate patient report text files used as top-layer documents:

```bash
python preprocess_mimic_demo.py \
  --mimic-root ./mimic-iv-3.1-pneumonia-100 \
  --output-dir ./dataset/mimic_demo_10_pneumonia \
  --n-patients 10
```

### Step 4: Prepare all three layer datasets

This builds:
1. Bottom layer dictionary documents
2. Middle layer guideline/knowledge chunks
3. Top layer files copied from ./dataset/mimic_demo_10

```bash
python prepare_three_layer_data.py \
  --repo-root . \
  --top-path ./dataset/mimic_demo_10_pneumonia \
  --out-root ./dataset/three_layer
```

Expected directories:
1. ./dataset/three_layer/bottom
2. ./dataset/three_layer/middle
3. ./dataset/three_layer/top

### Step 5: Run complete three-layer graph construction


```bash
USE_LLM_EXTRACTION=1 \
USE_LLM_SUMMARY=1 \
USE_REMOTE_EMBEDDINGS=1 \
python three_layer_import.py \
  --neo4j-password test1234 \
  --clear \
  --bottom ./dataset/three_layer/bottom \
  --middle ./dataset/three_layer/middle \
  --top ./dataset/three_layer/top \
  --trinity
```

Notes:
1. The pipeline automatically falls back to a non-APOC Neo4j client if APOC is unavailable.
2. Trinity linking is enabled by --trinity.

### Step 6: Verify graph build summary

At the end of import, confirm the script prints:
1. Entity node count
2. Summary node count
3. Total relationship count
4. REFERENCE relationship count
5. Per-layer subgraph counts (bottom/middle/top)

### Step 7: Visualize in Neo4j Browser

Open:
1. http://localhost:7474
2. Login with user neo4j and password test1234

Run these Cypher queries.

Global graph preview:

```cypher
MATCH p=(n)-[r]->(m)
RETURN p
LIMIT 200
```

Entity-only preview:

```cypher
MATCH p=(n:Entity)-[r]->(m:Entity)
RETURN p
LIMIT 300
```

Trinity links only:

```cypher
MATCH p=(m:Entity)-[:REFERENCE]->(n:Entity)
RETURN p
LIMIT 300
```

Relationship type distribution:

```cypher
MATCH ()-[r]->()
RETURN type(r) AS rel_type, count(*) AS c
ORDER BY c DESC
```

Node label distribution:

```cypher
MATCH (n)
UNWIND labels(n) AS lbl
RETURN lbl, count(*) AS c
ORDER BY c DESC
```

### Step 8: Run post-construction graph inference (paper-style continuation)

After graph construction, run layered retrieval + answer generation over the built three-layer graph:

```bash
python post_graph_inference.py \
  --neo4j-password test1234 \
  --question "What are the major clinical problems and likely interventions for this patient?" \
  --top-k 2 \
  --max-hops 2
```

Or use a prompt file:

```bash
cat > prompt.txt << 'EOF'
What are the main symptoms, likely diagnoses, and recommended treatment directions?
EOF

python post_graph_inference.py \
  --neo4j-password test1234 \
  --question-file ./prompt.txt
```

What this stage does:
1. Finds top-matching Summary subgraphs for your query.
2. Expands through Trinity `REFERENCE` links across layers.
3. Collects graph evidence triples.
4. Generates a cited answer grounded in retrieved evidence.

### Optional: Layer-specific visualization by gid

Bottom layer only:

```cypher
MATCH p=(n:Entity)-[r]->(m:Entity)
WHERE n.gid IN $bottom_gids
RETURN p
LIMIT 300
```

Middle layer only:

```cypher
MATCH p=(n:Entity)-[r]->(m:Entity)
WHERE n.gid IN $middle_gids
RETURN p
LIMIT 300
```

Top layer only:

```cypher
MATCH p=(n:Entity)-[r]->(m:Entity)
WHERE n.gid IN $top_gids
RETURN p
LIMIT 300
```

You can copy gids from the importer logs and pass them as Neo4j Browser parameters.

### Troubleshooting

Neo4j connection refused:
1. Ensure container is running: docker ps
2. Ensure ports are open: ss -ltnp | rg ':7474|:7687'

APOC or GDS function missing warnings:
1. This repo supports non-APOC fallback mode.
2. If you still see trinity link errors, rerun after pulling the latest code in this repository.

No space left on device:

```bash
docker system prune -af
docker volume prune -f
df -h
```

### One-command reference run

After prerequisites are done, this is the single command most users need for full graph construction:

```bash
USE_LLM_EXTRACTION=0 USE_LLM_SUMMARY=0 USE_REMOTE_EMBEDDINGS=0 \
python three_layer_import.py \
  --neo4j-password test1234 \
  --clear \
  --bottom ./dataset/three_layer/bottom \
  --middle ./dataset/three_layer/middle \
  --top ./dataset/three_layer/top \
  --trinity
```
