# Medical-Graph-RAG
We build a Graph RAG System specifically for the medical domain.

Check our paper here: https://arxiv.org/abs/2408.04187

## Demo
a docker demo is here: https://hub.docker.com/repository/docker/jundewu/medrag-post/general
 
Use it by: docker run -it --rm --storage-opt size=10G -p 7860:7860 \ -e OPENAI_API_KEY= your_key -e NCBI_API_KEY= your_key medrag-post

this demo used web-based searches on PubMed instead of locally storing medical papers and textbooks to detour the license wall.

## Quick Start (Baseline: a simple Graph RAG pipeline on medical data)
1. conda env create -f medgraphrag.yml

2. export OPENAI_API_KEY = your OPENAI_API_KEY

3. python run.py -simple True (now using ./dataset_ex/report_0.txt as RAG doc, "What is the main symptom of the patient?" as the prompt, change the prompt in run.py as you like.)

## Complete Local Workflow (Data Preprocessing -> Full Three-Layer Graph)

This section documents a complete local workflow that starts from MIMIC demo structured data and ends with a fully constructed three-layer graph in Neo4j, including trinity links and visualization queries.

### Prerequisites
1. Clone this repository and cd into it.
2. Install Docker (for Neo4j).
3. Place MIMIC-IV demo data under:
   - ./mimic-iv-clinical-database-demo-2.2
4. Ensure Python environment is available (conda or venv).

### Step 1: Create and activate environment

Option A (conda):

```bash
conda env create -f medgraphrag.yml
conda activate medgraphrag
```

Option B (venv):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt 2>/dev/null || true
pip install requests neo4j pandas
```

### Step 2: Start Neo4j (local Docker)

```bash
docker rm -f medgraphrag-neo4j >/dev/null 2>&1 || true
docker run -d \
  --name medgraphrag-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/test1234 \
  neo4j:5
```

Wait until ready:

```bash
for i in $(seq 1 30); do
  if docker logs medgraphrag-neo4j 2>&1 | rg -q 'Bolt enabled on|Started|Remote interface available'; then
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
  --mimic-root ./mimic-iv-clinical-database-demo-2.2 \
  --output-dir ./dataset/mimic_demo_10 \
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
  --top-path ./dataset/mimic_demo_10 \
  --out-root ./dataset/three_layer
```

Expected directories:
1. ./dataset/three_layer/bottom
2. ./dataset/three_layer/middle
3. ./dataset/three_layer/top

### Step 5: Run complete three-layer graph construction

Run in deterministic local mode (no remote LLM extraction/summarization/embeddings):

```bash
USE_LLM_EXTRACTION=0 \
USE_LLM_SUMMARY=0 \
USE_REMOTE_EMBEDDINGS=0 \
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

## Build from scratch (Complete Graph RAG flow in the paper)

### About the dataset
#### Paper Datasets
**Top-level Private data (user-provided)**: we used [MIMIC IV dataset](https://physionet.org/content/mimiciv/3.0/) as the private data.

**Medium-level Books and Papers**: We used MedC-K as the medium-level data. The dataset sources from [S2ORC](https://github.com/allenai/s2orc). Only those papers with PubMed IDs are deemed as medical-related and used during pretraining. The book is listed in this repo as [MedicalBook.xlsx](https://github.com/MedicineToken/Medical-Graph-RAG/blob/main/MedicalBook.xlsx), due to licenses, we cannot release raw content. For reproducing, pls buy and process the books.

**Bottom-level Dictionary data**: We used [Unified Medical Language System (UMLS)](https://www.nlm.nih.gov/research/umls/index.html) as the bottom level data. To access it, you'll need to create an account and apply for usage. It is free and approval is typically fast.

In the code, we use the 'trinity' argument to enable the hierarchy graph linking function. If set to True, you must also provide a 'gid' (graph ID) to specify which graphs the top-level should link to. UMLS is largely structured as a graph, so minimal effort is required to construct it. However, MedC-K must be constructed as graph data. There are several methods you can use, such as the approach we used to process the top-level in this repo (open-source LLMs are recommended to keep costs down), or you can opt for non-learning-based graph construction algorithms (faster, cheaper, and generally noisier)

#### Example Datasets
Recognizing that accessing and processing all the data mentioned may be challenging, we are working to provide simpler example dataset to demonstrate functionality. Currently, we are using the mimic_ex [here](https://huggingface.co/datasets/Morson/mimic_ex) here as the Top-level data, which is the processed smaller dataset derived from MIMIC. For Medium-level and Bottom-level data, we are in the process of identifying suitable alternatives to simplify the implementation, welcome for any recommendations.

### 1. Prepare the environment, Neo4j and LLM
1. conda env create -f medgraphrag.yml


2. prepare neo4j and LLM (using ChatGPT here for an example), you need to export:

export OPENAI_API_KEY = your OPENAI_API_KEY

export NEO4J_URL= your NEO4J_URL

export NEO4J_USERNAME= your NEO4J_USERNAME

export NEO4J_PASSWORD= your NEO4J_PASSWORD

### 2. Construct the graph (use "mimic_ex" dataset as an example)
1. Download mimic_ex [here](https://huggingface.co/datasets/Morson/mimic_ex), put that under your data path, like ./dataset/mimic_ex

2. python run.py -dataset mimic_ex -data_path ./dataset/mimic_ex(where you put the dataset) -grained_chunk -ingraphmerge -construct_graph

### 3. Model Inference
1. put your prompt to ./prompt.txt

2. python run.py -dataset mimic_ex -data_path ./dataset/mimic_ex(where you put the dataset) -inference

## Acknowledgement
We are building on [CAMEL](https://github.com/camel-ai/camel), an awesome framework for construcing multi-agent pipeline.

## Cite
~~~
@article{wu2024medical,
  title={Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation},
  author={Wu, Junde and Zhu, Jiayuan and Qi, Yunli},
  journal={arXiv preprint arXiv:2408.04187},
  year={2024}
}
~~~
