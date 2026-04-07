# Medical-Graph-RAG: Extreme Detail Project Step-by-Step Report

This report documents the entire operation of the **Medical-Graph-RAG** project without abstraction. It details exactly how data is ingested from raw MIMIC-IV clinical databases, transformed into text reports, structured into three distinct layers (Bottom, Middle, Top), digested into nodes and graph relationships, and finally retrieved and reasoned over by a Graph Retrieval-Augmented Generation (RAG) system. Every single Python code file is broken down logically explaining the implementation depth.

---

## Data Engineering & Preprocessing

The primary goal of preprocessing is transforming tabular or continuous clinical datasets into unstructured-yet-ordered text which is suitable for the Graph RAG pipeline.

### `preprocess_mimic_demo.py`
**Purpose**: Transforms tabular data from MIMIC-IV Demo CSVs into comprehensive synthetic narrative `.txt` reports per patient.
1. **Data Loading**: Uses pandas `pd.read_csv` on files inside `mimic-iv-clinical-database-demo-2.2/hosp/` including `patients.csv`, `admissions.csv`, `diagnoses_icd.csv`, `procedures_icd.csv`, `labevents.csv`, `prescriptions.csv`, `microbiologyevents.csv`, and matching dictionary files.
2. **Merging Dictionaries**: Enforces `icd_code` and `icd_version` columns to strings, then executes a `merge` (`left` join) of actual events (diagnoses and procedures) with their dictionary counterparts (`d_icd_diagnoses.csv`, `d_icd_procedures.csv`) to grab the `long_title`. It similarly joins `labevents.csv` with `d_labitems.csv` to fetch label strings.
3. **Patient Iteration**: Selects the first `n_patients` (via `--n-patients` arg). For each patient:
   - Fetches base demographic data (`gender`, `anchor_age`, `dod`).
   - Retrieves up to 3 admissions for the given patient, grabbing admission specifics like admittance and discharge locations and race/insurance details.
4. **Iterating Health Data Arrays**:
   - Computes up to 12 diagnoses, sorted by sequence number (`seq_num`).
   - Computes up to 8 procedures, also prioritized by sequence.
   - Computes up to 20 lab tests sorted by `charttime`, capturing `itemid`, `value`, `valueuom`, and `flag` (abnormal bounds).
   - Computes up to 15 medications sorted by `starttime`, capturing the formulation `dose_val_rx`, `dose_unit_rx` and `route`.
   - Computes up to 10 microbiology event results incorporating test names and organisms.
5. **Output**: Fills an array of lines combining all elements and effectively synthesizes an entire medical note. Uses `.write_text` to save to `./dataset/mimic_demo_X/report_YY_subject_XXXX.txt`.

### `prepare_three_layer_data.py`
**Purpose**: Divides the data into the 3 discrete knowledge layers specified in the paper (Bottom as ontologies, Middle as guidelines, Top as patient narratives) that will live in `./dataset/three_layer/{bottom,middle,top}`.
1. **Bottom Layer (`build_bottom_layer`)**: Reads dictionary files directly (`d_icd_diagnoses`, `d_icd_procedures`, `d_labitems`). Outputs `.txt` line-files in a pseudo-structured format:
   - Up to 2500 lines formatted as: `DIAGNOSIS code=[code] icd_version=[ver] name=[title]`.
   - Up to 1800 procedure nodes formatted as: `PROCEDURE code=[code] icd_version=[ver] name=[title]`.
   - Up to 1800 lab lines: `LAB_TEST itemid=[itemid] label=[label] fluid=[fluid] category=[category]`.
2. **Middle Layer (`build_middle_layer`)**: Attempts to read `MedicalBook.xlsx` to extract explicit guidelines parsing rows up to 400 lines into bar-delimited text files (`foo | bar`). If the Excel isn't present, falls back to parsing MIMIC ICD code titles and generating synthetic "Condition" and "Intervention" guidelines chunked 120 elements at a time.
3. **Top Layer**: Pure copying operator passing through `.txt` records from `dataset/mimic_demo_10` unchanged to `dataset/three_layer/top/`.

### `agentic_chunker.py` & `data_chunk.py`
**Purpose**: Handles long documents via semantic sentence extraction and logical grouping rather than brute-force limits.
1. **`data_chunk.py`**:
   - Intercepts raw clinical paragraph text and routes it to LangChain's pipeline (`ChatOpenAI(gpt-4-1106-preview)` with prompt `wfh/proposal-indexing`).
   - Extracts atomic medical propositions via a Pydantic extraction chain targeting a schema returning `List[str]`.
   - Propositions are funneled through `AgenticChunker`.
2. **`agentic_chunker.py`**:
   - **Proposition Evaluation**: When a new proposition arrives, `_find_relevant_chunk` runs an LLM comparison seeing if it naturally matches an existing chunk's context logic via ID mapping summary strings.
   - **Insertion or Instantiation**: If it matches, it inserts it (updating titles/summaries dynamically). If not, `_create_new_chunk` initializes a new uuid dictionary bucket.

### `split_medr.py` & `dataloader.py`
**Purpose**: Minor helper files.
1. **`dataloader.py`**: Exposes `load_high`, stripping whitespaces via `.strip()` traversing files line-by-line storing as a full combined UTF-8 document string.
2. **`split_medr.py`**: An auxiliary file segmenting an anonymized bulk file matching precisely on "history of present illness:" delimiters and chunk-writing them to individual dataset `.txt`s.

---

## Graph Construction

Reads the parsed clinical strings, identifies Entities and inter-Entity Relationships using semantic awareness, and writes them to Neo4j.

### `creat_graph.py` & `simple_neo4j_graph.py`
**Purpose**: Maps unstructured context text string chunks into actual graph structures composed of typed `Node`s and `Relationship`s.
1. **Execution Entry**: `creat_metagraph` receives `content` text. Chunking evaluates whether `agentic` or straightforward logic is used.
2. **Extraction Engine (`_extract_graph_elements_from_text`)**:
   - If `USE_LLM_EXTRACTION=1`, calls OpenRouter/OpenAI API with system prompt asking for `Node(id='', type='', ...)` format.
   - Regular expression pattern scanning validates and registers `Nodes` onto Python memory caching their types and IDs, and validates relationships mapping src `subj_id` to dest `obj_id`.
3. **Regex Fallback (`_fallback_extract_graph_elements`)**: If LLM fails or is disabled.
   - Diagnoses, Procedures, Labs are regex'ed out of Dictionary format text into Python objects.
   - Generates sequential chain relationships `icd_related` matching characters within ICD classifications or `same_category` for labs. Links Diagnosis targets to equivalent process actions.
   - In patient data, matches Regex patterns `\[\d+\]` or "Procedures:" splitting by semicolons into discrete nodes without centralizing to a 'Patient node macrohub'. It builds local relationship meshes pairing interventions round-robin style directly locally.
4. **Neo4j Transaction Processing (`simple_neo4j_graph.py`)**:
   - `SimpleNeo4jGraph` connects to the DB via basic user/pw credentials.
   - `add_graph_elements`: Cycles nodes executing specific specific `MERGE (n:CLASS {id,gid}) ON CREATE SET n+=props` logic handling node sanitization and duplicates seamlessly at the database layer. Executes relationship inserts merging endpoints exactly on `{id, gid}` filters avoiding cyclic overrides.

### `creat_graph_with_description.py`
**Alternate Builder System**: Mirrors `creat_graph.py`'s objective but operates asynchronously wrapping nano_graphrag logic `extract_entities_with_description`.
- Uses `PROMPTS["entity_extraction"]` pushing records into format splits like `<|>` delimiters.
- Translates `entity_name` and explicit contextual `description` into nodes resolving duplication mathematically instead of structurally.

### `three_layer_import.py`
**Orchestration Engine**: The main conductor loop.
- Establishes connection instance mapping to APOC graph or local fallback.
- Offers `clear_database()` wiping the slate cleanly via Cypher `DETACH DELETE`.
- Sequentially executes `import_layer` invoking iteration loops for bottom, middle, and top txt targets calling `creat_metagraph` per instance ensuring a unique global `str_uuid()` document ID.
- **Trinity Function**: Invokes `create_trinity_links()` combining sets matching Bottom->Middle and Middle->Top graph relationships through the cosine similarity embedding overlap function `utils.ref_link`.

### `utils.py` (Graph Helper Components)
**Graph Tools Support**:
- `_hash_embedding`: A local non-LLM sha-256 numeric hash simulating deterministic standard-normal vectors for testing or fast builds when `--use-remote-embeddings 0`.
- `add_ge_emb`: Fills GraphElement properties interacting with OpenAI API limits locally resolving dimensions vectors assigning them directly inside Neo4j node metadata.
- `merge_similar_nodes`: Triggers Neo4j Apoc plugin routine calling `gds.similarity.cosine` executing collapsing combinations avoiding overlap logic failures.
- `ref_link`: The magic bridge matching bottom ontologies to generic queries running Cypher equations manually defining vector inner products (`reduce(...)`) thresholding >=0.6 binding elements logically with type `REFERENCE`.
- `add_sum`: Summarizes context inputs using LLMs producing `(s:Summary)` tracking `gid` independently.

### `cleangraph.py`
**Database Wiping Tool**: A small standalone script leveraging `neo4j.GraphDatabase.driver` running a single `MATCH (n) DETACH DELETE n` wipe routine targeted rigidly at `bolt://host.docker.internal:7687` for localized cleanup independent of the graph construction pipeline.

---

## Inference & Execution Engine

Translates queries into logic traversing the fully linked knowledge graphs resolving into coherent context blocks for a final medical answer.

### `summerize.py`
**Content Generation Tool**: Used by data loaders to abstract contexts and by inference tools to abstract user queries.
1. Utilizes `tiktoken` analyzing parameter chunks capping at 500 token thresholds.
2. Initiates `ThreadPoolExecutor` parallel mapping invoking LLM completion returning strict semantic classification summaries defining body domains like `BM_RESULT` or `LABORATORY_DATA` extracting core elements cleanly.

### `retrieve.py`
**Primary Search Tool**: Retrieves base GIDs mapping.
1. Examines Summary nodes executing a LLM rating check assigning arbitrary metric integers 4 (very similar) down to 0 mapping similarity to queried prompts.
2. If offline metric applies, falls strictly on standard NumPy `.linalg.norm` and `.dot` products to calculate distance matching.
3. If Summary lacks graph presence, iterates 500 raw graph `entity_nodes` and performs equivalent array search sorting locally preserving ranking order up to `top_k` documents identifiers.

### `post_graph_inference.py`
**Evidence Reasoning Model**: Performs the exhaustive logical hop resolution answering.
1. Invokes `process_chunks` ensuring question is encoded properly.
2. Calls `select_top_gids` extracting the best match context starting points.
3. Expands bounds analyzing adjacent documentation layers utilizing `_neighbor_gids` stepping linearly `[:REFERENCE*1..X]` linking boundaries up to limit.
4. Sweeps Graph Evidence bounds via `_collect_evidence` processing `utils.ret_context` unwinding local connections (node<->node relationship tuples inside subgraph). Combines with `utils.link_context` unpacking macro level definitions returning structured plaintext lines appending them up to `--max-evidence`.
5. Finally aggregates findings mapping variables formatting string variables referencing evidence lines sequentially matching prompt boundaries (`[E1], [E2]`). Either calls a final LLM request or dynamically drops deterministic answers matching top logic outputs to standard stdout printing out inference logs cleanly without fail.

### `run.py`
**Pipeline Switchboard**: Legacy wrapper managing combinations seamlessly.
- Includes URL mapping definitions gracefully downgrading unreadable `bolt` protocol targets.
- Wraps direct execution switches handling `-simple` testing local Nano_graphrag states, `-construct_graph` spinning offline processing directories matching mimic test datasets natively into Neo4j instances, and `-inference` orchestrating the final test checks. Contains comprehensive exception wrapping and logic cleanup printing results clearly to outputs indicating total success/failure metric arrays.
