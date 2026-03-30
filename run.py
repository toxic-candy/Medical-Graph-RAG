import os
import traceback
from urllib.parse import urlparse
from camel.storages import Neo4jGraph
from simple_neo4j_graph import SimpleNeo4jGraph
from dataloader import load_high
import argparse
from creat_graph import creat_metagraph
from summerize import process_chunks
from retrieve import seq_ret
from utils import *

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-simple', action='store_true')
parser.add_argument('-construct_graph', action='store_true')
parser.add_argument('-inference',  action='store_true')
parser.add_argument('-grained_chunk',  action='store_true')
parser.add_argument('-trinity', action='store_true')
parser.add_argument('-trinity_gid1', type=str)
parser.add_argument('-trinity_gid2', type=str)
parser.add_argument('-ingraphmerge',  action='store_true')
parser.add_argument('-crossgraphmerge', action='store_true')
parser.add_argument('-dataset', type=str, default='mimic_ex')
parser.add_argument('-data_path', type=str, default='./dataset_test')
parser.add_argument('-test_data_path', type=str, default='./dataset_ex/report_0.txt')
args = parser.parse_args()


def _normalize_neo4j_url(raw_url: str | None) -> str:
    # Accept common env inputs and coerce browser/HTTP URLs to Bolt.
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
        # Allow host:port without scheme.
        if url.endswith(":7474"):
            url = url[:-5] + ":7687"
        return f"bolt://{url}"

    return url

if args.simple:
    from nano_graphrag import GraphRAG, QueryParam
    graph_func = GraphRAG(working_dir="./nanotest")

    with open("./dataset_ex/report_0.txt") as f:
        graph_func.insert(f.read())

    # Perform local graphrag search (I think is better and more scalable one)
    print(graph_func.query("What is the main symptom of the patient?", param=QueryParam(mode="local")))

else:

    url = _normalize_neo4j_url(os.getenv("NEO4J_URL") or os.getenv("NEO4J_URI"))
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        raise ValueError(
            "NEO4J_PASSWORD is not set. Export NEO4J_PASSWORD before running graph construction."
        )

    # Set Neo4j instance. Fallback to non-APOC client when needed.
    try:
        n4j = Neo4jGraph(
            url=url,
            username=username,
            password=password,
        )
    except Exception as e:
        print(f"Neo4jGraph init failed ({e}). Falling back to SimpleNeo4jGraph without APOC.")
        n4j = SimpleNeo4jGraph(
            url=url,
            username=username,
            password=password,
        )

    if args.construct_graph: 
        if args.dataset == 'mimic_ex':
            files = sorted(
                [
                    file
                    for file in os.listdir(args.data_path)
                    if os.path.isfile(os.path.join(args.data_path, file))
                ]
            )
            total = len(files)
            failed_files = []

            print(f"Starting graph construction for {total} files from {args.data_path}", flush=True)
            for idx, file_name in enumerate(files, start=1):
                print(f"[{idx}/{total}] Processing {file_name}", flush=True)
                file_path = os.path.join(args.data_path, file_name)
                try:
                    content = load_high(file_path)
                    gid = str_uuid()
                    n4j = creat_metagraph(args, content, gid, n4j)
                    print(f"[{idx}/{total}] Done {file_name} gid={gid}", flush=True)
                except Exception as e:
                    failed_files.append(file_name)
                    print(f"[{idx}/{total}] Failed {file_name}: {e}", flush=True)
                    traceback.print_exc()
                    continue

                if args.trinity:
                    link_context(n4j, args.trinity_gid1)
            if args.crossgraphmerge:
                merge_similar_nodes(n4j, None)

            print(
                f"Graph construction finished. Succeeded: {total - len(failed_files)} / {total}. "
                f"Failed: {len(failed_files)}",
                flush=True,
            )
            if failed_files:
                print("Failed files:", flush=True)
                for f in failed_files:
                    print(f" - {f}", flush=True)

    if args.inference:
        question = load_high("./prompt.txt")
        sum = process_chunks(question)
        gid = seq_ret(n4j, sum)
        response = get_response(n4j, gid, question)
        print(response)
