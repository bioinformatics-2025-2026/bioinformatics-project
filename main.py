import glob
import os

import pandas as pd
from grakel import Graph
from grakel.kernels import ShortestPath


def build_mapping(taxonomy_file_path: str) -> dict[str, str]:
    # reads the taxonomy file and skips the first row
    df = pd.read_csv(taxonomy_file_path, skiprows=0)
    df.columns = df.columns.str.strip()
    # TODO: save name and taxonomy_2 (kingdom level) as a list[str] with the same
    # code key
    mapping = dict(
        zip(df["code"].astype(str).str.strip(), df["name"].astype(str).str.strip())
    )
    return mapping


def load_graphs(edge_file: str, node_file: str) -> Graph | None:
    edges = pd.read_csv(edge_file)
    nodes = pd.read_csv(node_file)

    if edges.empty or nodes.empty:
        return None
    # 1. Build Adjacency: {src: {dst: weight}}
    # Force IDs to string to match node labels keys
    adj = {str(n): {} for n in nodes["id"]}

    for _, row in edges.iterrows():
        s, d = str(row["src"]), str(row["dst"])
        adj[s][d] = float(row["label"])

    # 2. Build Node Labels: {node_id: label_string}
    node_labels = dict(zip(nodes["id"].astype(str), nodes["label"]))

    return Graph(adj, node_labels=node_labels)


def main():
    # Obtaining all edge/node csv files (should be equally sorted, so glob_edges[0] "==" glob_nodes[0] except edge/node)
    root_path = os.path.abspath(os.curdir)
    pattern = root_path + "/data/Eukaryotes/Eukaryotes_*/"
    levels_of_completeness = sorted(glob.glob(pattern))
    mapping = build_mapping(root_path + "/data/KEGG_Eukaryotes_Taxonomy.csv")

    for level in levels_of_completeness:
        print(f"[*] Level: {level}")
        lvl_of_comp_path = level.strip().split("/")[-2]
        level_edges = sorted(glob.glob(os.path.join(level, "edges*.csv")))
        level_nodes = sorted(glob.glob(os.path.join(level, "nodes*")))
        print(f"[*] Loading graphs of {level}")
        # FIX: skipping empty graphs to avoid Grakel crashing
        # Those graphs are likely graphs with nodes and 0 edges,
        # or 0 shortest paths
        graphs = []
        for e, n in zip(level_edges, level_nodes):
            tmp = load_graphs(e, n)
            if isinstance(tmp, Graph):
                graphs.append(tmp)
            else:
                print(f"[!] Skipped {e}, {n} because either one of them was empty.")

        # Initialize Kernel
        sp = ShortestPath(normalize=True, with_labels=True, algorithm_type="auto")
        print("Fitting...")
        M = sp.fit_transform(graphs)
        # K_ij represent the similarity between species i and species j
        # diagonals of course do not count...
        print(f"Success.")
        print(M)
        out_file = f"/out/{lvl_of_comp_path}_results.txt"
        with open(out_file, "w+") as output:
            pass


if __name__ == "__main__":
    main()
