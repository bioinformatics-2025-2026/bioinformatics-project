import glob
import os

import pandas as pd
from grakel import Graph
from grakel.kernels import ShortestPath


def build_mapping(taxonomy_file_path: str) -> dict[str, str]:
    # reads the taxonomy file and skips the first row (malformed?)
    df = pd.read_csv(taxonomy_file_path, skiprows=0)
    df.columns = df.columns.str.strip()
    # TODO: save name and taxonomy_2 (kingdom level) as a list[str] with the same
    # code key
    mapping = dict(
        zip(df["code"].astype(str).str.strip(), df["name"].astype(str).str.strip())
    )
    return mapping


def load_graphs(edge_file: str, node_file: str) -> Graph:
    edges = pd.read_csv(edge_file)
    nodes = pd.read_csv(node_file)

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
    pattern = root_path + "/data/Eukaryotes/*/"
    glob_edges = sorted(glob.glob(pattern + "edges*", recursive=True))
    glob_nodes = sorted(glob.glob(pattern + "nodes*", recursive=True))
    mapping = build_mapping(root_path + "/data/KEGG_Eukaryotes_Taxonomy.csv")

    # limit = 100
    # subset_edges = glob_edges[:limit]
    # subset_nodes = glob_nodes[:limit]
    # graphs = [load_graphs(e, n) for e, n in zip(subset_edges, subset_nodes)]
    #
    # # Initialize Kernel
    # sp = ShortestPath(normalize=True, with_labels=True, algorithm_type="auto")
    #
    # print("Fitting...")
    # M = sp.fit_transform(graphs)
    # # K_ij represent the similarity between species i and species j
    # # diagonals do not count...
    # print(f"Success. Matrix shape: {M.shape}")
    # print(M)
    # with open("results.txt", "w+") as output:
    #     output.write("Level of completeness:")
    #     pass


if __name__ == "__main__":
    main()
