import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from grakel import Graph
from grakel.kernels import ShortestPath

OUT_DIR = "out"


def build_mappings(taxonomy_file_path: str) -> dict[str, str]:
    # reads the taxonomy file and skips the first row
    df = pd.read_csv(taxonomy_file_path, skiprows=0)
    df.columns = df.columns.str.strip()
    codes = df["code"].astype(str).str.strip()
    kingdoms = df["taxon_2"].astype(str).str.strip()

    # {'hsa': 'Animals', ...}
    mapping = dict(zip(codes, kingdoms))
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
        adj[s][d] = float(row.get("label", 1.0))

    # 2. Build Node Labels: {node_id: label_string}
    node_labels = dict(zip(nodes["id"].astype(str), nodes["label"]))

    return Graph(adj, node_labels=node_labels)


def save_clustermap(M, kingdom, level, out_dir):
    # Clustermap reorders the matrix to find blocks of similarity
    # cmap="viridis" or "mako" looks scientific.
    # xticklabels=False keeps it clean (names are unreadable at this scale anyway)
    try:
        g = sns.clustermap(
            M,
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
            figsize=(10, 10),
            dendrogram_ratio=0.15,
        )
        g.fig.suptitle(f"Intra-Kingdom Similarity: {kingdom} ({level})", y=1.02)

        filename = f"{level}_{kingdom}_heatmap.png"
        g.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close()
    except Exception as e:
        # Fails if matrix is 1x1 or singular
        print(f"[!] Could not plot heatmap for {kingdom}: {e}")


def main():
    root_path = os.path.abspath(os.curdir)
    pattern = root_path + "/data/Eukaryotes/Eukaryotes_*/"
    levels_of_completeness = sorted(glob.glob(pattern))

    taxonomy_mapping = build_mappings(root_path + "/data/KEGG_Eukaryotes_Taxonomy.csv")

    for level in levels_of_completeness:
        print(f"[*] Level: {level}.")
        lvl_of_comp_path = level.strip().split("/")[-2]
        level_edges = sorted(glob.glob(os.path.join(level, "edges*.csv")))
        level_nodes = sorted(glob.glob(os.path.join(level, "nodes*.csv")))
        print("[*] Loading graphs...")

        graphs = {}
        kingdoms = set()

        for e, n in zip(level_edges, level_nodes):
            graph = load_graphs(e, n)
            if isinstance(graph, Graph):
                code = os.path.basename(e).replace(".csv", "").split("_")[-1]
                kingdom = taxonomy_mapping.get(code, "Unknown")
                if kingdom not in graphs:
                    graphs[kingdom] = list()
                    kingdoms.add(kingdom)
                graphs[kingdom].append(graph)
            else:
                print(f"[!] Skipped {e} (empty).")

        for kingdom in kingdoms:
            # Kernel
            print(f"[*] Computing similarity matrix for {kingdom}...")
            sp = ShortestPath(normalize=True, with_labels=True, algorithm_type="auto")
            M = sp.fit_transform(graphs[kingdom])

            # Output
            out_file = os.path.join(
                OUT_DIR, f"{lvl_of_comp_path}_{kingdom}_similarity_matrix.txt"
            )

            if not os.path.exists(OUT_DIR):
                os.makedirs(OUT_DIR)
            # Saving complete similarity matrix
            np.savetxt(out_file, M, fmt="%.6f")
            print(f"[*] Generating clustermap for {kingdom}...")
            save_clustermap(M, kingdom, lvl_of_comp_path, OUT_DIR)


if __name__ == "__main__":
    main()
