import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from grakel import Graph
from grakel.kernels import ShortestPath
from sklearn.decomposition import KernelPCA

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


def main():
    # Obtaining all edge/node csv files (should be equally sorted, so glob_edges[0] "==" glob_nodes[0] except edge/node)
    root_path = os.path.abspath(os.curdir)
    pattern = root_path + "/data/Eukaryotes/Eukaryotes_*/"
    levels_of_completeness = sorted(glob.glob(pattern))
    mapping = build_mappings(root_path + "/data/KEGG_Eukaryotes_Taxonomy.csv")

    for level in levels_of_completeness:
        print(f"[*] Level: {level}")
        lvl_of_comp_path = level.strip().split("/")[-2]
        level_edges = sorted(glob.glob(os.path.join(level, "edges*.csv")))
        level_nodes = sorted(glob.glob(os.path.join(level, "nodes*.csv")))
        print(f"[*] Loading graphs.")

        # FIX: skipping empty graphs to avoid Grakel crashing
        # Those graphs are likely graphs with nodes and 0 edges,
        # or 0 shortest paths
        graphs = []
        graph_labels = []
        for e, n in zip(level_edges, level_nodes):
            tmp = load_graphs(e, n)
            if isinstance(tmp, Graph):
                graphs.append(tmp)
                code = os.path.basename(e).replace(".csv", "").split("_")[-1]
                kingdom = mapping.get(code, "Unknown")
                graph_labels.append(kingdom)
            else:
                print(f"[!] Skipped {e}, {n} because either one of them was empty.")

        # Initialize Kernel
        sp = ShortestPath(normalize=True, with_labels=True, algorithm_type="auto")
        print("Fitting...")
        # M_ij represent the similarity between species i and species j
        # diagonals of course do not count...
        M = sp.fit_transform(graphs)

        kernel_PCA = KernelPCA(n_components=2, kernel="precomputed")
        tmp = kernel_PCA.fit_transform(M)

        plt.figure(figsize=(12, 10))
        unique_kingdoms = sorted(list(set(graph_labels)))
        print(f" -> Found Kingdoms: {unique_kingdoms}")
        cmap = plt.colormaps["tab10"]
        kingdom_colors = {k: cmap(i) for i, k in enumerate(unique_kingdoms)}

        for k in unique_kingdoms:
            # Find indices for this kingdom
            idxs = [i for i, val in enumerate(mapping) if val == k]

            # Select the X, Y coordinates for these points
            x_points = tmp[idxs, 0]
            y_points = tmp[idxs, 1]

            plt.scatter(
                x_points,
                y_points,
                color=kingdom_colors[k],  # Explicit color assignment
                label=k,
                s=80,
                alpha=0.8,
                edgecolors="white",  # Adds a white rim so they pop
            )

        plt.title(f"Kingdom Comparison (KPCA)\nCompleteness: {lvl_of_comp_path}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        # Put legend outside to keep plot clean
        plt.legend(title="Kingdom", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        out_file = os.path.join(OUT_DIR, f"{lvl_of_comp_path}_results.txt")
        out_filename = f"{lvl_of_comp_path}_results.png"
        out_graph = os.path.join(OUT_DIR, out_filename)

        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        with open(out_file, "w") as output:
            _ = output.write(str(M))
        plt.savefig(out_graph, dpi=150)
        plt.close()
        print(f"Saved.\n")


def gpt_main():
    root_path = os.path.abspath(os.curdir)
    pattern = root_path + "/data/Eukaryotes/Eukaryotes_*/"
    levels_of_completeness = sorted(glob.glob(pattern))

    # Ensure correct path to taxonomy
    mapping = build_mappings(root_path + "/data/KEGG_Eukaryotes_Taxonomy.csv")

    for level in levels_of_completeness:
        print(f"[*] Level: {level}")
        lvl_of_comp_path = level.strip().split("/")[-2]
        level_edges = sorted(glob.glob(os.path.join(level, "edges*.csv")))
        level_nodes = sorted(glob.glob(os.path.join(level, "nodes*.csv")))
        print(f"[*] Loading graphs.")

        graphs = []
        graph_labels = []  # This matches graphs 1-to-1

        for e, n in zip(level_edges, level_nodes):
            tmp = load_graphs(e, n)
            if isinstance(tmp, Graph):
                graphs.append(tmp)
                code = os.path.basename(e).replace(".csv", "").split("_")[-1]
                kingdom = mapping.get(code, "Unknown")
                graph_labels.append(kingdom)
            else:
                print(f"[!] Skipped {e} (empty).")

        if not graphs:
            print("No graphs found.")
            continue

        # Kernel
        sp = ShortestPath(normalize=True, with_labels=True, algorithm_type="auto")
        print("Fitting...")
        M = sp.fit_transform(graphs)

        # PCA
        kernel_PCA = KernelPCA(n_components=2, kernel="precomputed")
        tmp = kernel_PCA.fit_transform(M)

        # Plotting
        plt.figure(figsize=(12, 10))
        unique_kingdoms = sorted(list(set(graph_labels)))
        print(f" -> Found Kingdoms: {unique_kingdoms}")

        cmap = plt.colormaps["tab10"]
        kingdom_colors = {k: cmap(i) for i, k in enumerate(unique_kingdoms)}

        for k in unique_kingdoms:
            # FIX 1: Enumerate graph_labels, NOT mapping
            idxs = [i for i, val in enumerate(graph_labels) if val == k]

            x_points = tmp[idxs, 0]
            y_points = tmp[idxs, 1]

            plt.scatter(
                x_points,
                y_points,
                color=kingdom_colors[k],
                label=k,
                s=80,
                alpha=0.8,
                edgecolors="white",
            )

        plt.title(f"Kingdom Comparison (KPCA)\nCompleteness: {lvl_of_comp_path}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(title="Kingdom", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        # Output
        out_file = os.path.join(OUT_DIR, f"{lvl_of_comp_path}_results.txt")
        out_graph = os.path.join(OUT_DIR, f"{lvl_of_comp_path}_results.png")

        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)

        # FIX 2: Save full matrix using numpy
        np.savetxt(out_file, M, fmt="%.6f")

        plt.savefig(out_graph, dpi=150)
        plt.close()
        print(f"Saved {out_file} and {out_graph}.\n")


if __name__ == "__main__":
    gpt_main()
