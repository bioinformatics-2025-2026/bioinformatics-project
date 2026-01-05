from grakel import ShortestPath, graph_from_csv

def main():
    g1 = graph_from_csv(
        edge_files=(['./data/Eukaryotes/Eukaryotes_mc_0/edges_mc_0_aacu.csv'], False, False),
        node_files=(['./data/Eukaryotes/Eukaryotes_mc_0/nodes_mc_0_aacu.csv'], False),
    )
    
    g2 = graph_from_csv(
        edge_files=(['./data/Eukaryotes/Eukaryotes_mc_0/edges_mc_0_aaf.csv'], False, False),
        node_files=(['./data/Eukaryotes/Eukaryotes_mc_0/nodes_mc_0_aaf.csv'], False),
    )


    sp = ShortestPath(
            normalize=True,
            with_labels=True
    )
    
    M = sp.fit_transform(g1)
    K = sp.transform(g2)
    print(M)


if __name__ == "__main__":
    main()
