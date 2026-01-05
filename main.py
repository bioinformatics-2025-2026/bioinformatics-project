from grakel import RandomWalkLabeled, graph_from_csv

def main():
    g1 = graph_from_csv(
        edge_files=(['./data/Eukaryotes/Eukaryotes_mc_0/edges_mc_0_aacu.csv'], False, False),
        node_files=(['./data/Eukaryotes/Eukaryotes_mc_0/nodes_mc_0_aacu.csv'], False),
    )
    
    g2 = graph_from_csv(
        edge_files=(['./data/Eukaryotes/Eukaryotes_mc_0/edges_mc_0_aaf.csv'], False, False),
        node_files=(['./data/Eukaryotes/Eukaryotes_mc_0/nodes_mc_0_aaf.csv'], False),
    )


    random_walk = RandomWalkLabeled(
        lamda=1,
        method_type='fast',
        kernel_type='geometric',
        p=1,
    )
    
    random_walk.fit_transform(g1)
    random_walk.transform(g2)



if __name__ == "__main__":
    main()
