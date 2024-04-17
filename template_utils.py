# If needed, write here your additional fuctions/classes with their signature and use them in the exercices:
# a specific place is available to copy them at the end of the Inginious task.

# First, import the libraries needed for your helper functions
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Then write the classes and/or functions you wishes to use in the exercises
def draw_graph(dataframe):
    G = nx.Graph()
    for index, row in dataframe.iterrows():
        src = row['Src']
        dst = row['Dst']
        G.add_edge(src, dst)
    plt.figure(0, figsize=(16, 16))
    pos = nx.nx_agraph.graphviz_layout(G, prog='sfdp')
    # pos = nx.nx_agraph.graphviz_layout(G, prog='twopi')
    nx.draw(G, pos=pos, arrows=None, with_labels=True, node_size=80, font_size=8)
    # plt.savefig("visual_network.png")
    # plt.show()


def similarity(set_adjacency, A, B):
    # number of common neighbours between A and B
    common = len(np.intersect1d(set_adjacency[A], set_adjacency[B]))
    if common == 0: return 0
    # total number of different neighbours of A and B
    total_list = np.union1d(set_adjacency[A], set_adjacency[B])
    total = len(total_list)
    if A in total_list: total -= 1
    if B in total_list: total -= 1
    # calculate similarity
    return common / total


def find_bridge(dataframe, visited, intime, lowtime):
    for i in range(len(dataframe)):
        src = dataframe.iloc[i, 0]
        dest = dataframe.iloc[i, 1]
        if not visited[dataframe.index[(dataframe['Src'] == src) & (dataframe['Dst'] == dest)]]:
            # TODO reste
            return 0
    return 0


def page_rank(dataframe, p, d=0.85):
    # Get total nodes in graph
    total_nodes = []
    for index, row in dataframe.iterrows():
        src = row['Src']
        dst = row['Dst']
        if src not in total_nodes:
            total_nodes.append(src)
        if dst not in total_nodes:
            total_nodes.append(dst)
    N = len(total_nodes)
    # Get B(p) : Set of nodes pointing to node p
    B_p = dataframe.loc[dataframe['Dst'] == p]

    # Get Nout_n : number of outgoing links of node n
    Nout_n = len(dataframe.loc[dataframe['Src'] == p])

    # PageRank score
    somme = 0
    for node in B_p:
        somme += page_rank(dataframe, node)
    return ((1 - d) / N) + d * (somme / Nout_n)
