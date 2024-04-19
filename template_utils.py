# If needed, write here your additional functions/classes with their signature and use them in the exercices:
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
    # pos = nx.nx_agraph.graphviz_layout(G, prog='sfdp')
    # pos = nx.nx_agraph.graphviz_layout(G, prog='twopi')
    # nx.draw(G, pos=pos, arrows=None, with_labels=True, node_size=80, font_size=8)
    # plt.savefig("visual_network.png")
    # plt.show()
    print("Nbr of local bridges:", len(list(nx.local_bridges(G))))
    # PR
    # pr = nx.pagerank(G)
    # print("nx:", pr)
    # print("sum pr:", sum(pr.values()))
    # print("id max PR", max(pr, key=lambda x: pr[x]))


def get_adjacencies(dataframe):
    # https://stackoverflow.com/questions/16476924/how-can-i-iterate-over-rows-in-a-pandas-dataframe
    set_adjacency = {}
    for index, row in dataframe.iterrows():
        # data
        src = row['Src']
        dst = row['Dst']
        # create new adjacency list for each new node
        if src not in set_adjacency:
            set_adjacency[src] = []
        if dst not in set_adjacency:
            set_adjacency[dst] = []
        # fill the lists with the given data on each iteration
        set_adjacency[src].append(dst)
        set_adjacency[dst].append(src)
    return set_adjacency


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


def get_all_nodes(dataframe):
    # Get total nodes in graph
    all_nodes = []
    for _, row in dataframe.iterrows():
        src = row['Src']
        dst = row['Dst']
        if src not in all_nodes:
            all_nodes.append(src)
        if dst not in all_nodes:
            all_nodes.append(dst)
    return all_nodes

# Task 1.2 and 1.3
timer = 0
visited_edges = {}
adjacency = {}


def find_bridges(dataframe):
    # https://cp-algorithms.com/graph/bridge-searching.html
    # get total nodes in graph
    all_nodes = get_all_nodes(dataframe)
    global timer, visited_edges, adjacency
    adjacency = get_adjacencies(dataframe)
    visited_edges = {node: 0 for node in all_nodes}
    intime = {node: -1 for node in all_nodes}
    lowtime = {node: -1 for node in all_nodes}

    # bridges
    nbr_bridges = 0
    for node in all_nodes:
        if not visited_edges[node]:
            search = dfs_bridge(dataframe, node, intime, lowtime)
            nbr_bridges += search[0]

    # local bridges
    nbr_local_bridge = 0
    for src in adjacency:
        for dst in adjacency[src]:
            # take union of neighbours + remove themselves
            intersection = np.intersect1d(adjacency[src], adjacency[dst])
            # remove src and dst from intersections
            if src in intersection :
                index_src = np.where(intersection == src)[0][0]
                intersection = np.delete(intersection, index_src)
            if dst in intersection :
                index_dst = np.where(intersection == dst)[0][0]
                intersection = np.delete(intersection, index_dst)
            # all empty intersection == 1 local bridge
            if len(intersection) == 0 : nbr_local_bridge += 1
    nbr_local_bridge = int(nbr_local_bridge / 2) # only if undirected graph

    return nbr_bridges, nbr_local_bridge


def no_commun(node, child):
    global adjacency
    if len(np.intersect1d(adjacency[node], adjacency[child])) == 0:
        return True
    return False


def dfs_bridge(dataframe, node, intime, lowtime, parent=-1):
    # https://stackoverflow.com/questions/68297463/how-to-find-bridges-in-a-graph-using-dfs
    global timer, visited_edges
    count_bridges = 0
    count_local_bridges = 0
    intime[node] = timer
    lowtime[node] = timer
    timer += 1
    visited_edges[node] = 1
    childs_df = dataframe.loc[(dataframe['Src'] == node) | (dataframe['Dst'] == node)]
    childs_list = childs_df['Src'].unique().tolist() + childs_df['Dst'].unique().tolist()
    # https://www.w3schools.com/python/python_howto_remove_duplicates.asp
    childs_list = list(dict.fromkeys(childs_list))
    for child in childs_list:
        if child == parent:
            continue
        if not visited_edges[child]:
            search = dfs_bridge(dataframe, child, intime, lowtime, node)
            count_bridges += search[0]
            count_local_bridges += search[1]
            lowtime[node] = min(lowtime[node], lowtime[child])
            if intime[node] < lowtime[child]:  # Bridge
                count_bridges += 1
            if intime[node] > lowtime[child] and no_commun(node, child):  # Local Bridge
                count_local_bridges += 1
        else:
            # Check for back edge
            lowtime[node] = min(lowtime[node], intime[child])
            # Local Bridge
            if lowtime[node] == lowtime[child] and intime[node] > intime[child] and no_commun(node, child):
                count_local_bridges += 1
    return count_bridges, count_local_bridges


# Task 3
def page_rank(dataframe, max_iter=100, d=0.85, tol=1e-06):
    # Get total nodes in graph
    all_nodes = get_all_nodes(dataframe)
    N = len(all_nodes)

    pagerank_score = {node: 1.0 / N for node in all_nodes}
    for _ in range(max_iter):
        pr = {}
        conv = 0
        for p in all_nodes:
            # Get B(p) : Set of nodes pointing to node p
            B_p = dataframe.loc[dataframe['Dst'] == p]
            # PageRank score
            pr[p] = (1 - d) / N
            if not B_p.empty:
                for n in B_p.Src.unique():  # Get each node pointing to p
                    # Get Nout_n : number of outgoing links of node n
                    Nout_n = len(dataframe.loc[dataframe['Src'] == n])
                    if Nout_n > 0:
                        pr[p] += d * (pagerank_score[n] / Nout_n)
            conv += abs(pagerank_score[p] - pr[p])
        pagerank_score = pr
        if conv < tol:
            return pagerank_score
    raise StopIteration(max_iter)


# Task 4
def bfs(adjacency_list, start_node):
    # LINFO1121-Algorithmique : BFS
    from collections import deque
    distances = {}
    visited = set([start_node])
    queue = deque([(start_node, 0)])
    while queue:
        # pop a vertex from queue
        vertex, distance = queue.popleft()
        distances[vertex] = distance
        # mark the not yet visited nodes as visited + enqueue
        for neighbour in adjacency_list[vertex]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, distance + 1))
    return distances
