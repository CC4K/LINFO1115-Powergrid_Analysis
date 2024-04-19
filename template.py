import sys

import matplotlib.pyplot as plt
import pandas as pd
from template_utils import *

sys.setrecursionlimit(6000)


# Undirected graph
# Task 1: Average degree, number of bridges, number of local bridges
def Q1(dataframe):
    # 1.1
    # Average degree = Sum adjacency / Nbr nodes

    # create set for adjacency
    set_adjacency = get_adjacencies(dataframe)

    # sum adjacency
    sum_adjacency = np.sum([len(set_adjacency[node]) for node in set_adjacency])
    # n_neighbours = [len(set_adjacency[node]) for node in set_adjacency]
    # print(sorted(n_neighbours))

    # nbr of nodes
    n_nodes = len(set_adjacency)

    # average degree
    avg_degree = sum_adjacency / n_nodes

    #-------------------------------------#
    # plot => https://mathinsight.org/degree_distribution
    n_adjacency = [len(set_adjacency[node]) for node in set_adjacency]
    plt.figure(figsize=(8, 6))
    plt.hist(n_adjacency)
    plt.grid(axis='y')
    plt.title("Degree of distribution")
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    # plt.savefig("degree_of_distribution.png")
    #-------------------------------------#

    # 1.2 and 1.3
    bridges = find_bridges(dataframe)
    nbr_bridges = bridges[0]
    nbr_local_bridges = bridges[1]

    return [avg_degree, nbr_bridges, nbr_local_bridges]  # [average degree, nb bridges, nb local bridges]


# Undirected graph
# Task 2: Average similarity score between neighbors
def Q2(dataframe):
    # create set for adjacency
    set_adjacency = get_adjacencies(dataframe)

    # compute the similarity score of each pair of nodes
    similarities = []
    for index, row in dataframe.iterrows():
        A = row['Src']
        B = row['Dst']
        # calculate similarity score for each pair of node
        similarities.append(similarity(set_adjacency, A, B))

    #-------------------------------------#
    # plot
    # turn list of scores into cumulative one : https://stackoverflow.com/questions/15889131/how-to-find-the-cumulative-sum-of-numbers-in-a-list
    cumulative_sum = np.cumsum(sorted(similarities))
    cumulative_percentage = (cumulative_sum / sum(similarities)) * 100
    # print(sorted(similarities))
    # print(cumulative_percentage)

    plt.figure(figsize=(8, 6))
    plt.plot(sorted(similarities), cumulative_percentage)  # percentage of edges vs similarity score
    plt.xlim(0)
    plt.ylim((0, 100))
    plt.grid()
    plt.title("Cumulative graph of the percentage of edges vs their similarity score")
    plt.xlabel("Similarity score")
    plt.ylabel("Cumulative percentage of edges")
    # plt.savefig("cumulative_graph_of_similarity_score.png")
    #-------------------------------------#
    # compute average similarity score for the network
    return np.sum(similarities) / len(similarities)


# Directed graph
# Task 3: PageRank
def Q3(dataframe):
    pagerank_score = page_rank(dataframe)
    # print(pagerank_score)
    # print("sum pr:", sum(pagerank_score.values()))
    # https://www.geeksforgeeks.org/python-get-key-with-maximum-value-in-dictionary/
    Idmax = max(pagerank_score, key=lambda x: pagerank_score[x])
    return [Idmax, pagerank_score[Idmax]]
    # the id of the node with the highest pagerank score, the associated pagerank value.
    # Note that we consider that we reached convergence when the sum of the updates on all nodes after one iteration of PageRank is smaller than 10^(-6)


# Undirected graph
# Task 4: Small-world phenomenon
def Q4(dataframe):
    # => measure the distance of the shortest path between each pair of nodes

    # create set for adjacency
    set_adjacency = get_adjacencies(dataframe)

    # compute shortest path between each pair of nodes (BFS ?) and put in dictionary
    shortest_paths = {}
    for node in set_adjacency: shortest_paths[node] = bfs(set_adjacency, node)

    # extract path lengths from dictionary
    path_lengths = []
    for big_dict in shortest_paths.values():
        for values in big_dict.values(): path_lengths.append(values)

    # return number of each length of shortest paths
    ret = [path_lengths.count(length) for length in range(np.max(path_lengths)+1)]


    # diameter and most common shortest distance
    diameter = np.max(path_lengths)
    # print(diameter) # 46
    commonest_dist = ret.index(np.max(ret))
    # print(commonest_dist) # 19

    #-------------------------------------#
    # plot => LINFO1115-course1.pdf, pg 34
    plt.figure(figsize=(8, 6))
    x = np.arange(0, diameter+1)
    plt.plot(x, ret, '-bo', markersize=5)
    plt.plot(commonest_dist, ret[commonest_dist], 'go', markersize=5, label="Most common shortest distance")
    plt.plot(diameter, ret[diameter], 'ro', markersize=5, label="Longest shortest distance")
    plt.xlim(0)
    plt.ylim(0)
    plt.legend()
    plt.grid()
    plt.title("Graph of the number of paths having a given distance")
    plt.xlabel("Number of intermediaries")
    plt.ylabel("Number of paths")
    # plt.savefig("small-world_phenomenon.png")
    #-------------------------------------#
    return ret  # ret[0] = nbr shortest paths of length 0, ret[1] = nbr shortest paths of length 1, ...
    # Note that we will ignore the value at index 0 as it can be set to 0 or the number of nodes in the graph


# Undirected graph
# Task 5: Betweenness centrality
def Q5(dataframe):
    # TODO
    return [0, 0.0]  # the id of the node with the highest betweenness centrality, the associated betweenness centrality value.


# you can write additional functions that can be used in Q1-Q5 functions in the file "template_utils.py", a specific place is available to copy them at the end of the Inginious task.

df = pd.read_csv('powergrid.csv')
#df = pd.read_csv('testgrid.csv')
print("Q1", Q1(df))
print("Q2", Q2(df))
print("Q3", Q3(df))
print("Q4", Q4(df))
print("Q5", Q5(df))
draw_graph(df)
# plt.show()
