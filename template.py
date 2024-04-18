import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from template_utils import *

sys.setrecursionlimit(6000)


# Task 1: Average degree, number of bridges, number of local bridges (Undirected graph)
def Q1(dataframe):
    # ------------ 1.1 ------------#
    # Average degree = Sum adjacency / Nbr nodes

    # create set for adjacency
    set_adjacency = {}
    # https://stackoverflow.com/questions/16476924/how-can-i-iterate-over-rows-in-a-pandas-dataframe
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

    # sum adjacency
    sum_adjacency = np.sum([len(set_adjacency[node]) for node in set_adjacency])
    # n_neighbours = [len(set_adjacency[node]) for node in set_adjacency]
    # print(sorted(n_neighbours))

    # nbr of nodes
    n_nodes = len(set_adjacency)

    # average degree
    avg_degree = sum_adjacency / n_nodes

    # plot => https://mathinsight.org/degree_distribution
    n_adjacency = [len(set_adjacency[node]) for node in set_adjacency]
    plt.figure(1)
    plt.hist(n_adjacency)
    plt.title("Degree of distribution")
    plt.xlabel("Degree")
    plt.ylabel("Fraction of nodes")
    # plt.savefig("degree_of_distribution.png")
    # plt.show()
    ###############################

    # ------------ 1.2 ------------#
    nbr_bridges = find_bridges(dataframe)
    print("nbr de bridge:", nbr_bridges)
    ###############################

    # ------------ 1.3 ------------#
    # TODO
    ###############################
    return [avg_degree, nbr_bridges, 0]  # [average degree, nb bridges, nb local bridges]


# Task 2: Average similarity score between neighbors (Undirected graph)
def Q2(dataframe):
    # create set for adjacency
    set_adjacency = {}
    # https://stackoverflow.com/questions/16476924/how-can-i-iterate-over-rows-in-a-pandas-dataframe
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

    # compute the similarity score of each pair of nodes
    similarities = []
    for index, row in dataframe.iterrows():
        A = row['Src']
        B = row['Dst']
        # calculate similarity score for each pair of node
        similarities.append(similarity(set_adjacency, A, B))

    # plot percentage
    # turn list of scores into cumulative one : https://stackoverflow.com/questions/15889131/how-to-find-the-cumulative-sum-of-numbers-in-a-list
    cumulative_sum = np.cumsum(sorted(similarities))
    cumulative_percentage = (cumulative_sum / sum(similarities)) * 100
    # print(sorted(similarities))
    # print(cumulative_percentage)

    plt.figure(2)
    plt.plot(sorted(similarities), cumulative_percentage)  # percentage of edges vs similarity score
    plt.title("Cumulative graph of the percentage of edges vs their similarity score")
    plt.xlabel("Similarity score")
    plt.ylabel("Cumulative percentage of edges")
    # plt.savefig("cumulative_graph_of_similarity_score.png")
    # plt.show()

    ###############################
    # compute average similarity score for the network
    return np.sum(similarities) / len(similarities)


# Directed graph
# Task 3: PageRank
def Q3(dataframe):
    pagerank_score = page_rank(dataframe)
    print(pagerank_score)
    print("sum pr:", sum(pagerank_score.values()))
    # https://www.geeksforgeeks.org/python-get-key-with-maximum-value-in-dictionary/
    Idmax = max(pagerank_score, key=lambda x: pagerank_score[x])
    return [Idmax, pagerank_score[Idmax]]
    # the id of the node with the highest pagerank score, the associated pagerank value.
    # Note that we consider that we reached convergence when the sum of the updates on all nodes after one iteration of PageRank is smaller than 10^(-6)


# Undirected graph
# Task 4: Small-world phenomenon
def Q4(dataframe):
    return [0, 0, 0, 0,
            0]  # at index 0 the number of shortest paths of lenght 0, at index 1 the number of shortest paths of length 1, ...
    # Note that we will ignore the value at index 0 as it can be set to 0 or the number of nodes in the graph


# Undirected graph
# Task 5: Betweenness centrality
def Q5(dataframe):
    # Your code here
    return [0,
            0.0]  # the id of the node with the highest betweenness centrality, the associated betweenness centrality value.


# you can write additionnal functions that can be used in Q1-Q5 functions in the file "template_utils.py", a specific place is available to copy them at the end of the Inginious task.

#df = pd.read_csv('powergrid.csv')
df = pd.read_csv('testgrid.csv')
draw_graph(df)
print("Q1", Q1(df))
print("Q2", Q2(df))
#print("Q3", Q3(df))
print("Q4", Q4(df))
print("Q5", Q5(df))
