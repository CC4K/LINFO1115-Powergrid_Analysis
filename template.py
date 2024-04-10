import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from template_utils import *

sys.setrecursionlimit(6000)


# Undirected graph
# Task 1: Average degree, number of bridges, number of local bridges
def Q1(dataframe):
    ####### 1.1 #######
    # Get nodes neighbour
    maxi = max(dataframe.Src.max(), dataframe.Dst.max())
    nodes_neighbour_count = np.zeros((maxi, 2))
    for i in range(len(dataframe)):
        src = dataframe.iloc[i, 0] - 1
        dest = dataframe.iloc[i, 1] - 1
        # Src has a neighbour
        nodes_neighbour_count[src, 0] += 1
        # Dst has a neighbour
        nodes_neighbour_count[dest, 1] += 1
    # Get sum of neighbours
    average = np.sum(nodes_neighbour_count, axis=0)[1]
    average = average / maxi
    # Plot
    plt.hist(nodes_neighbour_count[:, 1])
    plt.title("Degree of distribution")
    plt.xlabel("Number of neighbours")
    plt.ylabel("Number of nodes having x neighbours")
    plt.show()
    #######################

    ###### 1.2 #########
    visited_edges = np.zeros((len(dataframe),))
    find_bridge(dataframe, visited_edges, [], [])
    ####################

    ###### 1.3 #########

    ####################
    return [average, 0, 0]  # [average degree, nb bridges, nb local bridges]


# Undirected graph
# Task 2: Average similarity score between neighbors
def Q2(dataframe):
    score = 0
    for i in range(len(dataframe)):
        src = dataframe.iloc[i, 0]
        dest = dataframe.iloc[i, 1]
        score += neighbourhood_overlap(dataframe, src, dest)
    # TODO plot cumulative graph of the percentage of edges vs the similarity score.
    return score / len(dataframe)  # the average similarity score between neighbors


# Directed graph
# Task 3: PageRank
def Q3(dataframe):
    # Your code here
    return [0, 0.0]  # the id of the node with the highest pagerank score, the associated pagerank value.
    # Note that we consider that we reached convergence when the sum of the updates on all nodes after one iteration of PageRank is smaller than 10^(-6)


# Undirected graph
# Task 4: Small-world phenomenon
def Q4(dataframe):
    # Your code here
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

df = pd.read_csv('powergrid.csv')
print("Q1", Q1(df))
# print("Q2", Q2(df))
print("Q3", Q3(df))
print("Q4", Q4(df))
print("Q5", Q5(df))
