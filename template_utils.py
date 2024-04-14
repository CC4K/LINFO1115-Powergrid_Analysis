# If needed, write here your additional fuctions/classes with their signature and use them in the exercices:
# a specific place is available to copy them at the end of the Inginious task.

# First, import the libraries needed for your helper functions
import numpy as np
import pandas as pd


# Then write the classes and/or functions you wishes to use in the exercises
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
    return common/total


def find_bridge(dataframe, visited, intime, lowtime):
    for i in range(len(dataframe)):
        src = dataframe.iloc[i, 0]
        dest = dataframe.iloc[i, 1]
        if not visited[dataframe.index[(dataframe['Src'] == src) & (dataframe['Dst'] == dest)]]:
            # TODO reste
            return 0
    return 0
