# If needed, write here your additional fuctions/classes with their signature and use them in the exercices:
# a specific place is available to copy them at the end of the Inginious task.

# First, import the libraries needed for your helper functions
import numpy as np
import pandas as pd


# Then write the classes and/or functions you wishes to use in the exercises
def neighbourhood_overlap(dataframe, A, B, r=False):
    new_df = dataframe.loc[
        (dataframe['Src'] == A) | (dataframe['Src'] == B) | (dataframe['Dst'] == A) | (dataframe['Dst'] == B)]
    visited = {'A': [], 'B': []}
    for i in range(len(new_df)):
        src = new_df.iloc[i, 0]
        dest = new_df.iloc[i, 1]
        if src not in [A, B]:
            if dest == A and src not in visited['A']:
                visited['A'].append(src)
            elif dest == B and src not in visited['B']:
                visited['B'].append(src)
        elif dest not in [A, B]:
            if src == A and src not in visited['A']:
                visited['A'].append(dest)
            elif src == B and src not in visited['B']:
                visited['B'].append(dest)
    num = len(np.intersect1d(visited['A'], visited['B']))
    denom = len(new_df)
    if r:
        print("A : ", A, " B : ", B)
        print(new_df)
        print("Nombre de voisins commun :", num)
        print("Nombre de voisins A et/ou B :", denom)
    score = num / denom
    return score


def find_bridge(dataframe, visited, intime, lowtime):
    for i in range(len(dataframe)):
        src = dataframe.iloc[i, 0]
        dest = dataframe.iloc[i, 1]
        if not visited[dataframe.index[(dataframe['Src'] == src) & (dataframe['Dst'] == dest)]]:
            # TODO reste
            return 0
    return 0
